# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)
from vllm.platforms import current_platform

from ..base import MMLinearLayerConfig


@dataclass
class Int8ScaledMMLinearLayerConfig(MMLinearLayerConfig):
    # TODO: Change to QuantKey like FP8ScaledMMLinearLayerConfig
    is_static_input_scheme: bool
    is_channelwise: bool
    input_symmetric: bool


@dataclass
class FP8ScaledMMLinearLayerConfig(MMLinearLayerConfig):
    weight_quant_key: QuantKey
    activation_quant_key: QuantKey
    weight_shape: tuple[int, int]
    input_dtype: torch.dtype
    out_dtype: torch.dtype


_FP8ParamsT = tuple[
    torch.Tensor,  # weight
    torch.Tensor,  # weight_scale
    torch.Tensor | None,  # input_scale,
    torch.Tensor | None,  # input_scale_ub,
]
_Int8ParamsT = tuple[
    torch.Tensor,  # weight
    torch.Tensor,  # weight_scale
    torch.Tensor | None,  # input_scale,
    torch.Tensor | None,  # input_zp
    torch.Tensor | None,  # azp_adj
]

_ParamsT = TypeVar("_ParamsT", _Int8ParamsT, _FP8ParamsT)
_ConfigT = TypeVar("_ConfigT", bound=MMLinearLayerConfig)


class ScaledMMLinearKernel(Generic[_ConfigT, _ParamsT], ABC):
    @classmethod
    @abstractmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(cls, c: _ConfigT) -> tuple[bool, str | None]:
        raise NotImplementedError

    def __init__(self, c: _ConfigT, layer_param_names: Sequence[str]) -> None:
        assert self.can_implement(c)[0]
        assert self.is_supported()[0]
        self.config = c
        self.layer_param_names = layer_param_names

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    # return a covariant type in the subclass
    @abstractmethod
    def _get_layer_params(self, layer) -> _ParamsT:
        raise NotImplementedError


class FP8ScaledMMLinearKernel(
    ScaledMMLinearKernel[FP8ScaledMMLinearLayerConfig, _FP8ParamsT], ABC
):
    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        act_scale_descriptor = c.activation_quant_key.scale
        self.quant_fp8 = QuantFP8(
            static=act_scale_descriptor.static,
            group_shape=act_scale_descriptor.group_shape,
            num_token_padding=self.get_output_padding(),
        )
        self.fp8_dtype = current_platform.fp8_dtype()
        super().__init__(c, layer_param_names)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def _get_layer_params(self, layer) -> _FP8ParamsT:
        w, w_s, x_s, x_s_ub = self.layer_param_names
        return (
            getattr(layer, w),
            getattr(layer, w_s),
            getattr(layer, x_s, None),
            getattr(layer, x_s_ub, None),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fp8_dtype = self.fp8_dtype
        maybe_out_dtype = self.config.out_dtype
        w, w_s, x_s, x_s_ub = self._get_layer_params(layer)

        #   ops.scaled_fp8_quant supports both dynamic and static quant.
        #   If dynamic, layer.input_scale is None and x_s computed from x.
        #   If static, layer.input_scale is scalar and x_s is input_scale.
        # View input as 2D matrix for fp8 methods
        x_2d = x.view(-1, x.shape[-1])
        output_shape = [*x.shape[:-1], w.shape[1]]
        out_dtype = x.dtype if maybe_out_dtype is None else maybe_out_dtype

        # If input not quantized
        # TODO(luka) remove this path if not used anymore
        x_2d_q = x_2d
        if x.dtype != fp8_dtype:
            x_2d_q, x_s = self.quant_fp8(
                x_2d,
                x_s,
                x_s_ub,
            )
        # Low-latency FP8 GEMM dispatch for small M (autotuned)
        if (hasattr(layer, '_use_ll_gemm_fp8') and layer._use_ll_gemm_fp8
                and x_2d_q.shape[0] <= 8):
            from vllm.model_executor.layers.fused_moe.router.ll_a_gemm import (
                _get_compiled_splitk,
            )
            from cuda.bindings.driver import CUstream
            from torch.cuda import current_stream

            M = x_2d_q.shape[0]
            x8 = x_2d_q.view(torch.bfloat16)
            w8 = w.T.view(torch.bfloat16)  # zero copy: .t() is metadata-only
            N = w8.shape[0]
            K_view = w8.shape[1]

            # Fix tight strides for M=1
            if M == 1 and x8.stride(0) != x8.shape[1]:
                buf = torch.empty_like(x8)
                buf.copy_(x8)
                x8 = buf

            # Autotuned config selection based on N and K
            tiles = K_view // 256
            if N <= 256 and tiles >= 8:
                split_k, ns = 8, min(3, tiles // 8)
            elif N <= 1536 and tiles >= 4:
                split_k, ns = 4, min(4, tiles // 4)
            elif N <= 3072 and tiles >= 4:
                split_k, ns = 4, min(2, tiles // 4)
            else:
                split_k = 0  # fall through to scaled_mm

            if split_k > 0 and tiles % split_k == 0:
                combined_scale = (x_s * w_s).item()
                out_buf = torch.empty(N, M, dtype=torch.bfloat16, device=x8.device)
                compiled = _get_compiled_splitk(True, True, w8, x8, out_buf, split_k, ns)
                stream = CUstream(current_stream().cuda_stream)
                compiled(w8, x8, out_buf, stream, combined_scale)
                out = out_buf.T
                if bias is not None:
                    out = out + bias
                return out.view(*output_shape)

        return self.apply_scaled_mm(
            A=x_2d_q,
            B=w,
            out_dtype=out_dtype,
            As=x_s,
            Bs=w_s,
            bias=bias,
            output_shape=output_shape,
        )

    @abstractmethod
    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_output_padding(self) -> int | None:
        return None


class Int8ScaledMMLinearKernel(
    ScaledMMLinearKernel[Int8ScaledMMLinearLayerConfig, _Int8ParamsT], ABC
):
    def _get_layer_params(self, layer) -> _Int8ParamsT:
        w_q, w_s, i_s, i_zp, azp_adj = self.layer_param_names
        return (
            getattr(layer, w_q),
            getattr(layer, w_s),
            getattr(layer, i_s, None),
            getattr(layer, i_zp, None),
            getattr(layer, azp_adj, None),
        )
