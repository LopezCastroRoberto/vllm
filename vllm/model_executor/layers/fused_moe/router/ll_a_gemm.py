# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def is_available() -> bool:
    try:
        import cutlass.cute  # noqa: F401
        return True
    except ImportError:
        return False


# Cache: (is_fp8, swapped) -> compiled callable
_compiled_cache: dict[tuple, object] = {}


def _get_compiled(is_fp8: bool, swapped: bool, a, b, c):
    """Get or compile an A GEMM kernel."""
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from cutlass.cute.runtime import from_dlpack
    from torch.cuda import current_stream

    from ._ll_a_gemm_kernels import LLAGemm

    K = a.shape[1]
    ns = 12 if K >= 4096 else 4
    cache_key = (is_fp8, swapped, ns)
    if cache_key in _compiled_cache:
        return _compiled_cache[cache_key]

    div = 8
    # For swapped path, output C=[N,M] has small M — relax divisibility
    b_div = div  # B (activations) K dim always divisible by 8
    c_div = 1 if swapped else div  # C mode 1 = M (can be 1-8)

    mA = (from_dlpack(a, assumed_align=16, enable_tvm_ffi=True)
          .mark_layout_dynamic(leading_dim=1)
          .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1),
                                      divisibility=div))
    mB = (from_dlpack(b, assumed_align=16, enable_tvm_ffi=True)
          .mark_layout_dynamic(leading_dim=1)
          .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1),
                                      divisibility=b_div))
    mC = (from_dlpack(c, assumed_align=16, enable_tvm_ffi=True)
          .mark_layout_dynamic(leading_dim=1)
          .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1),
                                      divisibility=c_div))

    tk = 256
    tn = 8 if swapped else 16
    gemm = LLAGemm(tile_n=tn, tile_k=tk, num_stages=ns,
                    num_dma_warps=4, is_fp8=is_fp8)
    stream = CUstream(current_stream().cuda_stream)
    compiled = cute.compile(gemm, mA, mB, mC, stream,
                            options="--enable-tvm-ffi")
    _compiled_cache[cache_key] = compiled
    logger.debug("Compiled ll_a_gemm: is_fp8=%s swapped=%s tile_n=%d",
                 is_fp8, swapped, tn)
    return compiled


def ll_a_gemm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    is_fp8: bool = False,
    scale: float = 1.0,
) -> torch.Tensor:
    from cuda.bindings.driver import CUstream
    from torch.cuda import current_stream

    M = hidden_states.shape[0]
    N = weight.shape[0]

    if M <= 8:
        out_NM = torch.empty(N, M, dtype=torch.bfloat16,
                             device=hidden_states.device)
        compiled = _get_compiled(is_fp8, True, weight, hidden_states, out_NM)
        stream = CUstream(current_stream().cuda_stream)
        compiled(weight, hidden_states, out_NM, stream, scale)
        return out_NM.T 
    else:
        output = torch.empty(M, N, dtype=torch.bfloat16,
                             device=hidden_states.device)
        compiled = _get_compiled(is_fp8, False, hidden_states, weight, output)
        stream = CUstream(current_stream().cuda_stream)
        compiled(hidden_states, weight, output, stream, scale)
        return output


# Split-K compiled kernel cache
_splitk_cache: dict[tuple, object] = {}


def _get_compiled_splitk(is_fp8: bool, swapped: bool, a, b, c, split_k: int, num_stages: int = 0):
    """Compile split-K kernel variant."""
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from cutlass.cute.runtime import from_dlpack
    from torch.cuda import current_stream

    from ._ll_a_gemm_kernels import LLAGemm

    K = a.shape[1]
    tiles = K // 256
    ns = num_stages if num_stages > 0 else min(12, tiles // split_k)
    cache_key = (is_fp8, swapped, split_k, ns)
    if cache_key in _splitk_cache:
        return _splitk_cache[cache_key]

    div = 8
    b_div = div
    c_div = 1 if swapped else div
    tn = 8 if swapped else 16

    mA = (from_dlpack(a, assumed_align=16, enable_tvm_ffi=True)
          .mark_layout_dynamic(leading_dim=1)
          .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1),
                                      divisibility=div))
    mB = (from_dlpack(b, assumed_align=16, enable_tvm_ffi=True)
          .mark_layout_dynamic(leading_dim=1)
          .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1),
                                      divisibility=b_div))
    mC = (from_dlpack(c, assumed_align=16, enable_tvm_ffi=True)
          .mark_layout_dynamic(leading_dim=1)
          .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1),
                                      divisibility=c_div))

    gemm = LLAGemm(tile_n=tn, tile_k=256, num_stages=ns,
                    num_dma_warps=4, is_fp8=is_fp8, split_k=split_k)
    stream = CUstream(current_stream().cuda_stream)
    compiled = cute.compile(gemm.call_splitk, mA, mB, mC, stream,
                            options="--enable-tvm-ffi")
    _splitk_cache[cache_key] = compiled
    logger.debug("Compiled ll_a_gemm splitk: sk=%d ns=%d swapped=%s",
                 split_k, ns, swapped)
    return compiled


def ll_a_gemm_fp8(
    hidden_states: torch.Tensor,
    weight_fp8_viewed: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """FP8 per-tensor low-latency GEMM with adaptive split-K.

    Args:
        hidden_states: [M, K] bfloat16 input activations
        weight_fp8_viewed: [N, K/2] bf16-viewed FP8 weight (pre-cached)
        input_scale: float32 per-tensor input scale
        weight_scale: float32 per-tensor weight scale
    """
    from cuda.bindings.driver import CUstream
    from torch.cuda import current_stream

    M = hidden_states.shape[0]
    K_phys = hidden_states.shape[1]

    # Quantize input to FP8, view as bf16
    x_fp8 = (hidden_states / input_scale).to(torch.float8_e4m3fn)
    # Force tight strides for M=1 (PyTorch keeps loose stride[0])
    if M == 1:
        buf = torch.empty_like(x_fp8)
        buf.copy_(x_fp8)
        x_fp8 = buf
    x8 = x_fp8.view(torch.bfloat16)

    w8 = weight_fp8_viewed  # already [N, K/2] bf16-viewed
    N = w8.shape[0]
    K_view = w8.shape[1]

    # Select split_k
    tiles = K_view // 256
    if tiles >= 12 and N <= 256:
        split_k = 12
    elif tiles >= 6 and N <= 1536:
        split_k = 6
    elif tiles >= 4:
        split_k = 4
    elif tiles >= 2:
        split_k = 2
    else:
        split_k = 1
    while tiles % split_k != 0 and split_k > 1:
        split_k -= 1

    if split_k == 1:
        out = ll_a_gemm(x8, w8, is_fp8=True)
    else:
        swapped = M <= 8
        if swapped:
            out_buf = torch.empty(split_k * N, M, dtype=torch.bfloat16,
                                  device=x8.device)
            compiled = _get_compiled_splitk(True, True, w8, x8, out_buf, split_k)
            stream = CUstream(current_stream().cuda_stream)
            compiled(w8, x8, out_buf, stream)
            out = out_buf.view(split_k, N, M).sum(dim=0).T
        else:
            out_buf = torch.empty(split_k * M, N, dtype=torch.bfloat16,
                                  device=x8.device)
            compiled = _get_compiled_splitk(True, False, x8, w8, out_buf, split_k)
            stream = CUstream(current_stream().cuda_stream)
            compiled(x8, w8, out_buf, stream)
            out = out_buf.view(split_k, M, N).sum(dim=0)

    out = out * (input_scale * weight_scale)
    return out
