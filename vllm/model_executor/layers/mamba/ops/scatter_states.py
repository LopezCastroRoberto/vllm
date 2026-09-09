# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    DeclarativeTritonJitKernel,
    LaunchSpec,
    TritonWarmupTensor,
    compile_key,
    triton_scalar_specialization_rep,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _scatter_states_kernel(
    state_ptr,
    src_ptr,
    indices_ptr,
    stride_state_batch,
    stride_src_batch,
    stride_indices,
    row_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    block_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_size

    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    state_idx = tl.load(indices_ptr + batch_idx * stride_indices).to(tl.int64)
    values = tl.load(src_ptr + batch_idx * stride_src_batch + offsets, mask=mask)
    tl.store(state_ptr + state_idx * stride_state_batch + offsets, values, mask=mask)


class ScatterStatesKernel(DeclarativeTritonJitKernel):
    kernel = staticmethod(_scatter_states_kernel)

    def warmup_cases(
        self,
        *,
        row_size: int,
        dtype: torch.dtype,
        indices_dtype: torch.dtype,
    ) -> dict[str, object]:
        row_stride = triton_scalar_specialization_rep(row_size)
        block_size = min(triton.next_power_of_2(row_size), 1024)
        launch_pdl = current_platform.is_arch_support_pdl()
        return dict(
            dtype=compile_key(dtype),
            indices_dtype=compile_key(indices_dtype),
            stride_state_batch=compile_key(row_stride),
            stride_src_batch=compile_key(row_stride),
            stride_indices=compile_key(1),
            state=TritonWarmupTensor(
                dtype, shape=(2, row_size), strides=(row_stride, 1)
            ),
            src=TritonWarmupTensor(
                dtype, shape=(1, row_size), strides=(row_stride, 1)
            ),
            indices=TritonWarmupTensor(indices_dtype),
            num_indices=1,
            row_size=compile_key(row_size),
            block_size=compile_key(block_size),
            launch_pdl=compile_key(launch_pdl),
        )

    def launch_spec(
        self,
        state: torch.Tensor,
        src: torch.Tensor,
        indices: torch.Tensor,
        num_indices: int,
        row_size: int,
        block_size: int,
        launch_pdl: bool,
    ) -> LaunchSpec:
        grid = (triton.cdiv(row_size, block_size), num_indices)
        return grid, {
            "stride_state_batch": state.stride(0),
            "stride_src_batch": src.stride(0),
            "stride_indices": indices.stride(0),
            "row_size": row_size,
            "BLOCK_SIZE": block_size,
            "num_warps": 8,
            "launch_pdl": launch_pdl,
        }


def scatter_states(
    state: torch.Tensor,
    src: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    """Scatter ``src`` rows into ``state`` at ``indices`` (in place).

    Equivalent to ``state[indices] = src`` but non-atomic and bandwidth-bound,
    since mamba cache slots are unique per sequence. ``gather_initial_states``
    is the read-side counterpart.
    """
    assert state.ndim >= 2
    assert state.is_cuda
    assert src.ndim == state.ndim
    assert indices.ndim == 1
    assert indices.device == state.device
    assert src.shape[1:] == state.shape[1:]
    assert src.shape[0] == indices.shape[0]
    assert indices.dtype in (torch.int32, torch.int64)

    assert state[0].is_contiguous()
    assert src[0].is_contiguous()
    row_size = state[0].numel()
    block_size = min(triton.next_power_of_2(row_size), 1024)
    _SCATTER_STATES_KERNEL(
        state,
        src,
        indices,
        indices.numel(),
        row_size,
        block_size,
        current_platform.is_arch_support_pdl(),
    )


_SCATTER_STATES_KERNEL = ScatterStatesKernel()
