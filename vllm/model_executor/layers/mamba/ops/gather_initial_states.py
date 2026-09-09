# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.warmup.jit_warmup import WarmupChoices
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
def _gather_initial_states_kernel(
    state_ptr,
    indices_ptr,
    has_initial_state_ptr,
    output_ptr,
    stride_state_batch,
    stride_indices,
    stride_has_initial_state,
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

    has_initial_state = tl.load(
        has_initial_state_ptr + batch_idx * stride_has_initial_state
    ).to(tl.int1)
    state_idx = tl.load(indices_ptr + batch_idx * stride_indices).to(tl.int64)
    state_idx = tl.where(has_initial_state, state_idx, 0)
    values = tl.load(
        state_ptr + state_idx * stride_state_batch + offsets,
        mask=mask & has_initial_state,
        other=0.0,
    )
    tl.store(output_ptr + batch_idx * row_size + offsets, values, mask=mask)


class GatherInitialStatesKernel(DeclarativeTritonJitKernel):
    kernel = staticmethod(_gather_initial_states_kernel)
    max_block_size = 1024
    num_warps = 8

    def warmup_cases(
        self,
        *,
        row_size: int,
        state_dtype: torch.dtype,
        indices_dtype: torch.dtype,
        launch_pdl: bool,
    ) -> tuple[dict[str, object], ...]:
        state_stride = triton_scalar_specialization_rep(row_size)
        block_size = min(triton.next_power_of_2(row_size), self.max_block_size)
        indices_aligned = WarmupChoices(True, False)
        has_initial_state_aligned = WarmupChoices(True, False)
        return dict(
            state_stride=compile_key(state_stride),
            indices_stride=compile_key(1),
            has_initial_state_stride=compile_key(1),
            state_dtype=compile_key(state_dtype),
            indices_dtype=compile_key(indices_dtype),
            indices_aligned=compile_key(indices_aligned),
            has_initial_state_aligned=compile_key(has_initial_state_aligned),
            state=TritonWarmupTensor(
                state_dtype, shape=(1, row_size), strides=(state_stride, 1)
            ),
            indices=TritonWarmupTensor(indices_dtype, aligned=indices_aligned),
            has_initial_state=TritonWarmupTensor(
                torch.bool, aligned=has_initial_state_aligned
            ),
            output=TritonWarmupTensor(state_dtype, shape=(1, row_size)),
            num_indices=1,
            row_size=compile_key(row_size),
            block_size=compile_key(block_size),
            launch_pdl=compile_key(launch_pdl),
        )

    def launch_spec(
        self,
        state: torch.Tensor,
        indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        output: torch.Tensor,
        num_indices: int,
        row_size: int,
        block_size: int,
        launch_pdl: bool,
    ) -> LaunchSpec:
        grid = (triton.cdiv(row_size, block_size), num_indices)
        return grid, dict(
            stride_state_batch=state.stride(0),
            stride_indices=indices.stride(0),
            stride_has_initial_state=has_initial_state.stride(0),
            row_size=row_size,
            BLOCK_SIZE=block_size,
            num_warps=self.num_warps,
            launch_pdl=launch_pdl,
        )


def gather_initial_states(
    state: torch.Tensor,
    indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> torch.Tensor:
    """Gather dense state rows, replacing uninitialized rows with zeros."""
    assert state.ndim >= 2
    assert state.is_cuda or state.is_xpu
    assert indices.ndim == 1 and has_initial_state.ndim == 1
    assert indices.shape == has_initial_state.shape
    assert indices.device == state.device
    assert has_initial_state.device == state.device
    assert indices.dtype in (torch.int32, torch.int64)
    assert has_initial_state.dtype == torch.bool

    row_size = state[0].numel()
    # Mamba pages may pad stride(0), but each state row remains dense.
    assert state[0].is_contiguous()
    output = torch.empty(
        (indices.numel(), *state.shape[1:]),
        dtype=state.dtype,
        device=state.device,
    )
    block_size = min(triton.next_power_of_2(row_size), 1024)
    GATHER_INITIAL_STATES_KERNEL(
        state,
        indices,
        has_initial_state,
        output,
        indices.numel(),
        row_size,
        block_size,
        current_platform.is_arch_support_pdl(),
    )
    return output


GATHER_INITIAL_STATES_KERNEL = GatherInitialStatesKernel()
