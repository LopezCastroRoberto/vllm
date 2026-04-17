# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# Cache: (is_fp8,) -> compiled callable
_compiled_cache: dict[bool, object] = {}


def _get_compiled(is_fp8: bool, a, b, c):
    """Get or compile an A GEMM kernel."""
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from cutlass.cute.runtime import from_dlpack
    from torch.cuda import current_stream

    from ._ll_a_gemm_kernels import LLAGemm

    if is_fp8 in _compiled_cache:
        return _compiled_cache[is_fp8]

    div = 8
    mA = (from_dlpack(a, assumed_align=16, enable_tvm_ffi=True)
          .mark_layout_dynamic(leading_dim=1)
          .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1),
                                      divisibility=div))
    mB = (from_dlpack(b, assumed_align=16, enable_tvm_ffi=True)
          .mark_layout_dynamic(leading_dim=1)
          .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1),
                                      divisibility=div))
    mC = (from_dlpack(c, assumed_align=16, enable_tvm_ffi=True)
          .mark_layout_dynamic(leading_dim=1)
          .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1),
                                      divisibility=div))

    tk = 256 if is_fp8 else 512
    ns = 5 if is_fp8 else 3
    gemm = LLAGemm(tile_n=16, tile_k=tk, num_stages=ns,
                    num_dma_warps=4, is_fp8=is_fp8)
    stream = CUstream(current_stream().cuda_stream)
    compiled = cute.compile(gemm, mA, mB, mC, stream,
                            options="--enable-tvm-ffi")
    _compiled_cache[is_fp8] = compiled
    logger.debug("Compiled ll_a_gemm: is_fp8=%s", is_fp8)
    return compiled


def ll_a_gemm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    is_fp8: bool = False,
) -> torch.Tensor:
    """Low-latency A GEMM: C[M,N] = A[M,K] @ B[N,K]^T.

    Args:
        hidden_states: [M, K] bf16 (or fp8 viewed as bf16).
        weight: [N, K] bf16 (or fp8 viewed as bf16).
        is_fp8: If True, use fp8 MMA (data is fp8 viewed as bf16).

    Returns:
        [M, N] bf16 output tensor.
    """
    from cuda.bindings.driver import CUstream
    from torch.cuda import current_stream

    M = hidden_states.shape[0]
    N = weight.shape[0]
    output = torch.empty(M, N, dtype=torch.bfloat16,
                         device=hidden_states.device)

    compiled = _get_compiled(is_fp8, hidden_states, weight, output)

    stream = CUstream(current_stream().cuda_stream)
    compiled(hidden_states, weight, output, stream)

    return output
