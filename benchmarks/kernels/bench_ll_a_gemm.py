# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark: cuteDSL A GEMM (bf16 + fp8) vs DSV3-A vs cuBLAS."""

import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass.cute.runtime import from_dlpack
from torch.cuda import current_stream
from triton.testing import do_bench_cudagraph

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.router._ll_a_gemm_kernels import LLAGemm

q = [0.5, 0.2, 0.8]
K, N = 7168, 2112
_HAS_DSV3 = hasattr(ops, "dsv3_fused_a_gemm")


def bench_cutedsl(M, is_fp8, tk=512, ns=3):
    if is_fp8:
        a_fp8 = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
        b_fp8 = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)
        a, b = a_fp8.view(torch.bfloat16), b_fp8.view(torch.bfloat16)
    else:
        a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    c = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    d = 8
    mA = (
        from_dlpack(a, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1), divisibility=d)
    )
    mB = (
        from_dlpack(b, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1), divisibility=d)
    )
    mC = (
        from_dlpack(c, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1), divisibility=d)
    )
    gemm = LLAGemm(tile_n=16, tile_k=tk, num_stages=ns, num_dma_warps=4, is_fp8=is_fp8)
    s = CUstream(current_stream().cuda_stream)
    comp = cute.compile(gemm, mA, mB, mC, s)
    return (
        do_bench_cudagraph(
            lambda: comp(mA, mB, mC, CUstream(current_stream().cuda_stream)),
            rep=200,
            quantiles=q,
        )[0]
        * 1000
    )


print(f"Device: {torch.cuda.get_device_name()}")
print(f"A GEMM: K={K}, N={N}")
print()

hdr = f"{'M':>3} | {'dsl-bf16':>9} {'dsl-fp8':>9} {'DSV3-A':>9} {'cuBLAS':>9}"
print(hdr)
print("-" * len(hdr))

for M in [1, 2, 4, 8, 16]:
    dsl_bf = bench_cutedsl(M, False, tk=512, ns=3)
    dsl_fp = bench_cutedsl(M, True, tk=256, ns=5)

    dsv3 = float("nan")
    if _HAS_DSV3:
        a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").t()
        o = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        dsv3 = (
            do_bench_cudagraph(
                lambda: ops.dsv3_fused_a_gemm(o, a, b), rep=200, quantiles=q
            )[0]
            * 1000
        )

    ab = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    bb = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    omm = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    cub = (
        do_bench_cudagraph(lambda: torch.mm(ab, bb.T, out=omm), rep=200, quantiles=q)[0]
        * 1000
    )

    print(f" {M:2d} | {dsl_bf:8.2f}us {dsl_fp:8.2f}us {dsv3:8.2f}us {cub:8.2f}us")
