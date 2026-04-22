# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from triton.testing import do_bench_cudagraph

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.router.ll_a_gemm import ll_a_gemm

q = [0.5, 0.2, 0.8]
_HAS_DSV3 = hasattr(ops, 'dsv3_fused_a_gemm')

try:
    from flashinfer.gemm import tgv_gemm_sm100
    from flashinfer import autotune
    _HAS_TGV = True
except ImportError:
    _HAS_TGV = False

print(f'Device: {torch.cuda.get_device_name()}')
print(f'DSV3-A: {_HAS_DSV3} | TGV: {_HAS_TGV}')
print()

SHAPES = [
    (7168, 2112,  "a_proj combined"),
    (7168, 576,   "kv_a_proj"),
    (7168, 1536,  "q_a_proj"),
    (1536, 24576, "q_b_proj TP1"),
    (1536, 3072,  "q_b_proj TP8"),
    (512, 32768,  "kv_b_proj TP1"),
    (512, 4096,   "kv_b_proj TP8"),
]


def bench_one(M, K, N):
    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')

    results = {}

    # cuteDSL bf16
    results['dsl-bf16'] = do_bench_cudagraph(
        lambda: ll_a_gemm(a, b), rep=200, quantiles=q)[0] * 1000

    # cuteDSL fp8
    a8 = a.to(torch.float8_e4m3fn).view(torch.bfloat16)
    b8 = b.to(torch.float8_e4m3fn).view(torch.bfloat16)
    results['dsl-fp8'] = do_bench_cudagraph(
        lambda: ll_a_gemm(a8, b8, is_fp8=True),
        rep=200, quantiles=q)[0] * 1000

    # DSV3 fused A GEMM (C++) — only supports K=7168, N=2112
    if _HAS_DSV3 and K == 7168 and N == 2112:
        o = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
        b_col = b.T
        results['DSV3-A'] = do_bench_cudagraph(
            lambda: ops.dsv3_fused_a_gemm(o, a, b_col),
            rep=200, quantiles=q)[0] * 1000
    else:
        results['DSV3-A'] = float('nan')

    # TGV-sm100 (FlashInfer tinygemm2) — requires N%16==0
    if _HAS_TGV and N % 16 == 0:
        bias = torch.zeros(N, dtype=torch.bfloat16, device='cuda')
        out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
        with autotune(True):
            tgv_gemm_sm100(a, b.T, bias, out=out)
        torch.cuda.synchronize()
        results['TGV'] = do_bench_cudagraph(
            lambda: tgv_gemm_sm100(a, b.T, bias, out=out),
            rep=200, quantiles=q)[0] * 1000
    else:
        results['TGV'] = float('nan')

    # cuBLAS
    omm = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
    results['cuBLAS'] = do_bench_cudagraph(
        lambda: torch.mm(a, b.T, out=omm),
        rep=200, quantiles=q)[0] * 1000

    return results


for K, N, label in SHAPES:
    print(f'=== {label}: K={K}, N={N} ===')
    hdr = (f"{'M':>3} | {'dsl-bf16':>9} {'dsl-fp8':>9} "
           f"{'DSV3-A':>9} {'TGV':>9} {'cuBLAS':>9}")
    print(hdr)
    print('-' * len(hdr))

    for M in [1, 4, 16]:
        r = bench_one(M, K, N)
        best = min(r, key=r.get)
        print(f' {M:2d} | {r["dsl-bf16"]:8.2f}us {r["dsl-fp8"]:8.2f}us '
              f'{r["DSV3-A"]:8.2f}us {r["TGV"]:8.2f}us '
              f'{r["cuBLAS"]:8.2f}us  <- {best}')
    print()
