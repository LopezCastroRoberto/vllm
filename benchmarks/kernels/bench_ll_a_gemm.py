# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from triton.testing import do_bench_cudagraph

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.router.ll_a_gemm import ll_a_gemm

q = [0.5, 0.2, 0.8]
K, N = 7168, 2112
_HAS_DSV3 = hasattr(ops, 'dsv3_fused_a_gemm')

print(f'Device: {torch.cuda.get_device_name()}')
print(f'A GEMM: K={K}, N={N}')
print()

hdr = f"{'M':>3} | {'dsl-bf16':>9} {'dsl-fp8':>9} {'DSV3-A':>9} {'cuBLAS':>9}"
print(hdr)
print('-' * len(hdr))

for M in [1, 2, 4, 8, 16]:
    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    a8 = torch.randn(M, K, device='cuda').to(torch.float8_e4m3fn)
    b8 = torch.randn(N, K, device='cuda').to(torch.float8_e4m3fn)

    dsl_bf = do_bench_cudagraph(
        lambda: ll_a_gemm(a, b), rep=200, quantiles=q)[0] * 1000
    dsl_fp = do_bench_cudagraph(
        lambda: ll_a_gemm(a8.view(torch.bfloat16), b8.view(torch.bfloat16),
                           is_fp8=True), rep=200, quantiles=q)[0] * 1000

    dsv3 = float('nan')
    if _HAS_DSV3:
        b_col = torch.randn(N, K, dtype=torch.bfloat16, device='cuda').t()
        o = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
        dsv3 = do_bench_cudagraph(
            lambda: ops.dsv3_fused_a_gemm(o, a, b_col),
            rep=200, quantiles=q)[0] * 1000

    omm = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
    cub = do_bench_cudagraph(
        lambda: torch.mm(a, b.T, out=omm),
        rep=200, quantiles=q)[0] * 1000

    print(f' {M:2d} | {dsl_bf:8.2f}us {dsl_fp:8.2f}us '
          f'{dsv3:8.2f}us {cub:8.2f}us')
