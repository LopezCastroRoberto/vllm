# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass.cute.runtime import from_dlpack
from torch.cuda import current_stream

from vllm.model_executor.layers.fused_moe.router._ll_a_gemm_kernels import LLAGemm


def test_shape(M, K, N, is_fp8=False, num_stages=4):
    if is_fp8:
        a_raw = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
        b_raw = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)
        a = a_raw.view(torch.bfloat16)
        b = b_raw.view(torch.bfloat16)
    else:
        a_raw = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        b_raw = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
        a, b = a_raw, b_raw

    c = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    div = 8
    mA = (
        from_dlpack(a, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1), divisibility=div)
    )
    mB = (
        from_dlpack(b, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1), divisibility=div)
    )
    mC = (
        from_dlpack(c, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1), divisibility=div)
    )

    gemm = LLAGemm(
        out_dtype=cutlass.BFloat16,
        tile_n=16,
        tile_k=256,
        num_stages=num_stages,
        num_dma_warps=4,
        is_fp8=is_fp8,
    )

    stream = CUstream(current_stream().cuda_stream)
    compiled = cute.compile(gemm, mA, mB, mC, stream)
    compiled(mA, mB, mC, stream)
    torch.cuda.synchronize()

    ref = torch.mm(a_raw.float(), b_raw.float().T).to(torch.bfloat16)
    err = (c - ref).abs().max().item()
    rtol_ref = ref.abs().max().item()
    rel_err = err / max(rtol_ref, 1e-6)
    dtype_str = "fp8" if is_fp8 else "bf16"
    status = "PASS" if rel_err < 1e-2 else "FAIL"
    print(
        f"  M={M:2d} K={K:4d} N={N:4d} {dtype_str} s={num_stages}: "
        f"abs_err={err:.2e} rel_err={rel_err:.2e} [{status}]"
    )
    return rel_err < 1e-2


SHAPES = [
    (16, 512, 16),
    (16, 512, 32),
    (1, 512, 16),
    (1, 2048, 512),
    (4, 2048, 512),
    (16, 2048, 512),
    (1, 7168, 2112),
    (4, 7168, 2112),
    (8, 7168, 2112),
    (16, 7168, 2112),
]


def main():
    print("cuteDSL A GEMM smoke test")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    all_pass = True
    for dtype in [False, True]:
        label = "fp8" if dtype else "bf16"
        print(f"--- {label} ---")
        for M, K, N in SHAPES:
            all_pass &= test_shape(M, K, N, is_fp8=dtype)
        print()

    print("ALL PASS" if all_pass else "SOME FAILED")


if __name__ == "__main__":
    main()
