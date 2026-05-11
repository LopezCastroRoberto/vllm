# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@pytest.fixture(autouse=True, scope="module")
def _check_cutedsl():
    from vllm.model_executor.layers.fused_moe.router.ll_router_gemm import is_available
    if not is_available():
        pytest.skip("cuteDSL (CUTLASS Python) not installed")


# ===== Helpers =====

def _to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    amax = x.abs().amax().clamp(min=1e-12)
    scale = finfo.max / amax
    return (x * scale).clamp(min=finfo.min, max=finfo.max).to(dtype), scale.float().reciprocal()


def _assert_correct(out, ref, min_cos_sim=0.99, context=""):
    assert out.device.type == "cuda", f"{context}: output not on CUDA"
    assert torch.isfinite(out).all(), f"{context}: output contains NaN/Inf"
    cos = F.cosine_similarity(
        out.reshape(-1).float(), ref.reshape(-1).float(), dim=0
    ).item()
    abs_err = (out.float() - ref.float()).abs().max().item()
    msg = (f"{context}: cosine similarity {cos:.4f} < {min_cos_sim} "
           f"(abs_err={abs_err:.2e})")
    assert cos > min_cos_sim, msg


def _ref(a, b):
    return torch.mm(a.float(), b.float().T)


# ===== Shapes =====

SHAPES = [
    (256, 7168, "DSV3_router"),
    (256, 2048, "small_K"),
    (128, 5120, "DeepSeek_V2"),
    (8, 4096, "Mixtral"),
    (64, 2880, "non_aligned_K"),
]


# ===== bf16 correctness =====

@pytest.mark.parametrize("M", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("N,K,desc", SHAPES, ids=[s[2] for s in SHAPES])
def test_bf16(M, N, K, desc):
    from vllm.model_executor.layers.fused_moe.router.ll_router_gemm import ll_router_gemm
    torch.manual_seed(42)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    out = ll_router_gemm(a, b)
    ref = _ref(a, b)
    assert out.dtype == torch.float32
    assert out.shape == (M, N)
    _assert_correct(out, ref, context=f"bf16 {M}x{N}x{K}")


# ===== FP8 correctness =====

@pytest.mark.parametrize("M", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("N,K,desc",
    [(n, k, d) for n, k, d in SHAPES if k % 2 == 0],
    ids=[s[2] for s in SHAPES if s[1] % 2 == 0])
def test_fp8(M, N, K, desc):
    from vllm.model_executor.layers.fused_moe.router.ll_router_gemm import ll_router_gemm
    torch.manual_seed(42)
    a_fp8, _ = _to_float8(torch.randn(M, K, device="cuda"))
    b_fp8, _ = _to_float8(torch.randn(N, K, device="cuda"))
    out = ll_router_gemm(a_fp8, b_fp8)
    ref = _ref(a_fp8, b_fp8)
    assert out.dtype == torch.float32
    assert out.shape == (M, N)
    _assert_correct(out, ref, min_cos_sim=0.98, context=f"fp8 {M}x{N}x{K}")


# ===== Cross-validation: bf16 vs fp8 on same data =====

@pytest.mark.parametrize("M", [1, 4])
def test_bf16_vs_fp8_agreement(M):
    """bf16 and fp8 kernels on the same underlying data should roughly agree."""
    from vllm.model_executor.layers.fused_moe.router.ll_router_gemm import ll_router_gemm
    torch.manual_seed(42)
    K, N = 4096, 128
    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    out_bf16 = ll_router_gemm(a_bf16, b_bf16)

    a_fp8 = a_bf16.to(torch.float8_e4m3fn)
    b_fp8 = b_bf16.to(torch.float8_e4m3fn)
    out_fp8 = ll_router_gemm(a_fp8, b_fp8)

    _assert_correct(out_fp8, out_bf16, min_cos_sim=0.95,
                    context=f"bf16_vs_fp8 M={M}")


# ===== Arbitrary N =====

@pytest.mark.parametrize("N", [1, 3, 7, 17, 64, 256])
def test_arbitrary_N(N):
    from vllm.model_executor.layers.fused_moe.router.ll_router_gemm import ll_router_gemm
    torch.manual_seed(42)
    a = torch.randn(4, 2048, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, 2048, dtype=torch.bfloat16, device="cuda")
    out = ll_router_gemm(a, b)
    ref = _ref(a, b)
    assert out.shape == (4, N)
    _assert_correct(out, ref, context=f"N={N}")


# ===== Arbitrary K =====

@pytest.mark.parametrize("K", [64, 128, 256, 512, 1024, 2048, 4096, 7168])
def test_arbitrary_K(K):
    from vllm.model_executor.layers.fused_moe.router.ll_router_gemm import ll_router_gemm
    torch.manual_seed(42)
    a = torch.randn(4, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(32, K, dtype=torch.bfloat16, device="cuda")
    out = ll_router_gemm(a, b)
    ref = _ref(a, b)
    _assert_correct(out, ref, context=f"K={K}")


# ===== Numerical edge cases =====

def test_large_values():
    from vllm.model_executor.layers.fused_moe.router.ll_router_gemm import ll_router_gemm
    torch.manual_seed(42)
    a = torch.randn(4, 2048, dtype=torch.bfloat16, device="cuda") * 100
    b = torch.randn(64, 2048, dtype=torch.bfloat16, device="cuda") * 100
    out = ll_router_gemm(a, b)
    ref = _ref(a, b)
    assert torch.isfinite(out).all(), "Large values produced NaN/Inf"
    _assert_correct(out, ref, context="large_values")


def test_near_zero_values():
    from vllm.model_executor.layers.fused_moe.router.ll_router_gemm import ll_router_gemm
    torch.manual_seed(42)
    a = torch.randn(4, 2048, dtype=torch.bfloat16, device="cuda") * 1e-4
    b = torch.randn(64, 2048, dtype=torch.bfloat16, device="cuda") * 1e-4
    out = ll_router_gemm(a, b)
    assert torch.isfinite(out).all(), "Near-zero values produced NaN/Inf"
    assert out.abs().max() < 1.0, "Near-zero inputs should produce near-zero output"


# ===== Deterministic =====

@pytest.mark.parametrize("M", [1, 16])
def test_deterministic(M):
    from vllm.model_executor.layers.fused_moe.router.ll_router_gemm import ll_router_gemm
    torch.manual_seed(42)
    a = torch.randn(M, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(128, 4096, dtype=torch.bfloat16, device="cuda")
    out1 = ll_router_gemm(a, b)
    out2 = ll_router_gemm(a, b)
    torch.testing.assert_close(out1, out2, atol=0, rtol=0)


# ===== CUDA graph =====

def test_cudagraph():
    from vllm.model_executor.layers.fused_moe.router.ll_router_gemm import ll_router_gemm
    torch.manual_seed(42)
    a = torch.randn(4, 2048, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(64, 2048, dtype=torch.bfloat16, device="cuda")
    ll_router_gemm(a, b); torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = ll_router_gemm(a, b)

    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()

    ref = _ref(a, b)
    _assert_correct(out, ref, context="cudagraph")


def test_cudagraph_repeated_replay():
    """Many replays should produce identical results."""
    from vllm.model_executor.layers.fused_moe.router.ll_router_gemm import ll_router_gemm
    torch.manual_seed(42)
    a = torch.randn(4, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(256, 4096, dtype=torch.bfloat16, device="cuda")
    ll_router_gemm(a, b); torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = ll_router_gemm(a, b)

    results = []
    for _ in range(20):
        g.replay()
        torch.cuda.synchronize()
        results.append(out.clone())

    for i, r in enumerate(results[1:], 1):
        torch.testing.assert_close(results[0], r, atol=0, rtol=0,
                                   msg=f"Replay {i} differs from replay 0")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
