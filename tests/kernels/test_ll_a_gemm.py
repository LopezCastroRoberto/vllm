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
    from vllm.model_executor.layers.fused_moe.router.ll_a_gemm import is_available
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


def _ref_bf16(a, b):
    return torch.mm(a.float(), b.float().T).to(torch.bfloat16)


def _ref_fp8(a_fp8, b_fp8):
    s = torch.ones(1, device="cuda", dtype=torch.float32)
    return torch._scaled_mm(
        a_fp8, b_fp8.T.contiguous(), scale_a=s, scale_b=s, out_dtype=torch.bfloat16
    )


# ===== Shapes =====

SHAPES_BF16 = [
    (7168, 2112, "a_proj"),
    (7168, 576, "kv_a_proj"),
    (7168, 1536, "q_a_proj"),
    (1536, 3072, "q_b_TP8"),
    (512, 4096, "kv_b_TP8"),
    (512, 16, "small_N"),
    (256, 8, "tiny"),
]

SHAPES_FP8 = [
    (7168, 2112, "a_proj"),
    (7168, 576, "kv_a_proj"),
    (1536, 3072, "q_b_TP8"),
    (512, 4096, "kv_b_TP8"),
    (12288, 3072, "Mistral_Q"),
    (12288, 256, "Mistral_KV"),
    (12288, 128, "Mistral_KV8"),
]


# ===== bf16 peeled (cp.async) =====

@pytest.mark.parametrize("M", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("K,N,desc", SHAPES_BF16, ids=[s[2] for s in SHAPES_BF16])
def test_bf16_peeled(M, K, N, desc):
    from vllm.model_executor.layers.fused_moe.router.ll_a_gemm import ll_a_gemm
    torch.manual_seed(42)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    out = ll_a_gemm(a, b)
    ref = _ref_bf16(a, b)
    assert out.dtype == torch.bfloat16
    assert out.shape == (M, N)
    _assert_correct(out, ref, context=f"bf16_peeled M={M} {desc}")


# ===== bf16 TMA =====

@pytest.mark.parametrize("M", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("K,N,desc", SHAPES_BF16, ids=[s[2] for s in SHAPES_BF16])
def test_bf16_tma(M, K, N, desc):
    from vllm.model_executor.layers.fused_moe.router.ll_a_gemm_tma import ll_a_gemm_tma
    torch.manual_seed(42)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    out = ll_a_gemm_tma(a, b)
    ref = _ref_bf16(a, b)
    assert out.shape == (M, N)
    _assert_correct(out, ref, context=f"bf16_tma M={M} {desc}")


# ===== FP8 peeled =====

@pytest.mark.parametrize("M", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("K,N,desc", SHAPES_FP8, ids=[s[2] for s in SHAPES_FP8])
def test_fp8_peeled(M, K, N, desc):
    from vllm.model_executor.layers.fused_moe.router.ll_a_gemm import ll_a_gemm
    torch.manual_seed(42)
    a_fp8, _ = _to_float8(torch.randn(M, K, device="cuda"))
    b_fp8, _ = _to_float8(torch.randn(N, K, device="cuda"))
    out = ll_a_gemm(a_fp8.view(torch.bfloat16), b_fp8.view(torch.bfloat16), is_fp8=True)
    ref = _ref_fp8(a_fp8, b_fp8)
    assert out.shape == (M, N)
    _assert_correct(out, ref, min_cos_sim=0.98, context=f"fp8_peeled M={M} {desc}")


# ===== FP8 TMA =====

@pytest.mark.parametrize("M", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("K,N,desc", SHAPES_FP8, ids=[s[2] for s in SHAPES_FP8])
def test_fp8_tma(M, K, N, desc):
    from vllm.model_executor.layers.fused_moe.router.ll_a_gemm_tma import ll_a_gemm_tma
    torch.manual_seed(42)
    a_fp8, _ = _to_float8(torch.randn(M, K, device="cuda"))
    b_fp8, _ = _to_float8(torch.randn(N, K, device="cuda"))
    out = ll_a_gemm_tma(a_fp8.view(torch.bfloat16), b_fp8.view(torch.bfloat16), is_fp8=True)
    ref = _ref_fp8(a_fp8, b_fp8)
    assert out.shape == (M, N)
    _assert_correct(out, ref, min_cos_sim=0.98, context=f"fp8_tma M={M} {desc}")


# ===== Split-K FP8 =====

@pytest.mark.parametrize("M", [1, 4, 8])
@pytest.mark.parametrize("sk,ns", [(2, 2), (4, 2), (4, 4), (8, 3)],
                         ids=["sk2_ns2", "sk4_ns2", "sk4_ns4", "sk8_ns3"])
@pytest.mark.parametrize("K,N,desc", SHAPES_FP8, ids=[s[2] for s in SHAPES_FP8])
def test_splitk_fp8(M, K, N, desc, sk, ns):
    from vllm.model_executor.layers.fused_moe.router.ll_a_gemm import _get_compiled_splitk
    from cuda.bindings.driver import CUstream
    from torch.cuda import current_stream

    K_view = K // 2
    tiles = K_view // 256
    if tiles % sk != 0 or ns > tiles // sk:
        pytest.skip(f"tiles={tiles} incompatible with sk={sk} ns={ns}")

    torch.manual_seed(42)
    a_fp8, _ = _to_float8(torch.randn(M, K, device="cuda"))
    b_fp8, _ = _to_float8(torch.randn(N, K, device="cuda"))
    a8 = a_fp8.view(torch.bfloat16)
    b8 = b_fp8.view(torch.bfloat16)
    ref = _ref_fp8(a_fp8, b_fp8)

    out = torch.empty(N, M, dtype=torch.bfloat16, device="cuda")
    compiled = _get_compiled_splitk(True, True, b8, a8, out, sk, ns)
    compiled(b8, a8, out, CUstream(current_stream().cuda_stream), 1.0)
    torch.cuda.synchronize()

    assert out.T.shape == (M, N)
    _assert_correct(out.T, ref, min_cos_sim=0.98, context=f"sk{sk}_ns{ns} M={M} {desc}")


# ===== Cross-validation: peeled vs TMA must agree =====

@pytest.mark.parametrize("M", [1, 4, 16])
@pytest.mark.parametrize("K,N", [(7168, 2112), (512, 4096)])
def test_peeled_vs_tma_bf16(M, K, N):
    from vllm.model_executor.layers.fused_moe.router.ll_a_gemm import ll_a_gemm
    from vllm.model_executor.layers.fused_moe.router.ll_a_gemm_tma import ll_a_gemm_tma
    torch.manual_seed(42)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    out_p = ll_a_gemm(a, b)
    out_t = ll_a_gemm_tma(a, b)
    _assert_correct(out_p, out_t, min_cos_sim=0.999, context=f"peeled_vs_tma M={M}")


# ===== Cross-validation: split-K vs non-split-K FP8 =====

@pytest.mark.parametrize("M", [1, 4])
def test_splitk_vs_nosplitk_fp8(M):
    from vllm.model_executor.layers.fused_moe.router.ll_a_gemm import ll_a_gemm, _get_compiled_splitk
    from cuda.bindings.driver import CUstream
    from torch.cuda import current_stream

    K, N = 12288, 256
    torch.manual_seed(42)
    a_fp8, _ = _to_float8(torch.randn(M, K, device="cuda"))
    b_fp8, _ = _to_float8(torch.randn(N, K, device="cuda"))
    a8 = a_fp8.view(torch.bfloat16)
    b8 = b_fp8.view(torch.bfloat16)

    out_nosplit = ll_a_gemm(a8, b8, is_fp8=True)

    out_sk = torch.empty(N, M, dtype=torch.bfloat16, device="cuda")
    compiled = _get_compiled_splitk(True, True, b8, a8, out_sk, split_k=8, num_stages=3)
    compiled(b8, a8, out_sk, CUstream(current_stream().cuda_stream), 1.0)
    torch.cuda.synchronize()

    _assert_correct(out_sk.T, out_nosplit, min_cos_sim=0.999,
                    context=f"sk_vs_nosk M={M}")


# ===== Scale parameter =====

@pytest.mark.parametrize("scale", [0.5, 2.0, 0.01])
def test_scale_bf16(scale):
    from vllm.model_executor.layers.fused_moe.router.ll_a_gemm import ll_a_gemm
    torch.manual_seed(42)
    a = torch.randn(4, 7168, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(2112, 7168, dtype=torch.bfloat16, device="cuda")
    out_scaled = ll_a_gemm(a, b, scale=scale)
    out_unscaled = ll_a_gemm(a, b, scale=1.0)
    ref = out_unscaled * scale
    _assert_correct(out_scaled, ref, min_cos_sim=0.999, context=f"scale={scale}")


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_scale_fp8(scale):
    from vllm.model_executor.layers.fused_moe.router.ll_a_gemm import ll_a_gemm
    torch.manual_seed(42)
    a_fp8, _ = _to_float8(torch.randn(4, 7168, device="cuda"))
    b_fp8, _ = _to_float8(torch.randn(2112, 7168, device="cuda"))
    a8, b8 = a_fp8.view(torch.bfloat16), b_fp8.view(torch.bfloat16)
    out_scaled = ll_a_gemm(a8, b8, is_fp8=True, scale=scale)
    out_unscaled = ll_a_gemm(a8, b8, is_fp8=True, scale=1.0)
    ref = out_unscaled * scale
    _assert_correct(out_scaled, ref, min_cos_sim=0.999, context=f"fp8_scale={scale}")


# ===== Numerical edge cases =====

def test_large_values():
    from vllm.model_executor.layers.fused_moe.router.ll_a_gemm import ll_a_gemm
    torch.manual_seed(42)
    a = torch.randn(4, 1536, dtype=torch.bfloat16, device="cuda") * 100
    b = torch.randn(256, 1536, dtype=torch.bfloat16, device="cuda") * 100
    out = ll_a_gemm(a, b)
    ref = _ref_bf16(a, b)
    assert torch.isfinite(out).all(), "Large values produced NaN/Inf"
    _assert_correct(out, ref, min_cos_sim=0.99, context="large_values")


def test_near_zero_values():
    from vllm.model_executor.layers.fused_moe.router.ll_a_gemm import ll_a_gemm
    torch.manual_seed(42)
    a = torch.randn(4, 1536, dtype=torch.bfloat16, device="cuda") * 1e-4
    b = torch.randn(256, 1536, dtype=torch.bfloat16, device="cuda") * 1e-4
    out = ll_a_gemm(a, b)
    assert torch.isfinite(out).all(), "Near-zero values produced NaN/Inf"
    assert out.abs().max() < 1.0, "Near-zero inputs should produce near-zero output"


# ===== Swapped vs non-swapped boundary =====

def test_swap_boundary():
    """M=8 (swapped) and M=9 (non-swapped) should both be correct."""
    from vllm.model_executor.layers.fused_moe.router.ll_a_gemm import ll_a_gemm
    torch.manual_seed(42)
    b = torch.randn(2112, 7168, dtype=torch.bfloat16, device="cuda")
    for M in [8, 9]:
        a = torch.randn(M, 7168, dtype=torch.bfloat16, device="cuda")
        out = ll_a_gemm(a, b)
        ref = _ref_bf16(a, b)
        assert out.shape == (M, 2112)
        _assert_correct(out, ref, context=f"swap_boundary M={M}")


# ===== Deterministic =====

@pytest.mark.parametrize("M", [1, 8])
def test_deterministic(M):
    from vllm.model_executor.layers.fused_moe.router.ll_a_gemm import ll_a_gemm
    torch.manual_seed(42)
    a = torch.randn(M, 7168, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(2112, 7168, dtype=torch.bfloat16, device="cuda")
    out1 = ll_a_gemm(a, b)
    out2 = ll_a_gemm(a, b)
    torch.testing.assert_close(out1, out2, atol=0, rtol=0)


# ===== CUDA graph =====

def test_cudagraph_peeled():
    from vllm.model_executor.layers.fused_moe.router.ll_a_gemm import ll_a_gemm
    torch.manual_seed(42)
    a = torch.randn(4, 7168, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(2112, 7168, dtype=torch.bfloat16, device="cuda")
    ll_a_gemm(a, b); torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = ll_a_gemm(a, b)

    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()

    ref = _ref_bf16(a, b)
    _assert_correct(out, ref, context="CG_peeled")


def test_cudagraph_splitk():
    from vllm.model_executor.layers.fused_moe.router.ll_a_gemm import _get_compiled_splitk
    from cuda.bindings.driver import CUstream
    from torch.cuda import current_stream

    torch.manual_seed(42)
    a_fp8, _ = _to_float8(torch.randn(1, 12288, device="cuda"))
    b_fp8, _ = _to_float8(torch.randn(256, 12288, device="cuda"))
    a8, b8 = a_fp8.view(torch.bfloat16), b_fp8.view(torch.bfloat16)
    ref = _ref_fp8(a_fp8, b_fp8)

    out = torch.empty(256, 1, dtype=torch.bfloat16, device="cuda")
    compiled = _get_compiled_splitk(True, True, b8, a8, out, split_k=8, num_stages=3)
    compiled(b8, a8, out, CUstream(current_stream().cuda_stream), 1.0)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        compiled(b8, a8, out, CUstream(current_stream().cuda_stream), 1.0)

    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()

    _assert_correct(out.T, ref, min_cos_sim=0.98, context="CG_splitk")


def test_cudagraph_repeated_replay():
    """Many replays should produce consistent results (catches counter bugs)."""
    from vllm.model_executor.layers.fused_moe.router.ll_a_gemm import ll_a_gemm
    torch.manual_seed(42)
    a = torch.randn(4, 7168, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(2112, 7168, dtype=torch.bfloat16, device="cuda")
    ll_a_gemm(a, b); torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = ll_a_gemm(a, b)

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
