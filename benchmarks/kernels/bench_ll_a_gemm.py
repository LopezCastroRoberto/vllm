# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings.driver import CUstream
from torch.cuda import current_stream
from triton.testing import do_bench_cudagraph

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.router.ll_a_gemm import ll_a_gemm
from vllm.model_executor.layers.fused_moe.router.ll_a_gemm_tma import ll_a_gemm_tma
from vllm.model_executor.layers.fused_moe.router._ll_a_gemm_kernels import LLAGemm

q = [0.5, 0.2, 0.8]
_HAS_DSV3 = hasattr(ops, 'dsv3_fused_a_gemm')

try:
    from flashinfer.gemm import tgv_gemm_sm100
    from flashinfer import autotune
    _HAS_TGV = True
except ImportError:
    _HAS_TGV = False

parser = argparse.ArgumentParser()
parser.add_argument('--l2-pollute', action='store_true')
args = parser.parse_args()

if args.l2_pollute:
    _wp = torch.randn(2048, 7168, dtype=torch.bfloat16, device='cuda')
    _ap = torch.randn(1, 7168, dtype=torch.bfloat16, device='cuda')
    _op = torch.empty(1, 2048, dtype=torch.bfloat16, device='cuda')
    def _l2_prefix():
        torch.mm(_ap, _wp.T, out=_op)
    _tp = do_bench_cudagraph(_l2_prefix, rep=200, quantiles=q)[0] * 1000
else:
    _l2_prefix = None
    _tp = 0.0

print(f'Device: {torch.cuda.get_device_name()}')
print(f'DSV3-A: {_HAS_DSV3} | TGV: {_HAS_TGV} | L2-pollute: {args.l2_pollute}')
print()

SHAPES = [
    (7168, 2112,  "a_proj combined"),
    (7168, 576,   "kv_a_proj"),
    (7168, 1536,  "q_a_proj"),
    (1536, 24576, "q_b_proj TP1"),
    (1536, 3072,  "q_b_proj TP8"),
    (1536, 6144,  "q_b_proj TP4"),
    (512, 32768,  "kv_b_proj TP1"),
    (512, 4096,   "kv_b_proj TP8"),
    (512, 8192,   "kv_b_proj TP4"),
    # Mistral-Medium-3.5-128B (FP8 only)
    (12288, 3072, "Mistral TP4 Q [FP8]"),
    (12288, 256,  "Mistral TP4 K/V [FP8]"),
    (12288, 1536, "Mistral TP8 Q [FP8]"),
    (12288, 128,  "Mistral TP8 K/V [FP8]"),
]

_sk_cache = {}

def _bench(fn):
    if _l2_prefix:
        def wrapped():
            _l2_prefix()
            fn()
        t = do_bench_cudagraph(wrapped, rep=200, quantiles=q)[0] * 1000 - _tp
        return max(t, 0.01)
    return do_bench_cudagraph(fn, rep=200, quantiles=q)[0] * 1000


def _get_best_splitk(a8, b8, M, K_phys, N):
    div = 8
    K_view = K_phys // 2
    tiles = K_view // 256
    best_t = float('inf')
    for sk in [2, 3, 4, 6, 8, 12]:
        if tiles % sk != 0: continue
        for ns in [2, 3, 4]:
            if ns > tiles // sk: continue
            ck = (True, sk, ns, K_view, N)
            try:
                out = torch.empty(N, M, dtype=torch.bfloat16, device=a8.device)
                if ck not in _sk_cache:
                    mA = from_dlpack(b8, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=1).mark_compact_shape_dynamic(mode=1, stride_order=(0,1), divisibility=div)
                    mB = from_dlpack(a8, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=1).mark_compact_shape_dynamic(mode=1, stride_order=(0,1), divisibility=div)
                    mC = from_dlpack(out, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=1).mark_compact_shape_dynamic(mode=1, stride_order=(0,1), divisibility=1)
                    gemm = LLAGemm(tile_n=8, tile_k=256, num_stages=ns, num_dma_warps=4, is_fp8=True, split_k=sk)
                    _sk_cache[ck] = cute.compile(gemm.call_splitk, mA, mB, mC, CUstream(current_stream().cuda_stream), options="--enable-tvm-ffi")
                c = _sk_cache[ck]
                def rf(c=c, a=a8, b=b8, o=out):
                    if _l2_prefix: _l2_prefix()
                    c(b, a, o, current_stream().cuda_stream, 1.0)
                if _l2_prefix:
                    t = do_bench_cudagraph(rf, rep=200, quantiles=q)[0] * 1000 - _tp
                    t = max(t, 0.01)
                else:
                    t = do_bench_cudagraph(rf, rep=200, quantiles=q)[0] * 1000
                if t < best_t: best_t = t
            except Exception:
                pass
    return best_t if best_t < float('inf') else float('nan')


def bench_one(M, K, N, label=""):
    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    a8 = a.to(torch.float8_e4m3fn).view(torch.bfloat16)
    b8 = b.to(torch.float8_e4m3fn).view(torch.bfloat16)
    r = {}

    fp8_only = '[FP8]' in label

    if not fp8_only:
        r['p-bf16'] = _bench(lambda: ll_a_gemm(a, b))
    else:
        r['p-bf16'] = float('nan')

    r['p-fp8'] = _bench(lambda: ll_a_gemm(a8, b8, is_fp8=True))
    r['sk-fp8'] = _get_best_splitk(a8, b8, M, K, N)

    if not fp8_only:
        try:
            ll_a_gemm_tma(a, b); torch.cuda.synchronize()
            r['t-bf16'] = _bench(lambda: ll_a_gemm_tma(a, b))
        except: r['t-bf16'] = float('nan')
    else:
        r['t-bf16'] = float('nan')

    try:
        ll_a_gemm_tma(a8, b8, is_fp8=True); torch.cuda.synchronize()
        r['t-fp8'] = _bench(lambda: ll_a_gemm_tma(a8, b8, is_fp8=True))
    except: r['t-fp8'] = float('nan')

    if _HAS_DSV3 and K == 7168 and N == 2112 and M <= 16 and not fp8_only:
        o = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
        r['DSV3'] = _bench(lambda: ops.dsv3_fused_a_gemm(o, a, b.T))
    else: r['DSV3'] = float('nan')

    if _HAS_TGV and N % 16 == 0 and not fp8_only:
        bias = torch.zeros(N, dtype=torch.bfloat16, device='cuda')
        out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
        with autotune(True):
            tgv_gemm_sm100(a, b.T, bias, out=out)
        torch.cuda.synchronize()
        r['TGV'] = _bench(lambda: tgv_gemm_sm100(a, b.T, bias, out=out))
    else: r['TGV'] = float('nan')

    if not fp8_only:
        omm = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
        r['cuBLAS'] = _bench(lambda: torch.mm(a, b.T, out=omm))
    else:
        r['cuBLAS'] = float('nan')

    a_fp8 = a.to(torch.float8_e4m3fn)
    b_fp8 = b.to(torch.float8_e4m3fn)
    bt = b_fp8.T.contiguous()
    s1 = torch.ones(1, device='cuda', dtype=torch.float32)
    r['smm'] = _bench(lambda: torch._scaled_mm(a_fp8, bt, scale_a=s1, scale_b=s1, out_dtype=torch.bfloat16))

    return r


# bf16 group: speedups vs DSV3 (when available) else cuBLAS
# fp8 group: speedups vs scaled_mm
bf16_keys = ['p-bf16', 't-bf16', 'DSV3', 'TGV', 'cuBLAS']
fp8_keys = ['p-fp8', 'sk-fp8', 't-fp8', 'smm']
all_keys = ['p-bf16', 't-bf16', 'DSV3', 'TGV', 'cuBLAS', 'p-fp8', 'sk-fp8', 't-fp8', 'smm']

for K, N, label in SHAPES:
    print(f'=== {label}: K={K}, N={N} ===')
    hdr = f"{'M':>3} |" + "".join(f" {c:>9}" for c in all_keys)
    print(hdr)
    print('-' * len(hdr))

    for M in [1, 4, 8, 16]:
        r = bench_one(M, K, N, label)

        # bf16 baseline: DSV3 if available, else cuBLAS
        bf16_base = r['DSV3'] if r['DSV3'] == r['DSV3'] else r['cuBLAS']
        fp8_base = r['smm']

        parts = []
        for c in all_keys:
            v = r[c]
            if v != v:
                parts.append(f"{'N/A':>9s}")
                continue
            base = bf16_base if c in bf16_keys else fp8_base
            sp = base / v if v > 0 else 0
            if c == 'cuBLAS' or c == 'smm':
                parts.append(f" {v:5.1f}    ")
            else:
                parts.append(f" {v:4.1f}({sp:.2f})")

        # Winners
        vb = {k: r[k] for k in bf16_keys if r[k] == r[k]}
        vf = {k: r[k] for k in fp8_keys if r[k] == r[k]}
        best_b = min(vb, key=vb.get) if vb else '?'
        best_f = min(vf, key=vf.get) if vf else '?'

        print(f" {M:2d} |" + "".join(parts) + f"  bf16={best_b} fp8={best_f}")
    print()
