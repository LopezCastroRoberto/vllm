# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import sys
import torch
from triton.testing import do_bench_cudagraph

from vllm import _custom_ops as ops

q = [0.5, 0.2, 0.8]
_HAS_DSV3 = hasattr(ops, 'dsv3_fused_a_gemm')

try:
    from flashinfer.gemm import tgv_gemm_sm100
    from flashinfer import autotune
    _HAS_TGV = True
except ImportError:
    _HAS_TGV = False

try:
    from flashinfer.gemm import mm_bf16
    _HAS_MM_BF16 = True
except ImportError:
    _HAS_MM_BF16 = False

parser = argparse.ArgumentParser(description='Benchmark ll_a_gemm kernels')
parser.add_argument('--l2-pollute', action='store_true',
                    help='Measure with cold L2 via nsys+CG (slower but more realistic)')
parser.add_argument('--shape', type=str, default=None,
                    help='Filter: K,N (e.g. "7168,2112") or label substring (e.g. "a_proj")')
parser.add_argument('--M', type=str, default=None,
                    help='Filter M values: comma-separated (e.g. "1,4")')
# Internal: used by nsys subprocess
parser.add_argument('--nsys-kernel', type=str, default=None, help=argparse.SUPPRESS)
parser.add_argument('--nsys-M', type=int, default=1, help=argparse.SUPPRESS)
parser.add_argument('--nsys-K', type=int, default=7168, help=argparse.SUPPRESS)
parser.add_argument('--nsys-N', type=int, default=2112, help=argparse.SUPPRESS)
args = parser.parse_args()

# Shared: split-K autotuning (used by both nsys and normal mode) 
#TODO (roberto): need to autotune under L2 cache pollution for nsys+CG
#TODO (roberto): need to add autotuner to vLLM 
_sk_cache = {}

def _get_best_splitk(a8, b8, M, K_phys, N):
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cuda.bindings.driver import CUstream
    from torch.cuda import current_stream

    from vllm.model_executor.layers.fused_moe.router._ll_a_gemm_kernels import LLAGemm

    div = 8
    K_view = K_phys // 2
    tiles = K_view // 256
    best_t = float('inf')
    best_compiled = None
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
                t = do_bench_cudagraph(
                    lambda c=c, a=a8, b=b8, o=out: c(b, a, o, current_stream().cuda_stream, 1.0),
                    quantiles=q)[0]
                if t < best_t:
                    best_t = t
                    best_compiled = c
            except Exception:
                pass
    return best_t * 1000 if best_t < float('inf') else float('nan'), best_compiled

if args.nsys_kernel:
    from torch.cuda import current_stream as _cs
    M, K, N = args.nsys_M, args.nsys_K, args.nsys_N
    kernel_types = args.nsys_kernel.split(',')

    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    a8 = a.to(torch.float8_e4m3fn).view(torch.bfloat16)
    b8 = b.to(torch.float8_e4m3fn).view(torch.bfloat16)

    # L2 pollution via normal_()
    _l2_buf = torch.empty(64 * 1024 * 1024 // 2, dtype=torch.bfloat16, device='cuda')
    def _l2p(): _l2_buf.normal_()

    kernels = {}
    for kt in kernel_types:
        try:
            if kt == 'p-bf16':
                from vllm.model_executor.layers.fused_moe.router.ll_a_gemm import ll_a_gemm
                kernels[kt] = lambda: ll_a_gemm(a, b)
            elif kt == 'p-fp8':
                from vllm.model_executor.layers.fused_moe.router.ll_a_gemm import ll_a_gemm
                kernels[kt] = lambda: ll_a_gemm(a8, b8, is_fp8=True)
            elif kt == 't-bf16':
                from vllm.model_executor.layers.fused_moe.router.ll_a_gemm_tma import ll_a_gemm_tma
                ll_a_gemm_tma(a, b); torch.cuda.synchronize()
                kernels[kt] = lambda: ll_a_gemm_tma(a, b)
            elif kt == 't-fp8':
                from vllm.model_executor.layers.fused_moe.router.ll_a_gemm_tma import ll_a_gemm_tma
                ll_a_gemm_tma(a8, b8, is_fp8=True); torch.cuda.synchronize()
                kernels[kt] = lambda: ll_a_gemm_tma(a8, b8, is_fp8=True)
            elif kt == 'DSV3':
                o = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
                kernels[kt] = lambda o=o: ops.dsv3_fused_a_gemm(o, a, b.T)
            elif kt == 'TGV':
                bias = torch.zeros(N, dtype=torch.bfloat16, device='cuda')
                out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
                with autotune(True):
                    tgv_gemm_sm100(a, b.T, bias, out=out)
                torch.cuda.synchronize()
                kernels[kt] = lambda: tgv_gemm_sm100(a, b.T, bias, out=out)
            elif kt == 'fi-bf16':
                with autotune(True):
                    mm_bf16(a, b.T, backend='auto')
                torch.cuda.synchronize()
                kernels[kt] = lambda: mm_bf16(a, b.T, backend='auto')
            elif kt == 'sk-fp8':
                from cuda.bindings.driver import CUstream as _CUStream
                out_sk = torch.empty(N, M, dtype=torch.bfloat16, device='cuda')
                _, best_c = _get_best_splitk(a8, b8, M, K, N)
                if best_c:
                    kernels[kt] = lambda c=best_c, o=out_sk: c(b8, a8, o, _CUStream(_cs().cuda_stream), 1.0)
            elif kt == 'cuBLAS':
                omm = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
                kernels[kt] = lambda: torch.mm(a, b.T, out=omm)
            elif kt == 'smm':
                a_fp8 = a.to(torch.float8_e4m3fn)
                b_fp8 = b.to(torch.float8_e4m3fn)
                bt = b_fp8.T.contiguous()
                s1 = torch.ones(1, device='cuda', dtype=torch.float32)
                kernels[kt] = lambda: torch._scaled_mm(a_fp8, bt, scale_a=s1, scale_b=s1, out_dtype=torch.bfloat16)
        except Exception:
            pass

    stream = torch.cuda.Stream()
    graphs = {}
    for kt, kfn in kernels.items():
        with torch.cuda.stream(stream):
            _l2p(); kfn(); torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                _l2p(); kfn()
        graphs[kt] = g

    for g in graphs.values():
        for _ in range(3): g.replay()
    torch.cuda.synchronize()

    # Tag L2 prefix for identification in nsys stats
    torch.cuda.nvtx.range_push('BENCH__L2PREFIX')
    _l2p(); torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    # Profiled replays with NVTX markers
    for kt, g in graphs.items():
        torch.cuda.nvtx.range_push(f'BENCH_{kt}')
        for _ in range(20): g.replay()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    print('NSYS_KERNEL_DONE:' + ','.join(graphs.keys()))
    sys.exit(0)

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings.driver import CUstream
from torch.cuda import current_stream
from vllm.model_executor.layers.fused_moe.router.ll_a_gemm import ll_a_gemm
from vllm.model_executor.layers.fused_moe.router.ll_a_gemm_tma import ll_a_gemm_tma
from vllm.model_executor.layers.fused_moe.router._ll_a_gemm_kernels import LLAGemm

print(f'Device: {torch.cuda.get_device_name()}')
_mode = 'nsys+CG (cold L2)' if args.l2_pollute else 'do_bench_cudagraph (warm L2)'
print(f'DSV3-A: {_HAS_DSV3} | TGV: {_HAS_TGV} | mm_bf16: {_HAS_MM_BF16} | Mode: {_mode}', flush=True)
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
    (12288, 3072, "Mistral TP4 Q [FP8]"),
    (12288, 256,  "Mistral TP4 K/V [FP8]"),
    (12288, 1536, "Mistral TP8 Q [FP8]"),
    (12288, 128,  "Mistral TP8 K/V [FP8]"),
]

def _bench(fn):
    return do_bench_cudagraph(fn, quantiles=q)[0] * 1000

def _run_nsys_batch(M, K, N):
    import subprocess, shutil

    nsys = shutil.which('nsys') or '/usr/local/bin/nsys'
    script = os.path.abspath(sys.argv[0])

    kt_list = ['p-fp8', 'sk-fp8', 't-fp8', 'smm']
    if K <= 7168:
        kt_list = ['p-bf16', 't-bf16', 'cuBLAS', 'fi-bf16'] + kt_list
    if K == 7168 and N == 2112:
        kt_list.append('DSV3')
    if N % 16 == 0 and K <= 7168:
        kt_list.append('TGV')

    cmd = [nsys, 'profile', '--stats=true', '-t', 'cuda,nvtx',
           '--cuda-graph-trace=node', '-o', '/tmp/_bench_nsys_tmp', '-f', 'true',
           sys.executable, script,
           '--nsys-kernel', ','.join(kt_list),
           '--nsys-M', str(M), '--nsys-K', str(K), '--nsys-N', str(N)]

    results = {}
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # Parse nvtx_kern_sum from nsys stats
        stats_cmd = [nsys, 'stats', '--force-export=true',
                     '--report', 'nvtx_kern_sum',
                     '/tmp/_bench_nsys_tmp.nsys-rep']
        stats_result = subprocess.run(stats_cmd, capture_output=True, text=True, timeout=60)
        stats_out = stats_result.stdout + stats_result.stderr

        # Collect L2 prefix kernel names to exclude
        l2_prefixes = []
        for line in stats_out.split('\n'):
            if ':BENCH__L2PREFIX' in line:
                parts = line.split()
                if len(parts) > 12:
                    l2_prefixes.append(' '.join(parts[12:])[:40])

        # Extract per-kernel times from NVTX-tagged ranges
        for line in stats_out.split('\n'):
            if ':BENCH_' not in line or ':BENCH__L2PREFIX' in line:
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            kt = parts[0].replace(':BENCH_', '')
            try:
                med_ns = float(parts[8])
            except (ValueError, IndexError):
                continue
            kernel_name = ' '.join(parts[12:]) if len(parts) > 12 else ''
            if any(kernel_name.startswith(p) for p in l2_prefixes):
                continue
            if kt not in results:
                results[kt] = med_ns / 1000.0

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return results

bf16_keys = ['p-bf16', 't-bf16', 'DSV3', 'TGV', 'fi-bf16', 'cuBLAS']
fp8_keys = ['p-fp8', 'sk-fp8', 't-fp8', 'smm']
all_keys = ['p-bf16', 't-bf16', 'DSV3', 'TGV', 'fi-bf16', 'cuBLAS', 'p-fp8', 'sk-fp8', 't-fp8', 'smm']

def bench_one(M, K, N, label=""):
    fp8_only = '[FP8]' in label
    r = {}

    if args.l2_pollute:
        r = _run_nsys_batch(M, K, N)
        for k in all_keys:
            if k not in r:
                r[k] = float('nan')
        return r

    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    a8 = a.to(torch.float8_e4m3fn).view(torch.bfloat16)
    b8 = b.to(torch.float8_e4m3fn).view(torch.bfloat16)

    r['p-bf16'] = _bench(lambda: ll_a_gemm(a, b)) if not fp8_only else float('nan')
    r['p-fp8'] = _bench(lambda: ll_a_gemm(a8, b8, is_fp8=True))
    r['sk-fp8'], _ = _get_best_splitk(a8, b8, M, K, N)

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

    if _HAS_MM_BF16 and not fp8_only:
        try:
            with autotune(True):
                mm_bf16(a, b.T, backend='auto')
            torch.cuda.synchronize()
            r['fi-bf16'] = _bench(lambda: mm_bf16(a, b.T, backend='auto'))
        except: r['fi-bf16'] = float('nan')
    else: r['fi-bf16'] = float('nan')

    if not fp8_only:
        omm = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
        r['cuBLAS'] = _bench(lambda: torch.mm(a, b.T, out=omm))
    else: r['cuBLAS'] = float('nan')

    a_fp8 = a.to(torch.float8_e4m3fn)
    b_fp8 = b.to(torch.float8_e4m3fn)
    bt = b_fp8.T.contiguous()
    s1 = torch.ones(1, device='cuda', dtype=torch.float32)
    r['smm'] = _bench(lambda: torch._scaled_mm(a_fp8, bt, scale_a=s1, scale_b=s1, out_dtype=torch.bfloat16))
    return r

_M_vals = [int(x) for x in args.M.split(',')] if args.M else [1, 4, 8, 16]

for K, N, label in SHAPES:
    if args.shape:
        s = args.shape
        if ',' in s:
            fk, fn = s.split(',', 1)
            if int(fk) != K or int(fn) != N:
                continue
        elif s.lower() not in label.lower():
            continue

    print(f'=== {label}: K={K}, N={N} ===', flush=True)
    hdr = f"{'M':>3} |" + "".join(f" {c:>9}" for c in all_keys)
    print(hdr, flush=True)
    print('-' * len(hdr), flush=True)

    for M in _M_vals:
        r = bench_one(M, K, N, label)

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

        vb = {k: r[k] for k in bf16_keys if r[k] == r[k]}
        vf = {k: r[k] for k in fp8_keys if r[k] == r[k]}
        best_b = min(vb, key=vb.get) if vb else '?'
        best_f = min(vf, key=vf.get) if vf else '?'

        print(f" {M:2d} |" + "".join(parts) + f"  bf16={best_b} fp8={best_f}", flush=True)
    print()
