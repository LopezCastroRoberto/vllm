import torch
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cutlass_dsl import dsl_user_op
import cutlass, cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass.cute.runtime import from_dlpack
from torch.cuda import current_stream
from vllm.model_executor.layers.fused_moe.router.ll_router_gemm import ll_router_gemm

@dsl_user_op
def nanosleep(ns, *, loc=None, ip=None):
    _llvm.inline_asm(res=None, operands_=[ns.ir_value(loc=loc, ip=ip)],
        asm_string="nanosleep.u32 $0;", constraints="r",
        has_side_effects=True, loc=loc, ip=ip)

@cute.kernel
def producer_k(gOut: cute.Tensor, tail_ns: cutlass.Int32):
    tidx = cute.arch.thread_idx()[0]
    v = cutlass.Float32(1.0)
    v = v + cutlass.Float32(1.0)
    cute.arch.griddepcontrol_launch_dependents()
    nanosleep(tail_ns)
    if tidx == 0:
        gOut[0] = v

@cute.jit
def host_producer(gOut: cute.Tensor, tail_ns: cutlass.Int32, s: CUstream):
    producer_k(gOut, tail_ns).launch(
        grid=[1, 1, 1], block=[128, 1, 1], stream=s, use_pdl=True)

def bench_cg_n1(fn, n_retries=100):
    with torch.cuda.stream(torch.cuda.Stream()):
        fn(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        # warmup replays
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
        ret = []
        for _ in range(n_retries):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            g.replay()
            e.record()
            torch.cuda.synchronize()
            ret.append(s.elapsed_time(e) * 1000)
        ret.sort()
        return ret[len(ret)//2]

N, K = 256, 7168
buf = torch.empty(1, dtype=torch.float32, device="cuda")
bc = from_dlpack(buf, assumed_align=16).mark_layout_dynamic()
sfn = lambda: CUstream(current_stream().cuda_stream)
comp_p = cute.compile(host_producer, bc, 0, sfn())

print("Device:", torch.cuda.get_device_name())
print("Producer: 1 block, 128 threads | n_repeat=1, n_retries=100")
print()

for M in [1, 4, 16]:
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    ll_router_gemm(a, b); torch.cuda.synchronize()
    c_us = bench_cg_n1(lambda: ll_router_gemm(a, b))

    print("M=%d, GEMM solo: %.2fus" % (M, c_us))
    print("%10s | %10s %10s %11s %10s" % (
        "tail_ns", "producer", "pair", "sequential", "overlap"))
    print("-" * 55)

    for tns in [0, 2000, 5000, 10000, 20000, 50000]:
        def solo_p(_t=tns):
            comp_p(bc, _t, CUstream(current_stream().cuda_stream))
        p_us = bench_cg_n1(solo_p)

        def make_pair(t, aa, bb):
            def fn():
                comp_p(bc, t, CUstream(current_stream().cuda_stream))
                ll_router_gemm(aa, bb)
            return fn
        pair_us = bench_cg_n1(make_pair(tns, a, b))

        seq = p_us + c_us
        ovlp = seq - pair_us
        pct = ovlp / seq * 100 if seq > 0 else 0
        print("%8dns | %9.2fus %9.2fus %10.2fus %8.2fus (%4.1f%%)" % (
            tns, p_us, pair_us, seq, ovlp, pct))
    print()
