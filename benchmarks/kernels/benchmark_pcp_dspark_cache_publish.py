# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare DSpark local-store plus publish with fused PCP final stores."""

import argparse
import os
import statistics
from functools import partial

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.utils.network_utils import get_open_port


def _time_ms(fn, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _run_baseline(
    ops,
    copy_fn,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    peer_ptrs: torch.Tensor,
    scale: torch.Tensor,
    rank: int,
) -> None:
    ops.reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slots,
        "auto",
        scale,
        scale,
    )
    copy_fn(
        cache,
        slots,
        peer_ptrs,
        rank,
        token_dim=2,
        segment_dim=1,
    )


def _run_fused(
    store_fn,
    key: torch.Tensor,
    value: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    peer_ptrs: torch.Tensor,
) -> None:
    assert store_fn(key, value, cache, slots, peer_ptrs)


def _worker(
    rank: int,
    world_size: int,
    port: int,
    token_counts: list[int],
    warmup: int,
    iterations: int,
    rounds: int,
) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
        }
    )
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}"),
    )

    from vllm import _custom_ops as ops
    from vllm.distributed.device_communicators.symm_mem import allocate_symm_mem_peer
    from vllm.model_executor.layers.attention.pcp_direct_kv import (
        PCPPeerCacheFence,
        copy_pcp_cache_rows_to_peers,
        store_pcp_kv_rows_to_peers,
    )

    device = torch.device(f"cuda:{rank}")
    num_heads = 8
    head_size = 128
    block_size = 64
    dtype = torch.bfloat16
    scale = torch.ones(1, dtype=torch.float32, device=device)

    if rank == 0:
        print("tokens,baseline_us,fused_us,improvement_pct")
    for num_tokens in token_counts:
        num_blocks = (world_size * num_tokens + block_size - 1) // block_size
        shape = (num_blocks, num_heads, block_size, 2 * head_size)
        baseline_allocation = allocate_symm_mem_peer(
            shape, dtype, device, dist.group.WORLD
        )
        fused_allocation = allocate_symm_mem_peer(
            shape, dtype, device, dist.group.WORLD
        )
        baseline_cache = baseline_allocation.storage
        fused_cache = fused_allocation.storage
        baseline_ptrs = baseline_allocation.peer_ptrs_for_view(baseline_cache)
        fused_ptrs = fused_allocation.peer_ptrs_for_view(fused_cache)
        slots = torch.arange(
            rank * num_tokens,
            (rank + 1) * num_tokens,
            dtype=torch.int64,
            device=device,
        )
        key = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device=device)
        value = torch.randn_like(key)
        key_cache, value_cache = baseline_cache.transpose(1, 2).split(head_size, dim=-1)
        fence = PCPPeerCacheFence(dist.group.WORLD, device)

        baseline = partial(
            _run_baseline,
            ops,
            copy_pcp_cache_rows_to_peers,
            key,
            value,
            key_cache,
            value_cache,
            baseline_cache,
            slots,
            baseline_ptrs,
            scale,
            rank,
        )
        fused = partial(
            _run_fused,
            store_pcp_kv_rows_to_peers,
            key,
            value,
            fused_cache,
            slots,
            fused_ptrs,
        )

        for _ in range(warmup):
            baseline()
            fused()
        torch.cuda.synchronize()

        samples: dict[str, list[float]] = {"baseline": [], "fused": []}
        functions = {"baseline": baseline, "fused": fused}
        for round_idx in range(rounds):
            order = ("baseline", "fused")
            if round_idx % 2:
                order = tuple(reversed(order))
            for name in order:
                dist.barrier()
                local_ms = torch.tensor(
                    _time_ms(functions[name], iterations), device=device
                )
                dist.all_reduce(local_ms, op=dist.ReduceOp.MAX)
                samples[name].append(float(local_ms.item()))
                fence()

        baseline_ms = statistics.median(samples["baseline"])
        fused_ms = statistics.median(samples["fused"])
        if rank == 0:
            improvement = (baseline_ms - fused_ms) / baseline_ms * 100
            print(
                f"{num_tokens},{baseline_ms * 1000:.3f},"
                f"{fused_ms * 1000:.3f},{improvement:.2f}"
            )

        fence.close()
        baseline_allocation.close()
        fused_allocation.close()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--tokens", type=int, nargs="+", default=[64, 256, 1024, 4096])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if torch.cuda.device_count() < args.world_size:
        raise RuntimeError(f"Benchmark requires {args.world_size} GPUs")

    port = get_open_port()
    mp.spawn(
        _worker,
        args=(
            args.world_size,
            port,
            args.tokens,
            args.warmup,
            args.iterations,
            args.rounds,
        ),
        nprocs=args.world_size,
    )


if __name__ == "__main__":
    main()
