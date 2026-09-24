#!/usr/bin/env python3
"""Four-rank vLLM custom-all-reduce CUDA graph smoke test (TP4 adaptation).

Run only after the serving workload has stopped:

  VLLM_SKIP_P2P_CHECK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
    torchrun --standalone --nproc-per-node=4 check_custom_all_reduce_4rank.py

Requires the pinned vLLM runtime. Validates all-pairs CUDA peer access plus
small eager and graph sums; not model fit, quality, or throughput.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_SKIP_P2P_CHECK", "1")

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

WORLD = 4


def assert_value(tensor: torch.Tensor, expected: float, label: str) -> None:
    torch.cuda.synchronize()
    got = tensor.float().cpu()
    if not torch.equal(got, torch.full_like(got, expected)):
        raise RuntimeError(f"{label}: expected {expected}, got {got.tolist()}")


def main() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    for peer in range(WORLD):
        if peer == rank:
            continue
        if not torch.cuda.can_device_access_peer(rank, peer):
            raise RuntimeError(f"CUDA peer access missing for rank {rank} -> {peer}")
    dist.init_process_group("gloo")
    communicator = CustomAllreduce(
        dist.group.WORLD,
        torch.device("cuda", rank),
        max_size=64 * 1024,
        max_all_gather_size=64 * 1024,
        max_mnnvl_all_gather_size=64 * 1024,
        max_reduce_scatter_size=64 * 1024,
        max_mnnvl_reduce_scatter_size=64 * 1024,
    )
    if communicator.disabled:
        raise RuntimeError("vLLM custom all-reduce disabled itself")

    eager_input = torch.full((1024,), rank + 1, dtype=torch.bfloat16, device="cuda")
    eager_output = communicator.custom_all_reduce(eager_input)
    if eager_output is None:
        raise RuntimeError("eager custom all-reduce was not selected")
    assert_value(eager_output, float(sum(range(1, WORLD + 1))), "eager")

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    dist.barrier()
    with communicator.capture():
        with torch.cuda.stream(capture_stream):
            static_input = torch.full(
                (1024,), rank + 2, dtype=torch.bfloat16, device="cuda"
            )
            with torch.cuda.graph(graph, stream=capture_stream):
                graph_output = communicator.custom_all_reduce(static_input)
    torch.cuda.current_stream().wait_stream(capture_stream)
    if graph_output is None:
        raise RuntimeError("captured custom all-reduce was not selected")

    graph.replay()
    assert_value(graph_output, float(sum(range(2, WORLD + 2))), "graph replay")
    dist.barrier()
    if rank == 0:
        print(f"PASS: eager and CUDA-graph vLLM custom all-reduce on {WORLD} ranks")
    communicator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
