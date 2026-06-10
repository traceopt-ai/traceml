"""Compute-heavy DDP workload for measuring TraceML overhead.

The script is intentionally self-contained:

- no dataset download
- fixed synthetic batch per rank
- configurable steady-state step count
- optional persistent GPU memory reservation

Run it through ``traceml run`` so TraceML can collect its normal summary
artifacts. For a paired native baseline, run the same command with
``--disable-traceml``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW

import traceml_ai as traceml


BENCHMARK_NAME = "ddp_mlp_overhead"
DEFAULT_TARGET_GPU_MEM_FRAC = 0.30
MEMORY_CHUNK_BYTES = 512 * 1024 * 1024


class ComputeMLP(nn.Module):
    """A plain MLP that keeps GEMMs large enough for overhead measurement."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hidden_layers: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        ]
        for _ in range(max(0, hidden_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic DDP MLP workload for TraceML overhead runs."
    )
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--input-dim", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--output-dim", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-gpu-mem-frac",
        type=float,
        default=DEFAULT_TARGET_GPU_MEM_FRAC,
        help=(
            "Best-effort persistent allocation target per GPU after optimizer "
            "state is initialized. Use 0 to disable. Default: 0.30."
        ),
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="benchmark_metrics.json",
        help="Rank-0 JSON output path for workload timing metrics.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.warmup_steps < 0:
        raise SystemExit("--warmup-steps must be non-negative")
    if args.steps <= args.warmup_steps:
        raise SystemExit("--steps must be greater than --warmup-steps")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.input_dim <= 0 or args.hidden_dim <= 0 or args.output_dim <= 0:
        raise SystemExit("model dimensions must be positive")
    if args.hidden_layers <= 0:
        raise SystemExit("--hidden-layers must be positive")
    if not 0.0 <= args.target_gpu_mem_frac < 0.95:
        raise SystemExit("--target-gpu-mem-frac must be in [0.0, 0.95)")


def setup_distributed() -> tuple[bool, int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    return True, rank, local_rank, world_size


def cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    idx = int(math.ceil((pct / 100.0) * len(ordered))) - 1
    return ordered[min(max(idx, 0), len(ordered) - 1)]


def summarize_steps(values: list[float]) -> dict[str, float]:
    return {
        "min_step_ms": min(values),
        "median_step_ms": statistics.median(values),
        "mean_step_ms": statistics.fmean(values),
        "p90_step_ms": percentile(values, 90.0),
        "p95_step_ms": percentile(values, 95.0),
        "max_step_ms": max(values),
    }


def reserve_gpu_memory(
    *,
    device: torch.device,
    target_fraction: float,
    rank: int,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    if device.type != "cuda" or target_fraction <= 0:
        return [], {
            "target_fraction": float(target_fraction),
            "target_bytes": 0,
            "reserved_bytes": 0,
            "allocated_bytes_after": 0,
            "total_bytes": 0,
            "actual_allocated_fraction": 0.0,
        }

    total = torch.cuda.get_device_properties(device).total_memory
    target_bytes = int(total * target_fraction)
    current = int(torch.cuda.memory_allocated(device))
    remaining = max(0, target_bytes - current)

    chunks: list[torch.Tensor] = []
    reserved_bytes = 0
    while remaining > 0:
        chunk_bytes = min(MEMORY_CHUNK_BYTES, remaining)
        try:
            chunks.append(
                torch.empty(
                    (chunk_bytes,),
                    dtype=torch.uint8,
                    device=device,
                )
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if rank == 0:
                print(
                    "[benchmark] warning: stopped GPU memory reservation "
                    f"early after {reserved_bytes / (1024**3):.2f} GiB: "
                    f"{exc}"
                )
            break
        reserved_bytes += chunk_bytes
        remaining -= chunk_bytes

    torch.cuda.synchronize(device)
    allocated_after = int(torch.cuda.memory_allocated(device))
    return chunks, {
        "target_fraction": float(target_fraction),
        "target_bytes": target_bytes,
        "reserved_bytes": reserved_bytes,
        "allocated_bytes_after": allocated_after,
        "total_bytes": total,
        "actual_allocated_fraction": allocated_after / total,
    }


def all_gather_rank_metrics(
    rank_metrics: dict[str, Any],
    *,
    distributed: bool,
    rank: int,
    world_size: int,
) -> list[dict[str, Any]]:
    if not distributed:
        return [rank_metrics]

    gathered: list[dict[str, Any] | None] = (
        [None for _ in range(world_size)] if rank == 0 else []
    )
    dist.gather_object(rank_metrics, gathered, dst=0)
    if rank != 0:
        return []
    return [item for item in gathered if item is not None]


def write_metrics(path: str, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)

    distributed, rank, local_rank, world_size = setup_distributed()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    traceml_enabled = os.environ.get("TRACEML_DISABLED", "0") != "1"
    if traceml_enabled:
        traceml.init(mode="auto")

    set_seed(args.seed + rank)

    base_model = ComputeMLP(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        hidden_layers=args.hidden_layers,
    ).to(device)

    model: nn.Module
    if distributed:
        if use_cuda:
            model = torch.nn.parallel.DistributedDataParallel(
                base_model,
                device_ids=[local_rank],
                output_device=local_rank,
            )
        else:
            model = torch.nn.parallel.DistributedDataParallel(base_model)
    else:
        model = base_model

    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(
        enabled=use_cuda,
        device="cuda" if use_cuda else "cpu",
    )
    amp_dtype = torch.float16 if use_cuda else torch.float32

    inputs = torch.randn(args.batch_size, args.input_dim, device=device)
    labels = torch.randint(
        low=0,
        high=args.output_dim,
        size=(args.batch_size,),
        dtype=torch.long,
        device=device,
    )

    def run_step() -> float:
        context = (
            traceml.trace_step(base_model)
            if traceml_enabled
            else nullcontext()
        )
        with context:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type="cuda" if use_cuda else "cpu",
                enabled=use_cuda,
                dtype=amp_dtype,
            ):
                logits = model(inputs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        return float(loss.detach())

    if distributed:
        dist.barrier()

    # Initialize optimizer state before reserving filler memory. AdamW creates
    # its large state tensors on the first optimizer step.
    run_step()
    if use_cuda:
        torch.cuda.synchronize(device)

    memory_chunks, memory_summary = reserve_gpu_memory(
        device=device,
        target_fraction=args.target_gpu_mem_frac,
        rank=rank,
    )

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    if rank == 0:
        actual_pct = 100.0 * float(
            memory_summary.get("actual_allocated_fraction", 0.0)
        )
        print(
            f"[benchmark] {BENCHMARK_NAME} world_size={world_size} "
            f"device={device} traceml_enabled={traceml_enabled} "
            f"gpu_mem_allocated={actual_pct:.1f}%"
        )

    measured_step_ms: list[float] = []
    start_wall_s = time.perf_counter()
    last_loss = 0.0

    for step_idx in range(args.steps):
        if distributed:
            dist.barrier()
        if use_cuda:
            torch.cuda.synchronize(device)

        step_start = time.perf_counter()
        last_loss = run_step()

        if use_cuda:
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - step_start) * 1000.0

        if step_idx >= args.warmup_steps:
            measured_step_ms.append(elapsed_ms)

    if distributed:
        dist.barrier()
    total_wall_s = time.perf_counter() - start_wall_s

    rank_summary: dict[str, Any] = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "hostname": os.uname().nodename,
        "device": str(device),
        "last_loss": last_loss,
        "warmup_steps": args.warmup_steps,
        "measured_steps": len(measured_step_ms),
        "total_loop_wall_s": total_wall_s,
        **summarize_steps(measured_step_ms),
    }
    if use_cuda:
        total_mem = torch.cuda.get_device_properties(device).total_memory
        rank_summary.update(
            {
                "gpu_total_bytes": total_mem,
                "gpu_memory_allocated_bytes": int(
                    torch.cuda.memory_allocated(device)
                ),
                "gpu_memory_reserved_bytes": int(
                    torch.cuda.memory_reserved(device)
                ),
                "gpu_max_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
            }
        )
    rank_summary["memory_reservation"] = memory_summary

    gathered_metrics = all_gather_rank_metrics(
        rank_summary,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    # Keep the filler tensors alive until all timing and memory metrics have
    # been collected.
    _ = memory_chunks

    if rank == 0:
        rank_medians = [
            float(item["median_step_ms"]) for item in gathered_metrics
        ]
        payload = {
            "schema_version": 1,
            "benchmark": BENCHMARK_NAME,
            "traceml_enabled": traceml_enabled,
            "config": {
                "steps": args.steps,
                "warmup_steps": args.warmup_steps,
                "batch_size": args.batch_size,
                "input_dim": args.input_dim,
                "hidden_dim": args.hidden_dim,
                "hidden_layers": args.hidden_layers,
                "output_dim": args.output_dim,
                "target_gpu_mem_frac": args.target_gpu_mem_frac,
                "lr": args.lr,
                "seed": args.seed,
            },
            "global": {
                "world_size": world_size,
                "measured_steps": args.steps - args.warmup_steps,
                "primary_step_ms": max(rank_medians),
                "median_of_rank_medians_step_ms": statistics.median(
                    rank_medians
                ),
                "mean_rank_median_step_ms": statistics.fmean(rank_medians),
                "max_rank_median_step_ms": max(rank_medians),
                "min_rank_median_step_ms": min(rank_medians),
            },
            "ranks": gathered_metrics,
        }
        write_metrics(args.metrics_file, payload)
        print(
            "[benchmark] primary_step_ms="
            f"{payload['global']['primary_step_ms']:.3f} "
            f"metrics_file={args.metrics_file}"
        )

    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
