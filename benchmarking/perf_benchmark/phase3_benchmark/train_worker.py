"""Publishable TraceML benchmark worker.

The worker emits raw per-step timing arrays per rank. It supports two timing
modes:
- step: one CUDA synchronize before and after the full training step.
- phase: synchronize before and after each phase for attribution.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import platform
import random
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

BENCH_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from common.io_utils import write_json
from common.stats import summary_stats
from workloads import (
    WorkloadSpec,
    build_batch_iterator,
    build_model,
    is_transformer,
)


PHASE_KEYS = [
    "trace_context_enter_ms",
    "dataloader_ms",
    "h2d_ms",
    "zero_grad_ms",
    "forward_ms",
    "backward_ms",
    "optimizer_step_ms",
    "trace_context_exit_ms",
    "inter_step_idle_ms",
    "total_step_ms",
]


@dataclass(frozen=True)
class RankInfo:
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    node_rank: int


class GilVictim:
    def __init__(self, *, inner_iters: int, max_samples: int) -> None:
        self.inner_iters = int(inner_iters)
        self.samples: deque[dict[str, float]] = deque(maxlen=int(max_samples))
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="phase3-gil-victim", daemon=True
        )
        self.native_id: int | None = None
        self.iterations = 0

    def start(self) -> None:
        self._thread.start()
        deadline = time.perf_counter() + 2.0
        while self.native_id is None and time.perf_counter() < deadline:
            time.sleep(0.001)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        self.native_id = threading.get_native_id()
        while not self._stop.is_set():
            t0 = time.perf_counter()
            value = 0
            for i in range(self.inner_iters):
                value = (value + i + 1) % 1_000_003
            t1 = time.perf_counter()
            self.iterations += 1
            self.samples.append(
                {"t_abs_sec": t1, "chunk_ms": (t1 - t0) * 1000.0}
            )

    def to_payload(self, start_time: float) -> dict:
        return {
            "enabled": True,
            "native_id": self.native_id,
            "inner_iters": self.inner_iters,
            "iterations": self.iterations,
            "retained_samples": [
                {
                    "t_rel_sec": sample["t_abs_sec"] - start_time,
                    "chunk_ms": sample["chunk_ms"],
                }
                for sample in self.samples
            ],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cell-name", required=True)
    parser.add_argument(
        "--trace-mode",
        choices=[
            "never_init",
            "residual_hooks_optimizer_active",
            "trace_manual",
            "trace_auto",
            "trace_selective_h2d_only",
            "trace_selective_no_h2d",
        ],
        required=True,
    )
    parser.add_argument(
        "--timing-mode", choices=["step", "phase"], required=True
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument(
        "--model",
        choices=[
            "tiny_mlp",
            "small_mlp",
            "wide_mlp",
            "tiny_transformer",
            "transformer_small",
        ],
        default="tiny_mlp",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--dataloader",
        choices=["synthetic", "torch_synthetic", "realistic"],
        default="synthetic",
    )
    parser.add_argument("--input-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--realistic-num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto"
    )
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--inter-step-sleep-ms", type=float, default=0.0)
    parser.add_argument("--gil-probe", action="store_true")
    parser.add_argument("--gil-inner-iters", type=int, default=1000)
    parser.add_argument("--gil-max-samples", type=int, default=200000)
    parser.add_argument("--collector-interval-sec", type=float, default=1.0)
    parser.add_argument(
        "--network-interface",
        default="",
        help="Linux NIC to sample; empty selects the default-route interface.",
    )
    parser.add_argument("--allow-traceml-noop", action="store_true")
    return parser.parse_args()


def get_rank_info() -> RankInfo:
    return RankInfo(
        rank=int(os.environ.get("RANK", "0")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        local_world_size=int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
        node_rank=int(
            os.environ.get(
                "NODE_RANK", os.environ.get("TRACEML_NODE_RANK", "0")
            )
        ),
    )


def resolve_device(args: argparse.Namespace, rank: RankInfo) -> torch.device:
    cuda_available = torch.cuda.is_available()
    if args.require_cuda and not cuda_available:
        raise SystemExit("CUDA is required for publishable benchmark runs.")
    if args.device == "cpu" or (args.device == "auto" and not cuda_available):
        return torch.device("cpu")
    if args.device == "cuda" and not cuda_available:
        raise SystemExit("--device=cuda requested, but CUDA is unavailable.")
    torch.cuda.set_device(rank.local_rank)
    return torch.device("cuda", rank.local_rank)


def init_distributed(rank: RankInfo, device: torch.device) -> None:
    if rank.world_size <= 1 or dist.is_initialized():
        return
    dist.init_process_group(
        backend="nccl" if device.type == "cuda" else "gloo"
    )


def destroy_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def read_thread_cpu_seconds() -> dict[str, float]:
    if psutil is None:
        return {}
    try:
        proc = psutil.Process()
        return {
            str(thread.id): float(thread.user_time + thread.system_time)
            for thread in proc.threads()
        }
    except Exception:
        return {}


def read_memory() -> dict[str, int]:
    payload: dict[str, int] = {}
    if psutil is not None:
        try:
            info = psutil.Process().memory_info()
            payload["rss_bytes"] = int(info.rss)
            payload["vms_bytes"] = int(info.vms)
        except Exception:
            pass
    if torch.cuda.is_available():
        try:
            payload["cuda_memory_allocated_bytes"] = int(
                torch.cuda.memory_allocated()
            )
            payload["cuda_memory_reserved_bytes"] = int(
                torch.cuda.memory_reserved()
            )
            payload["cuda_max_memory_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated()
            )
            payload["cuda_max_memory_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved()
            )
        except Exception:
            pass
    return payload


def default_network_interface() -> str | None:
    """Resolve the Linux default-route interface without a shell dependency."""
    route_path = Path("/proc/net/route")
    if not route_path.is_file():
        return None
    try:
        for line in route_path.read_text(encoding="utf-8").splitlines()[1:]:
            fields = line.split()
            if (
                len(fields) >= 4
                and fields[1] == "00000000"
                and int(fields[3], 16) & 2
            ):
                return fields[0]
    except (OSError, ValueError):
        return None
    return None


def read_network_bytes(interface: str) -> dict:
    """Read node NIC counters; one rank per node makes this rank-scoped."""
    selected = interface or default_network_interface()
    if not selected:
        return {
            "available": False,
            "reason": "no Linux default-route interface",
        }
    root = Path("/sys/class/net") / selected / "statistics"
    try:
        return {
            "available": True,
            "scope": "node_interface; per-rank when one rank runs on the node",
            "interface": selected,
            "rx_bytes": int(
                (root / "rx_bytes").read_text(encoding="utf-8").strip()
            ),
            "tx_bytes": int(
                (root / "tx_bytes").read_text(encoding="utf-8").strip()
            ),
        }
    except (OSError, ValueError):
        return {
            "available": False,
            "interface": selected,
            "reason": "counter unavailable",
        }


def network_delta(
    start: dict, end: dict, duration_sec: float, interval_sec: float
) -> dict:
    if not start.get("available") or not end.get("available"):
        return {"available": False, "start": start, "end": end}
    rx = max(0, int(end["rx_bytes"]) - int(start["rx_bytes"]))
    tx = max(0, int(end["tx_bytes"]) - int(start["tx_bytes"]))
    total = rx + tx
    return {
        "available": True,
        "scope": start["scope"],
        "interface": start["interface"],
        "duration_sec": duration_sec,
        "collector_interval_sec": interval_sec,
        "rx_bytes": rx,
        "tx_bytes": tx,
        "total_bytes": total,
        "estimated_bytes_per_collector_interval": (
            total * interval_sec / duration_sec if duration_sec > 0 else None
        ),
        "start": start,
        "end": end,
    }


def install_residual_patches() -> None:
    from traceml_ai.instrumentation.hooks.optimizer_hooks import (
        ensure_optimizer_timing_installed,
    )
    from traceml_ai.instrumentation.patches.backward_auto_timer_patch import (
        patch_backward,
    )
    from traceml_ai.instrumentation.patches.dataloader_patch import (
        patch_dataloader,
    )
    from traceml_ai.instrumentation.patches.forward_auto_timer_patch import (
        patch_forward,
    )
    from traceml_ai.instrumentation.patches.h2d_auto_timer_patch import (
        patch_h2d,
    )

    patch_dataloader()
    patch_forward()
    patch_backward()
    patch_h2d()
    ensure_optimizer_timing_installed()


def configure_traceml(
    args: argparse.Namespace,
) -> Callable[[nn.Module], object]:
    if args.trace_mode == "never_init":
        return lambda _model: contextlib.nullcontext()
    if args.trace_mode == "residual_hooks_optimizer_active":
        install_residual_patches()
        return lambda _model: contextlib.nullcontext()

    import traceml_ai as traceml

    if args.trace_mode == "trace_manual":
        cfg = traceml.init(mode="manual")
    elif args.trace_mode == "trace_auto":
        cfg = traceml.init(mode="auto")
    elif args.trace_mode == "trace_selective_h2d_only":
        cfg = traceml.init(mode="selective", patch_h2d=True)
    elif args.trace_mode == "trace_selective_no_h2d":
        cfg = traceml.init(
            mode="selective",
            patch_dataloader=True,
            patch_forward=True,
            patch_backward=True,
            patch_h2d=False,
        )
    else:
        raise AssertionError(args.trace_mode)

    if getattr(cfg, "disabled", False) and not args.allow_traceml_noop:
        raise SystemExit("TraceML init resolved to disabled/no-op.")
    return traceml.trace_step


def environment_payload(device: torch.device, rank: RankInfo) -> dict:
    gpus = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gpus.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "major": int(props.major),
                    "minor": int(props.minor),
                    "multi_processor_count": int(props.multi_processor_count),
                }
            )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "rank": asdict(rank),
        "gpus": gpus,
        "traceml_env": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("TRACEML_")
        },
    }


def timed_phase(
    key: str,
    timings: dict[str, float],
    device: torch.device,
    fn: Callable[[], object],
) -> object:
    sync_device(device)
    t0 = time.perf_counter()
    result = fn()
    sync_device(device)
    timings[key] = (time.perf_counter() - t0) * 1000.0
    return result


def move_batch(
    x_cpu: torch.Tensor,
    y_cpu: torch.Tensor,
    *,
    device: torch.device,
    non_blocking: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        x_cpu.to(device, non_blocking=non_blocking),
        y_cpu.to(device, non_blocking=non_blocking),
    )


def compute_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    transformer: bool,
) -> torch.Tensor:
    logits = model(x)
    if transformer:
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
        )
    return nn.functional.cross_entropy(logits, y)


def run_phase_step(
    *,
    args: argparse.Namespace,
    device: torch.device,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iterator: object,
    trace_context_factory: Callable[[nn.Module], object],
    transformer: bool,
) -> dict[str, float]:
    timings = {key: 0.0 for key in PHASE_KEYS}
    sync_device(device)
    step_t0 = time.perf_counter()
    cm = trace_context_factory(model)

    sync_device(device)
    enter_t0 = time.perf_counter()
    cm.__enter__()
    sync_device(device)
    timings["trace_context_enter_ms"] = (
        time.perf_counter() - enter_t0
    ) * 1000.0
    exit_error: tuple[
        type[BaseException] | None, BaseException | None, object
    ] = (
        None,
        None,
        None,
    )
    try:
        batch = timed_phase(
            "dataloader_ms", timings, device, lambda: next(iterator)
        )
        x_cpu, y_cpu = batch  # type: ignore[misc]
        x, y = timed_phase(
            "h2d_ms",
            timings,
            device,
            lambda: move_batch(
                x_cpu,
                y_cpu,
                device=device,
                non_blocking=args.pin_memory,
            ),
        )
        timed_phase(
            "zero_grad_ms",
            timings,
            device,
            lambda: optimizer.zero_grad(set_to_none=True),
        )
        holder: dict[str, torch.Tensor] = {}
        timed_phase(
            "forward_ms",
            timings,
            device,
            lambda: holder.setdefault(
                "loss",
                compute_loss(model, x, y, transformer=transformer),
            ),
        )
        timed_phase(
            "backward_ms", timings, device, lambda: holder["loss"].backward()
        )
        timed_phase(
            "optimizer_step_ms", timings, device, lambda: optimizer.step()
        )
    except BaseException as exc:
        exit_error = (type(exc), exc, exc.__traceback__)
        raise
    finally:
        sync_device(device)
        exit_t0 = time.perf_counter()
        cm.__exit__(*exit_error)
        sync_device(device)
        timings["trace_context_exit_ms"] = (
            time.perf_counter() - exit_t0
        ) * 1000.0
    if args.inter_step_sleep_ms > 0.0:
        idle_t0 = time.perf_counter()
        time.sleep(args.inter_step_sleep_ms / 1000.0)
        timings["inter_step_idle_ms"] = (
            time.perf_counter() - idle_t0
        ) * 1000.0
    sync_device(device)
    timings["total_step_ms"] = (time.perf_counter() - step_t0) * 1000.0
    return timings


def run_realistic_step(
    *,
    args: argparse.Namespace,
    device: torch.device,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iterator: object,
    trace_context_factory: Callable[[nn.Module], object],
    transformer: bool,
) -> dict[str, float]:
    timings = {key: 0.0 for key in PHASE_KEYS}
    sync_device(device)
    step_t0 = time.perf_counter()
    with trace_context_factory(model):
        x_cpu, y_cpu = next(iterator)  # type: ignore[misc]
        x, y = move_batch(
            x_cpu,
            y_cpu,
            device=device,
            non_blocking=args.pin_memory,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model, x, y, transformer=transformer)
        loss.backward()
        optimizer.step()
    if args.inter_step_sleep_ms > 0.0:
        idle_t0 = time.perf_counter()
        time.sleep(args.inter_step_sleep_ms / 1000.0)
        timings["inter_step_idle_ms"] = (
            time.perf_counter() - idle_t0
        ) * 1000.0
    sync_device(device)
    timings["total_step_ms"] = (time.perf_counter() - step_t0) * 1000.0
    return timings


def main() -> int:
    args = parse_args()
    if args.steps <= 0 or args.warmup < 0:
        raise SystemExit("--steps must be > 0 and --warmup must be >= 0.")
    if args.collector_interval_sec <= 0:
        raise SystemExit("--collector-interval-sec must be > 0.")

    rank = get_rank_info()
    random.seed(args.seed + rank.rank)
    torch.manual_seed(args.seed + rank.rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank.rank)
        torch.backends.cudnn.benchmark = True

    device = resolve_device(args, rank)
    init_distributed(rank, device)

    spec = WorkloadSpec(
        model=args.model,
        batch_size=args.batch_size,
        dataloader=args.dataloader,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        num_classes=args.num_classes,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        realistic_num_workers=args.realistic_num_workers,
        pin_memory=args.pin_memory,
    )
    model, model_payload = build_model(spec)
    model.to(device)
    if rank.world_size > 1:
        ddp_kwargs = (
            {"device_ids": [rank.local_rank]} if device.type == "cuda" else {}
        )
        model = DistributedDataParallel(model, **ddp_kwargs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    iterator = build_batch_iterator(
        spec,
        total_batches=args.steps + args.warmup + 8,
        seed=args.seed + rank.rank * 1009,
    )
    trace_context_factory = configure_traceml(args)
    transformer = is_transformer(args.model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    gil_victim = None
    if args.gil_probe:
        gil_victim = GilVictim(
            inner_iters=args.gil_inner_iters,
            max_samples=args.gil_max_samples,
        )
        gil_victim.start()

    thread_cpu_start = read_thread_cpu_seconds()
    memory_start = read_memory()
    network_start = read_network_bytes(args.network_interface)
    records: list[dict] = []
    total_iterations = args.steps + args.warmup
    run_step = (
        run_phase_step if args.timing_mode == "phase" else run_realistic_step
    )

    try:
        for step_idx in range(total_iterations):
            timings = run_step(
                args=args,
                device=device,
                model=model,
                optimizer=optimizer,
                iterator=iterator,
                trace_context_factory=trace_context_factory,
                transformer=transformer,
            )
            is_warmup = step_idx < args.warmup
            records.append(
                {
                    "step_index": step_idx,
                    "measured_step_index": (
                        None if is_warmup else step_idx - args.warmup
                    ),
                    "is_warmup": is_warmup,
                    "phases_ms": timings,
                }
            )
    finally:
        if gil_victim is not None:
            gil_victim.stop()

    measured = [record for record in records if not record["is_warmup"]]
    if len(records) != total_iterations or len(measured) != args.steps:
        raise SystemExit("Worker produced a mismatched step count.")

    thread_cpu_end = read_thread_cpu_seconds()
    duration_sec = time.perf_counter() - start_time
    thread_cpu_delta = {
        thread_id: thread_cpu_end.get(thread_id, 0.0)
        - thread_cpu_start.get(thread_id, 0.0)
        for thread_id in sorted(set(thread_cpu_start) | set(thread_cpu_end))
    }
    phase_stats = {
        key: summary_stats([float(row["phases_ms"][key]) for row in measured])
        for key in PHASE_KEYS
    }

    payload = {
        "schema_version": 3,
        "cell_name": args.cell_name,
        "trace_mode": args.trace_mode,
        "timing_mode": args.timing_mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sync_policy": (
            "one sync before/after full step"
            if args.timing_mode == "step"
            else "sync before/after every phase"
        ),
        "args": vars(args) | {"output_dir": str(args.output_dir)},
        "rank": asdict(rank),
        "environment": environment_payload(device, rank),
        "model": model_payload,
        "memory_start": memory_start,
        "memory_end": read_memory(),
        "network": network_delta(
            network_start,
            read_network_bytes(args.network_interface),
            duration_sec,
            args.collector_interval_sec,
        ),
        "thread_cpu_seconds_delta": thread_cpu_delta,
        "gil_probe": (
            gil_victim.to_payload(start_time)
            if gil_victim is not None
            else {"enabled": False}
        ),
        "phase_stats_measured": phase_stats,
        "records": records,
    }
    out_path = args.output_dir / f"rank_{rank.rank}.json"
    write_json(out_path, payload)
    if rank.rank == 0:
        print(f"[phase3-worker] wrote {out_path}")

    destroy_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
