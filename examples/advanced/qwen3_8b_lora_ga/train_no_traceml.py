"""Run the Qwen3-8B LoRA workload without importing or starting TraceML."""

from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback

from train import main


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "minimum": None,
            "maximum": None,
        }
    ordered = sorted(values)
    p95_index = round(0.95 * (len(ordered) - 1))
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": ordered[p95_index],
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


class OptimizerStepTimer(TrainerCallback):
    """Record non-blocking CUDA and host timings for every optimizer step."""

    def __init__(self) -> None:
        self._train_started_at: float | None = None
        self._step_started_at: float | None = None
        self._start_event: torch.cuda.Event | None = None
        self._pending: list[
            tuple[int, torch.cuda.Event, torch.cuda.Event, float]
        ] = []

    def on_train_begin(self, *args: Any, **kwargs: Any) -> None:
        torch.cuda.synchronize()
        self._train_started_at = time.perf_counter()

    def on_step_begin(self, *args: Any, **kwargs: Any) -> None:
        self._step_started_at = time.perf_counter()
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._start_event.record()

    def on_step_end(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> None:
        if self._start_event is None or self._step_started_at is None:
            return
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        host_interval_ms = (time.perf_counter() - self._step_started_at) * 1000
        self._pending.append(
            (
                int(state.global_step),
                self._start_event,
                end_event,
                host_interval_ms,
            )
        )
        self._start_event = None
        self._step_started_at = None

    def on_train_end(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> None:
        torch.cuda.synchronize()
        train_loop_wall_s = (
            time.perf_counter() - self._train_started_at
            if self._train_started_at is not None
            else None
        )
        rows = [
            {
                "optimizer_step": step,
                "gpu_elapsed_ms": start.elapsed_time(end),
                "host_interval_ms": host_interval_ms,
            }
            for step, start, end, host_interval_ms in self._pending
        ]

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "optimizer_step_times.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "optimizer_step",
                    "gpu_elapsed_ms",
                    "host_interval_ms",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        gpu_values = [float(row["gpu_elapsed_ms"]) for row in rows]
        host_values = [float(row["host_interval_ms"]) for row in rows]
        payload = {
            "schema_version": 1,
            "instrumentation": "none",
            "timing_method": {
                "train_loop_wall": (
                    "time.perf_counter with one CUDA synchronization at each "
                    "boundary"
                ),
                "optimizer_step_gpu": (
                    "CUDA events recorded at Trainer optimizer-step "
                    "boundaries; one synchronization after training"
                ),
                "host_interval": (
                    "time.perf_counter without per-step synchronization"
                ),
            },
            "train_loop_wall_s": train_loop_wall_s,
            "gpu_elapsed_ms": _summary(gpu_values),
            "host_interval_ms": _summary(host_values),
        }
        json_path = output_dir / "baseline_timing.json"
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

        print("\nNo-TraceML timing")
        if train_loop_wall_s is not None:
            print(f"  Training-loop wall time:  {train_loop_wall_s:.2f} s")
        if gpu_values:
            print(
                "  Mean GPU optimizer step: "
                f"{statistics.fmean(gpu_values):.2f} ms"
            )
        print(f"  Per-step CSV:             {csv_path}")
        print(f"  Timing summary:           {json_path}\n", flush=True)


if __name__ == "__main__":
    main(
        enable_traceml=False,
        extra_callbacks=(OptimizerStepTimer(),),
    )
