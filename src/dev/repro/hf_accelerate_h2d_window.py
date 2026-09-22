# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Validate Hugging Face + Accelerate pre-step H2D coverage (#276).

Reported on the HF forum: with the Trainer + Accelerate path and gradient
accumulation, a real CPU-to-CUDA transfer previously happened while TraceML
reported ``H2D: 0.0 ms`` or ``H2D: n/a``.

The validation has two process roles:

* the parent measures an independent CUDA copy and launches TraceML;
* the launcher child runs a three-step Hugging Face workload.

The parent validates ``final_summary.json`` only after the runtime sampler has
exported the child's events. It intentionally does not drain TraceML's internal
event queue: that queue has a single consumer while the runtime is active.

Needs a CUDA device. On CPU it prints why it cannot run and exits 0.

Run from the repository root::

    python src/dev/repro/hf_accelerate_h2d_window.py \
        --logs-dir traceml-validation \
        --run-name hf-h2d-validation
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_EXPECTED_STEPS = 3


@dataclass(frozen=True)
class _SummaryMeasurement:
    h2d_ms: float | None
    completed_steps: int | None


def _independent_h2d_ms(num_bytes_mb: int = 64) -> float:
    """Time one CPU-to-CUDA copy with events independent of TraceML."""
    import torch

    host = torch.empty(num_bytes_mb * 1024 * 1024 // 4, dtype=torch.float32)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    device = host.to("cuda", non_blocking=False)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = float(start.elapsed_time(end))
    del device, host
    torch.cuda.empty_cache()
    return elapsed_ms


def _run_hf_workload(grad_accum: int = 2) -> int:
    """Run the launcher-owned workload without consuming internal queues."""
    import tempfile

    import torch
    from transformers import (
        BertConfig,
        BertForSequenceClassification,
        Trainer,
        TrainingArguments,
    )

    from traceml_ai.integrations.huggingface import (
        TraceMLTrainerCallback,
        init,
    )
    from traceml_ai.sdk.summary_client import final_summary

    class _Dataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 40

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.arange(16) % 128,
                "attention_mask": torch.ones(16, dtype=torch.long),
                "labels": torch.tensor(index % 4),
            }

    init()
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=128,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_labels=4,
        )
    )
    with tempfile.TemporaryDirectory() as output_dir:
        training_args = TrainingArguments(
            output_dir=output_dir,
            max_steps=_EXPECTED_STEPS,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=grad_accum,
            report_to=[],
            logging_strategy="no",
            save_strategy="no",
            disable_tqdm=True,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=_Dataset(),
            callbacks=[TraceMLTrainerCallback()],
        )
        trainer.train()

    completed_steps = int(trainer.state.global_step)
    print(
        f"HF workload completed optimizer steps: {completed_steps}",
        flush=True,
    )
    if completed_steps != _EXPECTED_STEPS:
        print(
            f"Expected {_EXPECTED_STEPS} optimizer steps, got "
            f"{completed_steps}.",
            flush=True,
        )
        return 1

    # Exercise the public worker-to-aggregator request/response protocol while
    # the real runtime is still alive. The aggregator settles the sampler and
    # SQLite writer before it returns this payload; callers must never inspect
    # or drain the sampler's private event queue themselves.
    summary = final_summary(timeout_sec=60.0, print_text=False)
    summary_steps = int(
        ((summary or {}).get("step_time") or {})
        .get("metadata", {})
        .get("training_total_steps", -1)
    )
    print(
        f"HF runtime summary round-trip optimizer steps: {summary_steps}",
        flush=True,
    )
    if summary_steps != _EXPECTED_STEPS:
        print(
            "Expected the live TraceML summary to contain "
            f"{_EXPECTED_STEPS} optimizer steps, got {summary_steps}.",
            flush=True,
        )
        return 1
    return 0


def _read_summary(path: Path) -> _SummaryMeasurement:
    """Read the two public summary values that establish this regression."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    step_time = payload.get("step_time") or {}
    metadata = step_time.get("metadata") or {}
    global_values = step_time.get("global") or {}
    average = global_values.get("average") or {}

    raw_h2d = average.get("h2d_ms")
    raw_steps = metadata.get("training_total_steps")
    return _SummaryMeasurement(
        h2d_ms=None if raw_h2d is None else float(raw_h2d),
        completed_steps=None if raw_steps is None else int(raw_steps),
    )


def _build_launch_command(*, logs_dir: Path, run_name: str) -> list[str]:
    """Build the product path used by the GPU validation."""
    return [
        sys.executable,
        "-m",
        "traceml_ai.launcher.cli",
        "run",
        "--mode",
        "summary",
        "--logs-dir",
        str(logs_dir),
        "--run-name",
        run_name,
        str(Path(__file__).resolve()),
        "--args",
        "--workload",
    ]


def _run_validation(*, logs_dir: Path, run_name: str) -> int:
    """Run the workload through TraceML, then validate its final summary."""
    try:
        import torch
    except ImportError:
        print("torch is not installed; cannot run this reproduction.")
        return 0

    if not torch.cuda.is_available():
        print(
            "No CUDA device. This reproduction needs a GPU to move tensors "
            "host-to-device. Run it on Colab or a GPU machine."
        )
        return 0

    independent_ms = _independent_h2d_ms()
    command = _build_launch_command(logs_dir=logs_dir, run_name=run_name)
    launched = subprocess.run(command, check=False)
    if launched.returncode != 0:
        print(
            "FAIL: the TraceML workload exited with code "
            f"{launched.returncode}."
        )
        return 1

    summary_path = logs_dir / run_name / "final_summary.json"
    if not summary_path.is_file():
        print(f"FAIL: expected final summary at {summary_path}.")
        return 1

    try:
        measurement = _read_summary(summary_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: could not read {summary_path}: {exc}")
        return 1

    print("=" * 60)
    print("HF + Accelerate pre-step H2D validation (#276)")
    print("=" * 60)
    print(
        "independent cuda-event H2D (real transfer): "
        f"{independent_ms:.3f} ms"
    )
    print(
        "TraceML final-summary H2D:                   "
        f"{measurement.h2d_ms} ms"
    )
    print(
        "TraceML completed optimizer steps:            "
        f"{measurement.completed_steps}"
    )
    print(f"TraceML summary: {summary_path}")
    print("-" * 60)

    if (
        measurement.h2d_ms is not None
        and measurement.h2d_ms > 0.0
        and measurement.completed_steps == _EXPECTED_STEPS
    ):
        print(
            "PASS: TraceML captured Accelerate's pre-step transfers and "
            "preserved one published step per optimizer update."
        )
        return 0

    print(
        "FAIL: expected nonzero final-summary H2D and exactly "
        f"{_EXPECTED_STEPS} completed optimizer steps."
    )
    return 1


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("traceml-validation"),
    )
    parser.add_argument(
        "--run-name",
        default=f"hf-h2d-validation-{time.strftime('%Y%m%d-%H%M%S')}",
    )
    parser.add_argument(
        "--workload", action="store_true", help=argparse.SUPPRESS
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.workload:
        return _run_hf_workload()
    return _run_validation(logs_dir=args.logs_dir, run_name=args.run_name)


if __name__ == "__main__":
    sys.exit(main())
