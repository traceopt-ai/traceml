# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Validate Hugging Face + Accelerate pre-step H2D coverage (#276).

Reported on the HF forum: with the Trainer + Accelerate path and gradient
accumulation, a real CPU->CUDA transfer previously happened while TraceML
reported ``H2D: 0.0 ms`` or ``H2D: n/a``.

Mechanism, measured rather than assumed (see
``tests/integrations/test_hf_h2d_window.py``, which pins it on CPU): the
transfer is performed by Accelerate's prepared DataLoader, which places the
batch on the device inside ``__next__``. The HF integration now arms H2D
timing during ``Trainer.get_batch_samples`` and records a matching short
step-time segment for each transfer before the main callback window opens.

This script measures the CPU->CUDA transfer with independent CUDA events and
compares it to what TraceML captured, plus a positive control that moves a
tensor inside ``trace_step``. It also verifies that gradient accumulation
still publishes one TraceML step per optimizer-update attempt.

Needs a CUDA device. On CPU it prints why it cannot run and exits 0.

Run:
    python -m dev.repro.hf_accelerate_h2d_window
"""

from __future__ import annotations

import sys


def _independent_h2d_ms(num_bytes_mb: int = 64) -> float:
    """Time one CPU->CUDA copy with CUDA events, independent of TraceML."""
    import torch

    host = torch.empty(num_bytes_mb * 1024 * 1024 // 4, dtype=torch.float32)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    host.to("cuda", non_blocking=False)
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def _positive_control_ms(num_bytes_mb: int = 64) -> float | None:
    """Move a tensor INSIDE trace_step; TraceML should capture this one."""
    import torch

    from traceml_ai.instrumentation.step_events import drain_step_time_batches
    from traceml_ai.sdk.instrumentation import trace_step

    model = torch.nn.Linear(4, 4).cuda()
    host = torch.empty(num_bytes_mb * 1024 * 1024 // 4, dtype=torch.float32)
    with trace_step(model):
        host.to("cuda", non_blocking=False)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    captured = None
    for batch in drain_step_time_batches():
        for evt in getattr(batch, "events", []):
            if evt.name == "_traceml_internal:h2d_time":
                evt.try_resolve()
                captured = float(getattr(evt, "gpu_time_ms", 0.0) or 0.0)
    return captured


def _traceml_reported_h2d(
    grad_accum: int = 2,
) -> tuple[float | None, int]:
    """Run a tiny HF Trainer loop and return H2D plus completed steps."""
    import tempfile

    import torch
    from transformers import (
        BertConfig,
        BertForSequenceClassification,
        Trainer,
        TrainingArguments,
    )

    from traceml_ai.instrumentation.step_events import drain_step_time_batches
    from traceml_ai.integrations.huggingface import TraceMLTrainerCallback

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return 40

        def __getitem__(self, i):
            return {
                "input_ids": torch.arange(16) % 128,
                "attention_mask": torch.ones(16, dtype=torch.long),
                "labels": torch.tensor(i % 4),
            }

    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=128,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_labels=4,
        )
    )
    with tempfile.TemporaryDirectory() as tmp:
        args = TrainingArguments(
            output_dir=tmp,
            max_steps=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=grad_accum,
            report_to=[],
            logging_strategy="no",
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=_DS(),
            callbacks=[TraceMLTrainerCallback()],
        )
        trainer.train()

    torch.cuda.synchronize()
    batches = drain_step_time_batches()
    reported = None
    for batch in batches:
        vals = [
            float(getattr(evt, "gpu_time_ms", 0.0) or 0.0)
            for evt in getattr(batch, "events", [])
            if evt.name == "_traceml_internal:h2d_time" and evt.try_resolve()
        ]
        if vals:
            reported = float(reported or 0.0) + sum(vals)
    return reported, len(batches)


def main() -> int:
    try:
        import torch
    except ImportError:
        print("torch is not installed; cannot run this reproduction.")
        return 0

    if not torch.cuda.is_available():
        print(
            "No CUDA device. This reproduction needs a GPU to move tensors "
            "host->device. Run it on Colab or an AWS GPU box."
        )
        return 0

    from traceml_ai.integrations.huggingface import init

    init()

    independent = _independent_h2d_ms()
    control = _positive_control_ms()
    reported, completed_steps = _traceml_reported_h2d(grad_accum=2)

    print("=" * 60)
    print("HF + Accelerate pre-step H2D validation (#276)")
    print("=" * 60)
    print(f"independent cuda-event H2D (real transfer): {independent:.3f} ms")
    print(f"positive control (transfer inside window):  {control} ms")
    print(f"TraceML reported H2D (GA=2, pre-step move):  {reported} ms")
    print(f"TraceML completed optimizer steps:            {completed_steps}")
    print("-" * 60)
    if control is None:
        # A None control means the run could not MEASURE, not that the bug
        # is absent: init() degrades to a disabled no-op when no aggregator
        # is reachable, so neither the control nor the trainer run recorded
        # anything. Saying "did not reproduce" here would be a false
        # all-clear from a rig that never armed.
        print(
            "INCONCLUSIVE: the in-window positive control captured nothing, "
            "which means TraceML tracing never armed in this process "
            "(init() is a no-op without a reachable aggregator). Run this "
            "script through the launcher instead:\n"
            "    traceml run src/dev/repro/hf_accelerate_h2d_window.py "
            "--mode=summary\n"
            "and inspect H2D in the final summary."
        )
        return 1
    if reported is not None and reported > 0.0 and completed_steps == 3:
        print(
            "PASS: TraceML captured Accelerate's pre-step transfers and "
            "preserved one published step per optimizer update."
        )
        return 0
    else:
        print(
            "FAIL: expected nonzero pre-step H2D and exactly three completed "
            "optimizer steps."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
