# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the Hugging Face + Accelerate H2D coverage-window gap (#276).

Reported on the HF forum: with the Trainer + Accelerate path and gradient
accumulation, a real CPU->CUDA transfer happens while TraceML reports
``H2D: 0.0 ms`` (pre-#274) or ``H2D: n/a`` (with #274).

Mechanism, measured rather than assumed (see
``tests/integrations/test_hf_h2d_window.py``, which pins it on CPU): the
transfer is performed by Accelerate's PREPARED DATALOADER, which places the
batch on the device inside ``__next__``. The fetch happens between steps,
after the previous ``on_step_end`` and before the next ``on_step_begin``,
so it falls outside the ``trace_step`` bracket that arms the H2D timer.
Under gradient accumulation, GA microbatches are fetched per optimizer step,
so GA transfers land outside the window. Note this is NOT
``Trainer._prepare_inputs``: on current transformers that call runs inside
the window and is a no-op once Accelerate has already moved the batch.

This script measures the CPU->CUDA transfer with independent CUDA events and
compares it to what TraceML captured, plus a positive control that moves a
tensor INSIDE ``trace_step`` to show the timer works when the transfer lands
in the window.

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

    from traceml_ai.sdk.instrumentation import trace_step
    from traceml_ai.utils.timing import get_step_time_queue

    model = torch.nn.Linear(4, 4).cuda()
    host = torch.empty(num_bytes_mb * 1024 * 1024 // 4, dtype=torch.float32)
    with trace_step(model):
        host.to("cuda", non_blocking=False)
        torch.cuda.synchronize()

    captured = None
    queue = get_step_time_queue()
    while not queue.empty():
        batch = queue.get_nowait()
        for evt in getattr(batch, "events", []):
            if evt.name == "_traceml_internal:h2d_time":
                captured = float(getattr(evt, "gpu_ms", 0.0) or 0.0)
    return captured


def _traceml_reported_h2d(grad_accum: int = 2) -> float | None:
    """Run a tiny HF Trainer + Accelerate GA loop; read the reported H2D."""
    import tempfile

    import torch
    from transformers import (
        BertConfig,
        BertForSequenceClassification,
        Trainer,
        TrainingArguments,
    )

    from traceml_ai.integrations.huggingface import TraceMLTrainerCallback
    from traceml_ai.utils.timing import get_step_time_queue

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

    reported = None
    queue = get_step_time_queue()
    while not queue.empty():
        batch = queue.get_nowait()
        vals = [
            float(getattr(evt, "gpu_ms", 0.0) or 0.0)
            for evt in getattr(batch, "events", [])
            if evt.name == "_traceml_internal:h2d_time"
        ]
        if vals:
            reported = sum(vals)
        elif reported is None:
            reported = 0.0  # no h2d event captured in the bracket
    return reported


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

    from traceml_ai.sdk.initial import init

    init()

    independent = _independent_h2d_ms()
    control = _positive_control_ms()
    reported = _traceml_reported_h2d(grad_accum=2)

    print("=" * 60)
    print("HF + Accelerate H2D coverage-window reproduction (#276)")
    print("=" * 60)
    print(f"independent cuda-event H2D (real transfer): {independent:.3f} ms")
    print(f"positive control (transfer inside window):  {control} ms")
    print(f"TraceML reported H2D (GA=2, pre-step move):  {reported} ms")
    print("-" * 60)
    if control and control > 0.0 and (reported in (0.0, None)):
        print(
            "REPRODUCED: the timer captures an in-window transfer but not "
            "the Accelerate pre-step transfer, so H2D reads as "
            f"{reported} despite a real {independent:.3f} ms copy."
        )
    else:
        print(
            "Did not reproduce with these settings; the Trainer/Accelerate "
            "version may move batches inside the step window."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
