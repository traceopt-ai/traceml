"""
Cross-integration telemetry-completeness conformance test (the StreamContract gate).

WHY THIS EXISTS
---------------
TraceML integrations can be individually correct yet silently emit INCOMPLETE
telemetry (a stream goes dark with no error). Diff-review and feature-scoped
tests catch what is *present*; the costly misses are *absences*. This turns
"the maintainer remembers HF should emit the same streams as Lightning" into a
CI gate: each integration DECLARES the telemetry streams it owes, and this test
runs a tiny end-to-end CPU run under the integration's documented `init()` path
and asserts every declared stream actually emitted >= 1 row.

DISCIPLINE: a new telemetry stream (e.g. all_reduce) is not "done" until it is
added to REQUIRED_STEP_TIME below AND emitted by every integration that owes it.

SCOPE (Stage 1, additive, zero training-path/import behavior change): POSITIVE
conformance only — it calls `init(mode='auto')` explicitly, so it regression-
gates the init() path (e.g. catches a future change that drops the DataLoader
patch from auto mode). The NEGATIVE control (assert the bare/no-init path lacks
a stream and that this harness DETECTS it) requires decoupling the integration
import from the legacy auto-init side-effect -> Stage 2.
"""

import importlib.util
import tempfile
from pathlib import Path

import pytest

from traceml_ai.sdk.initial import init
from traceml_ai.samplers.utils import drain_queue_nowait
from traceml_ai.utils.timing import _STEP_BUFFER, get_step_time_queue

# --- StreamContract registry -------------------------------------------------
# Logical stream name -> wire name (utils/timing TimeEvent.name).
STEP_TIME_WIRE = {
    "step_time": "_traceml_internal:step_time",
    "forward": "_traceml_internal:forward_time",
    "backward": "_traceml_internal:backward_time",
    "optimizer": "_traceml_internal:optimizer_step",
    "dataloader_fetch": "_traceml_internal:dataloader_next",
    "h2d": "_traceml_internal:h2d_time",  # GPU-only (arming-gated) -> not on CPU
}

# Each integration declares the STEP-TIME streams it owes on a CPU run.
# (h2d is GPU-only; step_memory rides a separate queue -> both tracked separately.)
REQUIRED_STEP_TIME = {
    "huggingface": {
        "step_time",
        "forward",
        "backward",
        "optimizer",
        "dataloader_fetch",  # the #88/#135 stream
    },
    "lightning": {
        "step_time",
        "forward",
        "backward",
        "optimizer",
        "dataloader_fetch",
    },
}


def _have(*mods: str) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in mods)


def _drain_step_time_names() -> set[str]:
    names: set[str] = set()
    for batch in drain_queue_nowait(get_step_time_queue()):
        for evt in getattr(batch, "events", []):
            names.add(evt.name)
    return names


# --- Per-integration runnable harnesses --------------------------------------
def _run_huggingface() -> set[str]:
    import torch
    from transformers import (
        BertConfig,
        BertForSequenceClassification,
        TrainingArguments,
    )

    from traceml_ai.integrations.huggingface import TraceMLTrainer

    class _TinyDS(torch.utils.data.Dataset):
        def __init__(self, n=20, seq=16, vocab=128, labels=4):
            self._rows = [
                {
                    "input_ids": (torch.arange(seq) % vocab) + (i % 7),
                    "attention_mask": torch.ones(seq, dtype=torch.long),
                    "labels": torch.tensor(i % labels),
                }
                for i in range(n)
            ]

        def __len__(self):
            return len(self._rows)

        def __getitem__(self, i):
            return self._rows[i]

    init(mode="auto")  # documented path; idempotent if already auto
    drain_queue_nowait(get_step_time_queue())
    _STEP_BUFFER.clear()

    with tempfile.TemporaryDirectory() as tmp:
        model = BertForSequenceClassification(
            BertConfig(
                vocab_size=128,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=64,
                max_position_embeddings=32,
                num_labels=4,
            )
        )
        args = TrainingArguments(
            output_dir=str(Path(tmp) / "r"),
            per_device_train_batch_size=4,
            max_steps=5,
            logging_steps=1,
            use_cpu=not torch.cuda.is_available(),
            save_strategy="no",
        )
        TraceMLTrainer(
            model=model,
            args=args,
            train_dataset=_TinyDS(),
            traceml_enabled=True,
        ).train()

    return _drain_step_time_names()


# integration -> (runner, required deps). No entry => no runnable harness yet.
_HARNESSES = {
    "huggingface": (_run_huggingface, ("torch", "transformers")),
}


@pytest.mark.parametrize("integration", sorted(REQUIRED_STEP_TIME))
def test_declared_step_time_streams_emit(integration: str):
    """Every declared step-time stream must emit >= 1 row end-to-end."""
    harness = _HARNESSES.get(integration)
    if harness is None:
        pytest.skip(f"no runnable conformance harness for {integration} yet")
    runner, deps = harness
    if not _have(*deps):
        pytest.skip(f"{integration}: required deps not installed: {deps}")

    emitted = runner()
    required = {STEP_TIME_WIRE[k] for k in REQUIRED_STEP_TIME[integration]}
    dark = sorted(w for w in required if w not in emitted)
    assert not dark, (
        f"{integration}: declared telemetry stream(s) DARK: {dark}. "
        f"Emitted: {sorted(emitted)}"
    )
