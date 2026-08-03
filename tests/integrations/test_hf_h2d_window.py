# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""H2D coverage window for the Hugging Face Trainer callback (issue #276).

Reported from a Colab reproduction on the Hugging Face forum: under
``Trainer`` + Accelerate, independent CUDA-event timing showed a real
CPU->CUDA transfer while TraceML reported ``H2D: 0.0 ms``.

The mechanism is a COVERAGE-WINDOW gap, not a broken timer. Accelerate's
prepared dataloader performs device placement while the batch is FETCHED,
and the fetch happens between steps: after ``on_step_end`` of the previous
step and before ``on_step_begin`` of the next. The H2D auto-timer is armed
only inside the ``trace_step`` bracket that ``on_step_begin`` opens, so the
transfer is never observed. Under gradient accumulation the effect scales:
GA microbatches are fetched per optimizer step, so GA transfers land
outside the window.

These tests pin the ordering with a REAL ``Trainer`` and a REAL prepared
dataloader rather than a hand-called callback, because the defect is in the
interaction between the two. Ordering is device-independent, so they run on
CPU; the end-to-end CUDA reproduction that compares TraceML's reported value
against independent CUDA-event timing lives in
``src/dev/repro/hf_accelerate_h2d_window.py`` and needs a GPU.

What this file does NOT do is widen the window. Timing Accelerate's device
placement as its own pre-step phase is deliberately left as follow-up work
on #276, so the optimizer-step definition does not change silently. The
guarantee asserted here is the honest one: the transfer is outside the
window, therefore H2D is reported as unavailable rather than as a measured
zero, and no H2D-based verdict may rest on evidence that was never captured.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("accelerate")

from torch.utils.data import Dataset  # noqa: E402
from transformers import Trainer, TrainingArguments  # noqa: E402

from traceml_ai.diagnostics.step_time.api import (  # noqa: E402
    diagnose_step_time_window,
)
from traceml_ai.diagnostics.step_time.policy import (  # noqa: E402
    SUMMARY_STEP_TIME_POLICY,
)
from traceml_ai.instrumentation.patches.h2d_auto_timer_patch import (  # noqa: E402,E501
    _enabled,
)
from traceml_ai.integrations import huggingface as hf  # noqa: E402
from traceml_ai.sdk.instrumentation import trace_step  # noqa: E402
from tests.step_time.factories import (  # noqa: E402
    rank_average,
    window_from_events,
)

# ---------------------------------------------------------------------------
# Minimal real training setup
# ---------------------------------------------------------------------------


class _Batches(Dataset):
    def __len__(self) -> int:
        return 32

    def __getitem__(self, index):
        return {"x": torch.randn(4), "labels": torch.randn(1)}


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)

    def forward(self, x=None, labels=None):
        out = self.linear(x)
        loss = torch.nn.functional.mse_loss(out, labels)
        return {"loss": loss, "logits": out}


class _FetchSpy:
    """Wrap the prepared dataloader and record the timer state per fetch.

    Accelerate does its device placement inside ``__next__``, so the timer
    state AT FETCH is exactly the question: armed means the transfer would
    be captured, unarmed means it is invisible to TraceML.
    """

    def __init__(self, inner, log):
        self._inner = inner
        self._log = log

    def __iter__(self):
        iterator = iter(self._inner)
        while True:
            self._log.append(("fetch", _enabled()))
            try:
                yield next(iterator)
            except StopIteration:
                return

    def __len__(self):
        return len(self._inner)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _run_trainer(gradient_accumulation_steps: int, max_steps: int = 2):
    """Run a real Trainer and return the ordered (event, armed) log."""
    log: list[tuple[str, bool]] = []
    callback = hf.TraceMLTrainerCallback()

    begin, end = callback.on_step_begin, callback.on_step_end

    def _begin(*args, **kwargs):
        result = begin(*args, **kwargs)
        log.append(("window_open", _enabled()))
        return result

    def _end(*args, **kwargs):
        log.append(("window_close", _enabled()))
        return end(*args, **kwargs)

    callback.on_step_begin = _begin
    callback.on_step_end = _end

    trainer = Trainer(
        model=_Tiny(),
        args=TrainingArguments(
            output_dir="/tmp/traceml-hf-h2d-window",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_steps=max_steps,
            logging_strategy="no",
            save_strategy="no",
            report_to=[],
            use_cpu=True,
        ),
        train_dataset=_Batches(),
        callbacks=[callback],
    )

    real_loader = trainer.get_train_dataloader
    trainer.get_train_dataloader = lambda: _FetchSpy(real_loader(), log)
    trainer.train()
    return log


# ---------------------------------------------------------------------------
# The regression: batch fetch lands outside the step window (GA=1 and GA=2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ga", [1, 2])
def test_accelerate_batch_fetch_is_outside_the_step_window(ga: int) -> None:
    log = _run_trainer(gradient_accumulation_steps=ga)

    fetch_states = [armed for event, armed in log if event == "fetch"]
    assert fetch_states, "no batch fetch observed"
    # The defect, stated positively: not one fetch is covered by the window.
    assert not any(fetch_states)

    # ... while the window itself really does arm the timer, so this is a
    # coverage gap and not a dead timer.
    window_states = [armed for event, armed in log if event == "window_open"]
    assert window_states and all(window_states)

    # Under accumulation the exposure scales with GA: every microbatch for
    # the next optimizer step is fetched before that step's window opens.
    events = [event for event, _ in log]
    first_open = events.index("window_open")
    assert events[:first_open].count("fetch") == ga


def test_in_window_transfer_is_covered_positive_control() -> None:
    # Control for the assertion above: the same timer, armed the same way,
    # DOES cover work performed inside the bracket. Without this a reader
    # cannot tell a coverage gap from an instrumentation that never works.
    assert _enabled() is False
    with trace_step(_Tiny()):
        assert _enabled() is True
    assert _enabled() is False


# ---------------------------------------------------------------------------
# Consequence: an uncaptured transfer must read as unavailable, never 0.0
# ---------------------------------------------------------------------------

_EVENTS = {
    "input_wait": "_traceml_internal:dataloader_next",
    "h2d": "_traceml_internal:h2d_time",
    "forward": "_traceml_internal:forward_time",
    "backward": "_traceml_internal:backward_time",
    "optimizer_step": "_traceml_internal:optimizer_step",
    "step_time": "_traceml_internal:step_time",
}

_RUN_MS = {
    "input_wait": 3.0,
    "h2d": 2.0,
    "forward": 20.0,
    "backward": 30.0,
    "optimizer_step": 10.0,
    "step_time": 66.0,
}


def _stat(value: float) -> dict:
    return {
        "cuda:0": {
            "duration_ms": value,
            "cpu_ms": value,
            "gpu_ms": value,
            "n_calls": 1,
        }
    }


def _gpu_window(*, omit=(), h2d_value=None, steps: int = 30):
    values = dict(_RUN_MS)
    if h2d_value is not None:
        values["h2d"] = h2d_value
    per_rank = {0: {}}
    for step in range(steps):
        per_rank[0][step] = {
            _EVENTS[key]: _stat(values[key])
            for key in _EVENTS
            if key not in set(omit)
        }
    return window_from_events(per_rank, max_rows=steps)


def test_uncaptured_h2d_reports_unavailable_not_measured_zero() -> None:
    # The GA case on CUDA: the transfer happened, outside the window, so no
    # h2d event exists. Reporting 0.0 here would assert a measurement that
    # was never taken, which is what the issue reported.
    absent = _gpu_window(omit=("h2d",))
    assert absent.clock == "gpu"
    assert rank_average(absent, 0).h2d_ms is None

    # Twin: an in-window transfer that really measured 0.0 stays 0.0.
    measured = _gpu_window(h2d_value=0.0)
    assert rank_average(measured, 0).h2d_ms == 0.0


def test_uncaptured_h2d_cannot_produce_an_h2d_verdict() -> None:
    absent = _gpu_window(omit=("h2d",))
    result = diagnose_step_time_window(absent, policy=SUMMARY_STEP_TIME_POLICY)
    assert result.primary.kind != "H2D_BOUND"
