# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""H2D coverage window for the Hugging Face Trainer callback (issue #276).

Reported from a Colab reproduction on the Hugging Face forum: under
``Trainer`` + Accelerate, independent CUDA-event timing showed a real
CPU->CUDA transfer while TraceML reported ``H2D: 0.0 ms``.

The mechanism was a COVERAGE-WINDOW gap, not a broken timer. Accelerate's
prepared dataloader performs device placement while the batch is FETCHED,
and the fetch happens between steps: after ``on_step_end`` of the previous
step and before ``on_step_begin`` of the next. The main ``trace_step`` bracket
still opens at ``on_step_begin``, but the HF integration now arms H2D timing
while ``get_batch_samples`` performs those fetches. Each transfer receives a
matching short step-time segment and remains in the same pending capture as
the later compute region.

These tests pin the ordering with a REAL ``Trainer`` and a REAL prepared
dataloader rather than a hand-called callback, because the defect is in the
interaction between the two. Ordering is device-independent, so they run on
CPU; the end-to-end CUDA reproduction that compares TraceML's reported value
against independent CUDA-event timing lives in
``src/dev/repro/hf_accelerate_h2d_window.py`` and needs a GPU.

The optimizer-step definition does not change: ``on_step_end`` still publishes
one capture. The final tests also retain the fallback contract for paths where
H2D is genuinely unobserved: absence remains unavailable rather than a
measured zero.
"""

from __future__ import annotations

import inspect
import logging

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("accelerate")

from torch.utils.data import Dataset, IterableDataset  # noqa: E402
from transformers import Trainer, TrainingArguments  # noqa: E402

from traceml_ai.diagnostics.step_time.api import (  # noqa: E402
    diagnose_step_time_window,
)
from traceml_ai.diagnostics.step_time.policy import (  # noqa: E402
    SUMMARY_STEP_TIME_POLICY,
)
from traceml_ai.instrumentation.patches.h2d_auto_timer_patch import (  # noqa: E402,E501
    _enabled,
    _include_step_time,
)
from traceml_ai.instrumentation.patches.dataloader_patch import (  # noqa: E402,E501
    _timing_allowed,
    dataloader_timing_scope,
)
from traceml_ai.instrumentation.step_events import (  # noqa: E402
    abort_step_capture,
    begin_step_capture,
    drain_step_memory_events,
    drain_step_time_batches,
)
from traceml_ai.integrations import huggingface as hf  # noqa: E402
from traceml_ai.runtime.state import reset_trace_session_state  # noqa: E402
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


class _IterableBatches(IterableDataset):
    def __init__(self, timing_states=None) -> None:
        self._timing_states = timing_states

    def __iter__(self):
        for index in range(64):
            if self._timing_states is not None:
                self._timing_states.append(
                    (_timing_allowed(), _enabled(), _include_step_time())
                )
            value = torch.full((4,), float(index))
            yield {"x": value, "labels": value[:1]}


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
            self._log.append(("fetch", _enabled(), _include_step_time()))
            try:
                yield next(iterator)
            except StopIteration:
                return

    def __len__(self):
        return len(self._inner)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _PhaseSpy:
    """Record integration gates where Trainer consumes a prepared loader."""

    def __init__(self, inner, phase, log):
        self._inner = inner
        self._phase = phase
        self._log = log

    def __iter__(self):
        iterator = iter(self._inner)
        while True:
            self._log.append(
                (
                    self._phase,
                    _timing_allowed(),
                    _enabled(),
                    _include_step_time(),
                )
            )
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
    log: list[tuple[str, bool, bool]] = []
    hf._install_training_batch_timing()
    callback = hf.TraceMLTrainerCallback()

    begin, end = callback.on_step_begin, callback.on_step_end

    def _begin(*args, **kwargs):
        result = begin(*args, **kwargs)
        log.append(("window_open", _enabled(), _include_step_time()))
        return result

    def _end(*args, **kwargs):
        log.append(("window_close", _enabled(), _include_step_time()))
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
# The fix: batch transfers are covered before the main step window
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ga", [1, 2])
def test_accelerate_batch_transfer_is_covered_during_collection(
    ga: int,
) -> None:
    log = _run_trainer(gradient_accumulation_steps=ga)

    fetch_states = [
        (armed, includes_step)
        for event, armed, includes_step in log
        if event == "fetch"
    ]
    assert fetch_states, "no batch fetch observed"
    assert all(
        armed and includes_step for armed, includes_step in fetch_states
    )

    # The main callback window covers compute without asking each in-window
    # transfer to contribute another standalone step-time segment.
    window_states = [
        (armed, includes_step)
        for event, armed, includes_step in log
        if event == "window_open"
    ]
    assert window_states
    assert all(
        armed and not includes_step for armed, includes_step in window_states
    )

    # Under accumulation the exposure scales with GA: every microbatch for
    # the next optimizer step is fetched before that step's window opens.
    events = [event for event, _, _ in log]
    first_open = events.index("window_open")
    assert events[:first_open].count("fetch") == ga


def test_in_window_transfer_is_covered_positive_control() -> None:
    # Control for the assertion above: the same timer, armed the same way,
    # DOES cover work performed inside the bracket. Without this a reader
    # cannot tell a coverage gap from an instrumentation that never works.
    assert _enabled() is False
    assert _include_step_time() is False
    with trace_step(_Tiny()):
        assert _enabled() is True
        assert _include_step_time() is False
    assert _enabled() is False
    assert _include_step_time() is False


def test_batch_collection_hook_installation_is_idempotent() -> None:
    hf._install_training_batch_timing()
    installed = Trainer.get_batch_samples
    hf._install_training_batch_timing()
    assert Trainer.get_batch_samples is installed


def test_resume_skip_tracking_installation_is_idempotent() -> None:
    import transformers.trainer as trainer_module

    hf._install_resume_skip_tracking()
    installed = trainer_module.skip_first_batches
    hf._install_resume_skip_tracking()
    assert trainer_module.skip_first_batches is installed


def test_resume_skip_tracking_distinguishes_accelerate_loader_shapes() -> None:
    """Only Accelerate's iterable skip wrapper consumes discarded batches lazily."""
    import transformers.trainer as trainer_module
    from torch.utils.data import DataLoader

    hf._install_resume_skip_tracking()
    state = hf._TRAINING_ATTEMPT_STATE
    previous = (
        state.active,
        state.suppress_next_input_group,
        state.omit_current_group,
    )
    try:
        state.active = True
        state.suppress_next_input_group = False
        state.omit_current_group = False

        map_loader = trainer_module.skip_first_batches(
            DataLoader(_Batches(), batch_size=1), 1
        )
        assert getattr(map_loader, "skip_batches", 0) == 0
        assert state.suppress_next_input_group is False

        iterable_loader = trainer_module.skip_first_batches(
            DataLoader(_IterableBatches(), batch_size=1), 1
        )
        assert getattr(iterable_loader, "skip_batches", 0) == 1
        assert state.suppress_next_input_group is True
    finally:
        (
            state.active,
            state.suppress_next_input_group,
            state.omit_current_group,
        ) = previous


def test_non_training_input_scope_installation_is_idempotent() -> None:
    hf._install_non_training_input_scopes()
    installed = (Trainer.evaluate, Trainer.predict)
    hf._install_non_training_input_scopes()
    assert (Trainer.evaluate, Trainer.predict) == installed


def test_missing_get_batch_samples_warns_once_and_skips_installation(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setattr(hf, "_WARNED_CAPABILITIES", set())
    monkeypatch.delattr(Trainer, "get_batch_samples")

    with caplog.at_level(logging.WARNING, logger=hf.__name__):
        hf._install_training_batch_timing()
        hf._install_training_batch_timing()

    messages = [record.getMessage() for record in caplog.records]
    unsupported = [
        message
        for message in messages
        if "requires transformers>=4.46.1" in message
    ]
    assert len(unsupported) == 1
    assert "training Input Wait and pre-step H2D" in unsupported[0]
    assert "Training will continue" in unsupported[0]


def test_custom_batch_collection_override_warns_once(
    monkeypatch, caplog
) -> None:
    monkeypatch.setattr(hf, "_WARNED_CAPABILITIES", set())

    class CustomTrainer:
        def get_batch_samples(self):
            return []

    trainer = CustomTrainer()
    with caplog.at_level(logging.WARNING, logger=hf.__name__):
        hf._warn_if_training_batch_timing_is_bypassed(trainer)
        hf._warn_if_training_batch_timing_is_bypassed(trainer)

    messages = [record.getMessage() for record in caplog.records]
    bypassed = [
        message
        for message in messages
        if "overrides get_batch_samples" in message
    ]
    assert len(bypassed) == 1
    assert "CustomTrainer" in bypassed[0]


def test_batch_collection_warning_failure_cannot_abort_training(
    monkeypatch, tmp_path, capsys
) -> None:
    """An application-owned logging handler cannot break Trainer control."""
    monkeypatch.setattr(hf, "_WARNED_CAPABILITIES", set())

    class DelegatingTrainer(Trainer):
        def get_batch_samples(self, *args, **kwargs):
            return super().get_batch_samples(*args, **kwargs)

    class RaisingHandler(logging.Handler):
        def emit(self, record):
            raise RuntimeError("application logging failed")

    hf.init()
    trainer = DelegatingTrainer(
        model=_Tiny(),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=4,
            max_steps=1,
            logging_strategy="no",
            save_strategy="no",
            report_to=[],
            disable_tqdm=True,
            use_cpu=True,
        ),
        train_dataset=_Batches(),
        callbacks=[hf.TraceMLTrainerCallback()],
    )

    handler = RaisingHandler()
    previous_level = hf.logger.level
    hf.logger.setLevel(logging.WARNING)
    hf.logger.addHandler(handler)
    try:
        trainer.train()
    finally:
        hf.logger.removeHandler(handler)
        hf.logger.setLevel(previous_level)

    assert trainer.state.global_step == 1
    stderr = capsys.readouterr().err
    assert "[TraceML]" in stderr
    assert "overrides get_batch_samples" in stderr


def test_installed_batch_collection_hook_does_not_warn(
    monkeypatch, caplog
) -> None:
    monkeypatch.setattr(hf, "_WARNED_CAPABILITIES", set())
    hf._install_training_batch_timing()

    class InheritedTrainer(Trainer):
        pass

    trainer = object.__new__(InheritedTrainer)
    with caplog.at_level(logging.WARNING, logger=hf.__name__):
        hf._warn_if_training_batch_timing_is_bypassed(trainer)

    assert "cannot guarantee pre-step H2D timing" not in " ".join(
        record.getMessage() for record in caplog.records
    )


def test_training_collection_excludes_evaluation_fetches(tmp_path) -> None:
    """Only the standard training iterator opens the input timing gates."""
    reset_trace_session_state()
    abort_step_capture(begin_step_capture())
    drain_step_time_batches()
    drain_step_memory_events()
    hf.init()

    log: list[tuple[str, bool, bool, bool]] = []
    trainer = Trainer(
        model=_Tiny(),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            max_steps=2,
            eval_strategy="steps",
            eval_steps=1,
            logging_strategy="no",
            save_strategy="no",
            report_to=[],
            disable_tqdm=True,
            use_cpu=True,
        ),
        train_dataset=_Batches(),
        eval_dataset=_Batches(),
        callbacks=[hf.TraceMLTrainerCallback()],
    )

    real_train_loader = trainer.get_train_dataloader
    real_eval_loader = trainer.get_eval_dataloader
    trainer.get_train_dataloader = lambda: _PhaseSpy(
        real_train_loader(), "train", log
    )
    trainer.get_eval_dataloader = lambda *args, **kwargs: _PhaseSpy(
        real_eval_loader(*args, **kwargs), "eval", log
    )

    trainer.train()

    training = [entry[1:] for entry in log if entry[0] == "train"]
    evaluation = [entry[1:] for entry in log if entry[0] == "eval"]
    assert training and evaluation
    assert all(
        input_wait and h2d and step_time
        for input_wait, h2d, step_time in training
    )
    assert all(
        not input_wait and not h2d and not step_time
        for input_wait, h2d, step_time in evaluation
    )

    batches = drain_step_time_batches()
    assert [batch.step for batch in batches] == [1, 2]
    assert begin_step_capture().timing_events == []


def test_collection_bookkeeping_runs_outside_timing_gates(tmp_path) -> None:
    """Token accounting after each fetch is not mislabeled as input or H2D."""
    reset_trace_session_state()
    abort_step_capture(begin_step_capture())
    drain_step_time_batches()
    hf.init()

    states: list[tuple[bool, bool, bool]] = []

    class BookkeepingBatch(dict):
        def __contains__(self, key):
            if key == "labels":
                states.append(
                    (_timing_allowed(), _enabled(), _include_step_time())
                )
            return super().__contains__(key)

    trainer = Trainer(
        model=_Tiny(),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=4,
            max_steps=1,
            logging_strategy="no",
            save_strategy="no",
            report_to=[],
            disable_tqdm=True,
            use_cpu=True,
        ),
        train_dataset=_Batches(),
        callbacks=[hf.TraceMLTrainerCallback()],
    )

    batch = BookkeepingBatch(labels=torch.tensor([[1.0, -100.0]]))
    parameters = inspect.signature(trainer.get_batch_samples).parameters
    args = [iter([batch]), 1]
    if "device" in parameters:
        args.append(trainer.args.device)

    # Match the lifecycle guard's disabled-by-default policy. The collection
    # proxy enables timing only for iterator.next(); the subsequent labels
    # check is ordinary Trainer bookkeeping and must observe the outer policy.
    with dataloader_timing_scope(lambda: False):
        trainer.get_batch_samples(*args)

    assert states
    assert all(
        not input_wait and not h2d and not step_time
        for input_wait, h2d, step_time in states
    )


@pytest.mark.parametrize(
    ("ga", "ignore_data_skip", "first_group_measured"),
    [(1, False, False), (2, False, False), (1, True, True)],
)
@pytest.mark.parametrize("fail_before_retry", [False, True])
def test_iterable_resume_omits_only_ambiguous_group(
    tmp_path,
    monkeypatch,
    ga,
    ignore_data_skip,
    first_group_measured,
    fail_before_retry,
) -> None:
    """Lazy checkpoint skipping must not look like first-step input work."""
    reset_trace_session_state()
    abort_step_capture(begin_step_capture())
    drain_step_time_batches()
    drain_step_memory_events()
    hf.init()

    first = Trainer(
        model=_Tiny(),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=ga,
            max_steps=1,
            logging_strategy="no",
            save_strategy="steps",
            save_steps=1,
            report_to=[],
            disable_tqdm=True,
            use_cpu=True,
        ),
        train_dataset=_IterableBatches(),
        callbacks=[hf.TraceMLTrainerCallback()],
    )
    first.train()
    checkpoint = tmp_path / "checkpoint-1"
    assert checkpoint.is_dir()

    assert (
        len(drain_step_time_batches()) == 1
    )  # Fresh training starts immediately.
    assert len(drain_step_memory_events()) == 1
    reset_trace_session_state()
    timing_states: list[tuple[bool, bool, bool]] = []
    resumed = Trainer(
        model=_Tiny(),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=ga,
            max_steps=3,
            ignore_data_skip=ignore_data_skip,
            logging_strategy="no",
            save_strategy="no",
            report_to=[],
            disable_tqdm=True,
            use_cpu=True,
        ),
        train_dataset=_IterableBatches(timing_states),
        callbacks=[hf.TraceMLTrainerCallback()],
    )
    traced_global_steps = []
    original_trace_step = hf.trace_step

    def record_trace_step(model):
        traced_global_steps.append(resumed.state.global_step)
        return original_trace_step(model)

    monkeypatch.setattr(hf, "trace_step", record_trace_step)
    if fail_before_retry:
        original_training_step = resumed.training_step

        def fail_training_step(*args, **kwargs):
            original_training_step(*args, **kwargs)
            raise RuntimeError("resume group failed")

        with monkeypatch.context() as patch:
            patch.setattr(resumed, "training_step", fail_training_step)
            with pytest.raises(RuntimeError, match="resume group failed"):
                resumed.train(resume_from_checkpoint=str(checkpoint))
        assert drain_step_time_batches() == []
        assert drain_step_memory_events() == []
        assert hf._TRAINING_ATTEMPT_STATE.active is False
        assert hf._TRAINING_ATTEMPT_STATE.suppress_next_input_group is False
        assert hf._TRAINING_ATTEMPT_STATE.omit_current_group is False
        traced_global_steps.clear()
        timing_states.clear()

    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 3

    batches = drain_step_time_batches()
    memory_events = drain_step_memory_events()
    assert [batch.step for batch in batches] == (
        [1, 2] if first_group_measured else [1]
    )
    assert len(memory_events) == (2 if first_group_measured else 1)
    # No trace_step means no compute or memory publication for the omitted
    # group; HF nevertheless completes both optimizer updates.
    assert traced_global_steps == ([1, 2] if first_group_measured else [2])

    fetch = "_traceml_internal:dataloader_next"
    forward = "_traceml_internal:forward_time"
    names = [{event.name for event in batch.events} for batch in batches]
    assert all(fetch in group and forward in group for group in names)
    assert timing_states
    expected_first_state = (
        (True, True, True) if first_group_measured else (False, False, False)
    )
    assert timing_states[0] == expected_first_state
    assert any(state == (True, True, True) for state in timing_states)
    assert hf._TRAINING_ATTEMPT_STATE.active is False
    assert hf._TRAINING_ATTEMPT_STATE.suppress_next_input_group is False
    assert hf._TRAINING_ATTEMPT_STATE.omit_current_group is False


@pytest.mark.parametrize("ga", [1, 2])
def test_map_style_resume_keeps_first_groups_input_timing(
    tmp_path, ga
) -> None:
    """Sampler-level checkpoint skipping does not fetch discarded batches."""
    reset_trace_session_state()
    abort_step_capture(begin_step_capture())
    drain_step_time_batches()
    hf.init()

    arguments = dict(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=ga,
        logging_strategy="no",
        report_to=[],
        disable_tqdm=True,
        use_cpu=True,
    )
    first = Trainer(
        model=_Tiny(),
        args=TrainingArguments(
            **arguments,
            max_steps=1,
            save_strategy="steps",
            save_steps=1,
        ),
        train_dataset=_Batches(),
        callbacks=[hf.TraceMLTrainerCallback()],
    )
    first.train()

    drain_step_time_batches()
    reset_trace_session_state()
    resumed = Trainer(
        model=_Tiny(),
        args=TrainingArguments(
            **arguments,
            max_steps=2,
            save_strategy="no",
        ),
        train_dataset=_Batches(),
        callbacks=[hf.TraceMLTrainerCallback()],
    )
    resumed.train(resume_from_checkpoint=str(tmp_path / "checkpoint-1"))

    batches = drain_step_time_batches()
    assert len(batches) == 1
    assert any(
        event.name == "_traceml_internal:dataloader_next"
        for event in batches[0].events
    )


@pytest.mark.parametrize("method_name", ["evaluate", "predict"])
def test_standalone_non_training_input_leaves_no_pending_events(
    tmp_path, method_name
) -> None:
    """Public non-training loops cannot contaminate a later traced step."""
    reset_trace_session_state()
    abort_step_capture(begin_step_capture())
    drain_step_time_batches()
    hf.init()

    trainer = Trainer(
        model=_Tiny(),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            per_device_eval_batch_size=4,
            logging_strategy="no",
            save_strategy="no",
            report_to=[],
            disable_tqdm=True,
            use_cpu=True,
        ),
        eval_dataset=_Batches(),
        callbacks=[hf.TraceMLTrainerCallback()],
    )

    if method_name == "evaluate":
        trainer.evaluate()
    else:
        trainer.predict(_Batches())

    assert drain_step_time_batches() == []
    assert begin_step_capture().timing_events == []


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


def test_captured_h2d_is_already_part_of_traced_step_time() -> None:
    measured = _gpu_window()
    average = rank_average(measured, 0)

    assert average.h2d_ms == pytest.approx(2.0)
    assert average.traced_step_time_ms == pytest.approx(66.0)
    assert average.residual_ms == pytest.approx(4.0)


def test_uncaptured_h2d_cannot_produce_an_h2d_verdict() -> None:
    absent = _gpu_window(omit=("h2d",))
    result = diagnose_step_time_window(absent, policy=SUMMARY_STEP_TIME_POLICY)
    assert result.primary.kind != "H2D_BOUND"
