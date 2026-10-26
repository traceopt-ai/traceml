# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the canonical Step Time analyzer."""

from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path
from typing import Mapping, Optional

import pytest

from traceml_ai.step_time.analysis import StepTimeAnalyzer
from traceml_ai.step_time.model import (
    STEP_TIME_EVENT_NAMES,
    StepTimeClockValues,
    StepTimeRepositorySnapshot,
    StepTimeSourceRow,
    StepTimeValues,
    StepTimeWindow,
)
from traceml_ai.step_time.sqlite import normalize_step_time_events
from tests.step_time.factories import window_from_events

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _PROJECT_ROOT / "src" / "traceml_ai"

_COMPLETE = {
    "input_wait": 5.0,
    "h2d": 5.0,
    "forward": 30.0,
    "backward": 45.0,
    "optimizer_step": 10.0,
    "traced_step_time": 95.0,
}


def _row(
    source_id: int,
    rank: int,
    step: int,
    values: Mapping[str, float],
    *,
    gpu: bool = True,
) -> StepTimeSourceRow:
    return StepTimeSourceRow(
        source_id=source_id,
        global_rank=rank,
        step=step,
        metrics={
            metric: StepTimeClockValues(
                cpu_ms=float(value * 2.0 if gpu else value),
                gpu_ms=float(value) if gpu else None,
            )
            for metric, value in values.items()
        },
    )


def _snapshot(
    profiles: Mapping[int, Mapping[str, float]],
    *,
    steps: tuple[int, ...] = (1, 2),
    gpu: bool = True,
    training_strategy: str = "ddp",
) -> StepTimeRepositorySnapshot:
    rows = []
    source_id = 0
    for rank, profile in sorted(profiles.items()):
        for step in steps:
            source_id += 1
            rows.append(_row(source_id, rank, step, profile, gpu=gpu))
    return StepTimeRepositorySnapshot(
        rows=tuple(rows),
        global_ranks=tuple(sorted(profiles)),
        training_strategy=training_strategy,
    )


def _raw_events(
    values: Mapping[str, float],
    *,
    gpu: bool,
) -> dict:
    events = {}
    for metric, value in values.items():
        events[STEP_TIME_EVENT_NAMES[metric]] = {
            "cuda:0" if gpu else "cpu": {
                "cpu_ms": value * 2.0 if gpu else value,
                "gpu_ms": value if gpu else None,
            }
        }
    return events


def _metric(window: StepTimeWindow, name: str):
    return next(metric for metric in window.metrics if metric.metric == name)


def test_analyzer_builds_one_typed_fact_set() -> None:
    window = StepTimeAnalyzer().analyze(
        _snapshot({0: _COMPLETE}),
        window_size=2,
    )

    assert window.clock == "gpu"
    assert window.steps == [1, 2]
    assert len(window.rank_facts) == 1
    rank = window.rank_facts[0]
    assert rank.global_rank == 0
    assert tuple(step.step for step in rank.steps) == (1, 2)
    assert rank.steps[0].values.forward_ms == 30.0
    assert rank.average.compute_ms == 85.0
    assert rank.average.residual_ms == 5.0
    assert rank.average.step_time_ms == 100.0
    assert rank.average.traced_step_time_ms == 95.0
    assert rank.average.step_time_cpu_ms == 200.0
    assert rank.average.step_time_gpu_ms == 100.0
    assert rank.average.traced_step_time_cpu_ms == 190.0
    assert rank.average.traced_step_time_gpu_ms == 95.0
    assert rank.average.dataloader_fetch_cpu_ms == 10.0

    assert window.step_ranks == (0,)
    assert window.compute_ranks == (0,)
    assert window.straggler_ranks == (0,)
    assert window.input_wait_share == pytest.approx(0.05)
    assert window.h2d_share == pytest.approx(0.05)
    assert window.compute_share == pytest.approx(0.85)
    assert window.residual_share == pytest.approx(0.05)

    summary_statistics = {
        metric.metric: metric
        for metric in window.metrics
        if metric.metric
        in {
            "compute",
            "dataloader_fetch",
            "step_time_cpu",
            "step_time_gpu",
            "traced_step_time_cpu",
            "traced_step_time_gpu",
        }
    }
    assert summary_statistics["compute"].median_total == pytest.approx(85.0)
    dataloader = summary_statistics["dataloader_fetch"]
    assert dataloader.median_total == pytest.approx(10.0)
    assert summary_statistics["step_time_cpu"].median_total == pytest.approx(
        200.0
    )
    assert summary_statistics["step_time_gpu"].median_total == pytest.approx(
        100.0
    )
    assert summary_statistics["traced_step_time_cpu"].median_total == (
        pytest.approx(190.0)
    )
    assert summary_statistics["traced_step_time_gpu"].median_total == (
        pytest.approx(95.0)
    )
    assert all(metric.series is None for metric in summary_statistics.values())


def test_missing_and_measured_zero_keep_distinct_metric_cohorts() -> None:
    zero_forward = {**_COMPLETE, "forward": 0.0}
    missing_forward = {
        key: value for key, value in _COMPLETE.items() if key != "forward"
    }
    window = StepTimeAnalyzer().analyze(
        _snapshot({0: zero_forward, 1: missing_forward}),
        window_size=2,
    )

    assert window.rank(0) is not None
    assert window.rank(1) is not None
    assert window.rank(0).average.forward_ms == 0.0
    assert window.rank(1).average.forward_ms is None
    assert window.rank(0).average.compute_ms == 55.0
    assert window.rank(1).average.compute_ms is None
    assert window.ranks_for("forward") == (0,)
    assert window.compute_ranks == (0,)

    forward = _metric(window, "forward")
    assert forward.median_total == 0.0
    assert forward.representative_rank == 0
    assert forward.representative_total == 0.0
    assert forward.series is not None
    assert forward.series.median == [0.0, 0.0]


def test_occurrence_signals_fill_zero_but_fixed_signals_abstain() -> None:
    intermittent_occurrence = StepTimeAnalyzer().analyze(
        StepTimeRepositorySnapshot(
            rows=(
                _row(1, 0, 1, _COMPLETE),
                _row(
                    2,
                    0,
                    2,
                    {
                        key: value
                        for key, value in _COMPLETE.items()
                        if key not in {"h2d", "optimizer_step"}
                    },
                ),
            ),
            global_ranks=(0,),
        ),
        window_size=2,
    )
    average = intermittent_occurrence.rank_facts[0].average
    assert average.h2d_ms == pytest.approx(2.5)
    assert average.optimizer_step_ms == pytest.approx(5.0)
    assert average.compute_ms == pytest.approx(80.0)
    assert average.residual_ms == pytest.approx(12.5)

    intermittent_fixed = StepTimeAnalyzer().analyze(
        StepTimeRepositorySnapshot(
            rows=(
                _row(1, 0, 1, _COMPLETE),
                _row(
                    2,
                    0,
                    2,
                    {
                        key: value
                        for key, value in _COMPLETE.items()
                        if key != "forward"
                    },
                ),
            ),
            global_ranks=(0,),
        ),
        window_size=2,
    )
    fixed_average = intermittent_fixed.rank_facts[0].average
    assert fixed_average.forward_ms is None
    assert fixed_average.compute_ms is None
    assert fixed_average.residual_ms is None


def test_cpu_fallback_and_strategy_specific_straggler_cohorts() -> None:
    profiles = {
        0: {
            "input_wait": 5.0,
            "backward": 45.0,
            "traced_step_time": 95.0,
        },
        1: _COMPLETE,
    }
    ddp = StepTimeAnalyzer().analyze(
        _snapshot(profiles, gpu=False, training_strategy="ddp"),
        window_size=2,
    )
    fsdp = StepTimeAnalyzer().analyze(
        _snapshot(profiles, gpu=False, training_strategy="fsdp"),
        window_size=2,
    )

    assert ddp.clock == "cpu"
    assert ddp.rank(0).average.step_time_cpu_ms == 100.0
    assert ddp.rank(0).average.traced_step_time_cpu_ms == 95.0
    assert ddp.rank(0).average.step_time_gpu_ms is None
    assert ddp.rank(0).average.traced_step_time_gpu_ms is None
    assert ddp.straggler_ranks == (0, 1)
    assert fsdp.straggler_ranks == (1,)

    gpu_rows = _snapshot({0: _COMPLETE}).rows
    partial_gpu = StepTimeAnalyzer().analyze(
        StepTimeRepositorySnapshot(
            rows=tuple(
                StepTimeSourceRow(
                    source_id=row.source_id,
                    global_rank=row.global_rank,
                    step=row.step,
                    metrics={
                        metric: StepTimeClockValues(
                            cpu_ms=timing.cpu_ms,
                            gpu_ms=(
                                None if metric == "forward" else timing.gpu_ms
                            ),
                        )
                        for metric, timing in row.metrics.items()
                    },
                )
                for row in gpu_rows
            ),
            global_ranks=(0,),
        ),
        window_size=2,
    )
    partial_values = partial_gpu.rank(0).average
    assert partial_gpu.clock == "cpu"
    assert partial_values.step_time_ms == 200.0
    assert partial_values.traced_step_time_ms == 190.0
    assert partial_values.step_time_gpu_ms == 100.0
    assert partial_values.traced_step_time_gpu_ms == 95.0


_NO_INPUT = {
    metric: value
    for metric, value in _COMPLETE.items()
    if metric != "input_wait"
}


def test_gpu_clock_when_no_row_has_input_wait() -> None:
    # A loop with no DataLoader: every phase and the traced envelope carry
    # GPU times, and no row on any rank carries input wait on either clock.
    window = StepTimeAnalyzer().analyze(
        _snapshot({0: _NO_INPUT, 1: _NO_INPUT}),
        window_size=2,
    )

    assert window.clock == "gpu"
    average = window.rank(0).average
    assert average.forward_ms == 30.0
    assert average.optimizer_step_ms == 10.0
    assert average.h2d_ms == 5.0
    assert average.traced_step_time_ms == 95.0
    # Input wait stays absent, never 0.0, so outer Step Time stays unknown.
    assert average.input_wait_ms is None
    assert average.step_time_ms is None


@pytest.mark.parametrize(
    "rank1_input_wait",
    [
        pytest.param(
            StepTimeClockValues(cpu_ms=5.0, gpu_ms=None), id="cpu_only"
        ),
        pytest.param(None, id="missing_on_one_rank"),
    ],
)
def test_cpu_clock_when_input_wait_exists_without_gpu_time(
    rank1_input_wait,
) -> None:
    rows = []
    for rank in (0, 1):
        for step in (1, 2):
            row = _row(len(rows) + 1, rank, step, _NO_INPUT)
            metrics = dict(row.metrics)
            if rank == 0:
                metrics["input_wait"] = StepTimeClockValues(10.0, 5.0)
            elif rank1_input_wait is not None:
                metrics["input_wait"] = rank1_input_wait
            rows.append(row._replace(metrics=metrics))
    snapshot = StepTimeRepositorySnapshot(
        rows=tuple(rows), global_ranks=(0, 1), training_strategy="ddp"
    )

    window = StepTimeAnalyzer().analyze(snapshot, window_size=2)

    # Input wait is measured somewhere, so it is not absent: without a GPU
    # value on every row the window keeps the CPU clock.
    assert window.clock == "cpu"


def test_an_input_wait_entry_with_no_duration_counts_as_absent() -> None:
    # The decoder keeps an event whose durations were all invalid as an
    # entry with both clocks None; that is still no measured input wait.
    rows = []
    for step in (1, 2):
        row = _row(step, 0, step, _NO_INPUT)
        metrics = dict(row.metrics)
        metrics["input_wait"] = StepTimeClockValues(None, None)
        rows.append(row._replace(metrics=metrics))
    snapshot = StepTimeRepositorySnapshot(
        rows=tuple(rows), global_ranks=(0,), training_strategy="ddp"
    )

    window = StepTimeAnalyzer().analyze(snapshot, window_size=2)

    assert window.clock == "gpu"
    assert window.rank(0).average.input_wait_ms is None


def test_cpu_clock_without_input_wait_when_traced_time_lacks_gpu() -> None:
    rows = []
    for rank in (0, 1):
        for step in (1, 2):
            row = _row(len(rows) + 1, rank, step, _NO_INPUT)
            if rank == 1:
                metrics = dict(row.metrics)
                metrics["traced_step_time"] = StepTimeClockValues(190.0, None)
                row = row._replace(metrics=metrics)
            rows.append(row)
    snapshot = StepTimeRepositorySnapshot(
        rows=tuple(rows), global_ranks=(0, 1), training_strategy="ddp"
    )

    assert StepTimeAnalyzer().analyze(snapshot, window_size=2).clock == "cpu"


def test_cpu_clock_when_a_present_phase_lacks_gpu_time() -> None:
    rows = []
    for step in (1, 2):
        row = _row(step, 0, step, _NO_INPUT)
        metrics = dict(row.metrics)
        if step == 2:
            metrics["forward"] = StepTimeClockValues(60.0, None)
        rows.append(row._replace(metrics=metrics))
    snapshot = StepTimeRepositorySnapshot(
        rows=tuple(rows), global_ranks=(0,), training_strategy="ddp"
    )

    assert StepTimeAnalyzer().analyze(snapshot, window_size=2).clock == "cpu"


def test_occurrence_phases_keep_their_rule_without_input_wait() -> None:
    rows = []
    for step in (1, 2):
        values = dict(_NO_INPUT)
        if step == 2:
            del values["optimizer_step"]
            del values["h2d"]
        rows.append(_row(step, 0, step, values))
    snapshot = StepTimeRepositorySnapshot(
        rows=tuple(rows), global_ranks=(0,), training_strategy="ddp"
    )

    window = StepTimeAnalyzer().analyze(snapshot, window_size=2)

    assert window.clock == "gpu"
    steps = window.rank(0).steps
    assert steps[0].values.optimizer_step_ms == 10.0
    assert steps[1].values.optimizer_step_ms == 0.0
    assert steps[1].values.h2d_ms == 0.0


@pytest.mark.parametrize(
    ("profiles", "median", "representative", "worst"),
    [
        ({0: 100.0, 1: 300.0}, 200.0, 0, 1),
        ({0: 100.0, 1: 200.0, 2: 300.0}, 200.0, 1, 2),
        ({0: 100.0, 1: 300.0, 2: 300.0}, 300.0, 1, 1),
    ],
)
def test_step_time_metric_names_median_representative_and_worst(
    profiles: Mapping[int, float],
    median: float,
    representative: int,
    worst: int,
) -> None:
    rank_profiles = {
        rank: {"input_wait": 0.0, "traced_step_time": total}
        for rank, total in profiles.items()
    }
    window = StepTimeAnalyzer().analyze(
        _snapshot(rank_profiles, gpu=False),
        window_size=2,
    )

    step_time = _metric(window, "step_time")
    assert step_time.median_total == median
    assert step_time.representative_rank == representative
    assert step_time.representative_total == profiles[representative]
    assert step_time.worst_rank == worst
    assert step_time.worst_total == profiles[worst]


def test_alignment_uses_latest_common_suffix() -> None:
    rows = (
        _row(1, 0, 1, _COMPLETE),
        _row(2, 0, 2, _COMPLETE),
        _row(3, 0, 3, _COMPLETE),
        _row(4, 1, 2, _COMPLETE),
        _row(5, 1, 3, _COMPLETE),
        _row(6, 1, 4, _COMPLETE),
    )
    analyzer = StepTimeAnalyzer()
    aligned = analyzer.analyze(
        StepTimeRepositorySnapshot(rows=rows, global_ranks=(0, 1)),
        window_size=1,
    )
    assert aligned.steps == [3]

    empty = analyzer.analyze(
        StepTimeRepositorySnapshot(
            rows=(
                _row(1, 0, 1, _COMPLETE),
                _row(2, 1, 2, _COMPLETE),
            ),
            global_ranks=(0, 1),
        ),
        window_size=2,
    )
    assert empty.steps == []
    assert empty.rank_facts == ()
    assert empty.coverage.ranks_present == 2


def test_raw_fixture_adapter_matches_direct_normalized_analysis() -> None:
    raw = {
        rank: {step: _raw_events(profile, gpu=True) for step in (1, 2)}
        for rank, profile in {0: _COMPLETE, 1: _COMPLETE}.items()
    }
    adapter_window = window_from_events(raw, max_rows=2)

    rows = []
    source_id = 0
    for rank, step_map in raw.items():
        for step, events in step_map.items():
            source_id += 1
            rows.append(
                StepTimeSourceRow(
                    source_id,
                    rank,
                    step,
                    normalize_step_time_events(events) or {},
                )
            )
    direct_window = StepTimeAnalyzer().analyze(
        StepTimeRepositorySnapshot(
            rows=tuple(rows),
            global_ranks=(0, 1),
        ),
        window_size=2,
    )

    assert adapter_window == direct_window


def test_analyzer_has_one_input_and_no_removed_shapes() -> None:
    signature = inspect.signature(StepTimeAnalyzer.analyze)
    assert tuple(signature.parameters) == ("self", "snapshot", "window_size")
    assert "rank_facts" in {field.name for field in fields(StepTimeWindow)}
    assert "per_rank_timing" not in {
        field.name for field in fields(StepTimeWindow)
    }
    assert not hasattr(StepTimeWindow(), "per_rank_timing")
    assert "iteration_ranks" not in {
        field.name for field in fields(StepTimeWindow)
    }
    assert "step_ranks" in {field.name for field in fields(StepTimeWindow)}

    value_fields = {field.name for field in fields(StepTimeValues)}
    assert {
        "step_time_ms",
        "traced_step_time_ms",
        "step_time_cpu_ms",
        "step_time_gpu_ms",
        "traced_step_time_cpu_ms",
        "traced_step_time_gpu_ms",
    }.issubset(value_fields)
    assert {
        "total_step_ms",
        "total_step_cpu_ms",
        "dataloader_cpu_ms",
    }.isdisjoint(value_fields)

    production = tuple(_SOURCE_ROOT.glob("**/*.py"))
    mapping_owners = {
        path.relative_to(_SOURCE_ROOT).as_posix()
        for path in production
        if "per_rank_timing" in path.read_text(encoding="utf-8")
    }
    assert mapping_owners == set()
    source = "\n".join(path.read_text(encoding="utf-8") for path in production)
    assert "per_rank_step_timing" not in source
    assert "_group_source_rows" not in source
    assert "build_step_time_metrics" not in source

    analyzer_source = (_SOURCE_ROOT / "step_time" / "analysis.py").read_text(
        encoding="utf-8"
    )
    assert "isinstance(" not in analyzer_source
    for forbidden in (
        "traceml_ai.step_time.sqlite",
        "traceml_ai.diagnostics",
        "traceml_ai.renderers",
        "traceml_ai.reporting",
        "nicegui",
        "rich",
    ):
        assert forbidden not in analyzer_source


def test_empty_snapshot_defaults_to_no_observed_ranks() -> None:
    window = StepTimeAnalyzer().analyze(
        StepTimeRepositorySnapshot(),
        window_size=100,
    )
    assert window.rank_universe == ()
    assert window.rank_facts == ()
    assert window.metrics == []
    assert window.coverage.expected_steps == 100
    assert window.coverage.steps_used == 0
