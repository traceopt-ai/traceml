# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Canonical SQLite scenarios for cross-surface Step Time tests.

This module deliberately keeps scenario definition and persistence together.
Contract tests should need only two operations: select a named scenario and
write it to SQLite. Keeping that path shallow avoids repeating event encodings
across the CLI, dashboard, and final-summary test suites.
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from tests.sqlite_fixtures import (
    init_step_time_schema,
    insert_step_time_sample,
    insert_training_strategy,
    sqlite_database,
)

MetricProfile = Mapping[str, float]
RankProfiles = Mapping[int, MetricProfile]

EVENT_NAMES: Mapping[str, str] = {
    "input_wait": "_traceml_internal:dataloader_next",
    "h2d": "_traceml_internal:h2d_time",
    "forward": "_traceml_internal:forward_time",
    "backward": "_traceml_internal:backward_time",
    "optimizer_step": "_traceml_internal:optimizer_step",
    # Instrumentation still emits this historical raw event until its own
    # migration. The scenario key uses the canonical inner-envelope name.
    "traced_step_time": "_traceml_internal:step_time",
}

BALANCED_PROFILE: Mapping[str, float] = {
    "input_wait": 5.0,
    "h2d": 5.0,
    "forward": 30.0,
    "backward": 45.0,
    "optimizer_step": 10.0,
    "traced_step_time": 95.0,
}


@dataclass(frozen=True)
class StepTimeScenario:
    """One deterministic telemetry window shared by every output surface.

    Parameters
    ----------
    name:
        Stable pytest id and contributor-facing scenario name.
    profiles:
        Selected-clock metric values keyed by global rank. Omitting a metric
        represents an unavailable signal; an explicit ``0.0`` is measured.
    steps:
        Completed step ids persisted for every rank.
    clock:
        Expected selected diagnosis clock. GPU scenarios also persist a CPU
        compatibility clock at twice the selected value.
    training_strategy:
        Advisory strategy persisted in ``runtime_environment``.
    jitter:
        Relative per-step variation applied to every metric value, drawn
        from a generator seeded by ``seed``, rank, step and metric. The
        same scenario always writes the same values; nothing is random at
        test time.
    seed:
        Seed for ``jitter``.
    missing_steps:
        Steps a rank never reports, keyed by global rank. The other ranks
        still report them, so the rank drops out of those steps mid-window.
    metric_every:
        Metrics that occur only on some steps: the metric is written on a
        step when ``step % n == 0``, on every rank.
    dropped_events:
        Steps on which one rank's metric is absent, keyed by global rank,
        then metric.
    """

    name: str
    profiles: RankProfiles
    steps: tuple[int, ...]
    clock: str = "cpu"
    training_strategy: str = "ddp"
    jitter: float = 0.0
    seed: int = 0
    missing_steps: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    metric_every: Mapping[str, int] = field(default_factory=dict)
    dropped_events: Mapping[int, Mapping[str, tuple[int, ...]]] = field(
        default_factory=dict
    )


class SQLiteSelectRecorder:
    """Open traced SQLite connections and retain only read statements.

    The recorder is test-side instrumentation. Patch ``sqlite3.connect`` with
    :meth:`connect` only around the production call being measured; database
    setup must happen before the patch so fixture writes are excluded.
    """

    def __init__(self) -> None:
        self._connect = sqlite3.connect
        self.statements: list[str] = []

    def connect(self, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        """Return a connection whose SELECT/WITH statements are recorded."""
        conn = self._connect(*args, **kwargs)
        conn.set_trace_callback(self._record)
        return conn

    @property
    def count(self) -> int:
        """Return the number of recorded read statements."""
        return len(self.statements)

    def _record(self, statement: str) -> None:
        normalized = statement.lstrip().upper()
        if normalized.startswith(("SELECT", "WITH")):
            self.statements.append(statement)


def _profile(**overrides: float) -> dict[str, float]:
    """Return a balanced profile with selected values replaced explicitly."""
    return {**BALANCED_PROFILE, **overrides}


def _without(profile: MetricProfile, metric: str) -> dict[str, float]:
    """Copy a profile while making one metric unavailable."""
    return {key: value for key, value in profile.items() if key != metric}


SCENARIOS: tuple[StepTimeScenario, ...] = (
    StepTimeScenario(
        name="complete_gpu",
        profiles={0: _profile(), 1: _profile()},
        steps=(10, 11, 12, 13),
        clock="gpu",
    ),
    StepTimeScenario(
        name="sparse_missing_forward",
        profiles={
            0: _profile(),
            1: _without(_profile(), "forward"),
        },
        steps=(20, 21, 22, 23),
    ),
    StepTimeScenario(
        name="measured_zero_forward",
        profiles={0: _profile(), 1: _profile(forward=0.0)},
        steps=(30, 31, 32, 33),
    ),
    StepTimeScenario(
        name="single_rank_cpu",
        profiles={0: _profile()},
        steps=(40, 41, 42, 43),
    ),
    StepTimeScenario(
        name="ddp_rank_straggler",
        profiles={
            0: _profile(
                input_wait=100.0,
                h2d=0.0,
                forward=20.0,
                backward=20.0,
                optimizer_step=0.0,
                traced_step_time=40.0,
            ),
            1: _profile(
                input_wait=0.0,
                h2d=0.0,
                forward=20.0,
                backward=120.0,
                optimizer_step=0.0,
                traced_step_time=140.0,
            ),
        },
        steps=tuple(range(50, 74)),
        training_strategy="ddp",
    ),
    StepTimeScenario(
        name="fsdp_rank_straggler",
        profiles={
            0: _profile(
                input_wait=100.0,
                h2d=0.0,
                forward=20.0,
                backward=20.0,
                optimizer_step=0.0,
                traced_step_time=40.0,
            ),
            1: _profile(
                input_wait=0.0,
                h2d=0.0,
                forward=80.0,
                backward=80.0,
                optimizer_step=0.0,
                traced_step_time=160.0,
            ),
        },
        steps=tuple(range(80, 104)),
        training_strategy="fsdp",
    ),
)

# Time-varying windows: per-step variation, a rank that drops out
# mid-window, and metrics that do not occur on every step. They are kept
# apart from SCENARIOS, whose hand-written goldens assume one identical
# payload per rank; the public-schema golden covers both sets.
REALISM_SCENARIOS: tuple[StepTimeScenario, ...] = (
    StepTimeScenario(
        name="jittered_ddp",
        profiles={0: _profile(), 1: _profile(backward=60.0)},
        steps=tuple(range(110, 134)),
        jitter=0.1,
        seed=7,
    ),
    StepTimeScenario(
        name="rank_missing_steps",
        profiles={0: _profile(), 1: _profile()},
        steps=tuple(range(140, 164)),
        missing_steps={1: (150, 151, 152, 153)},
    ),
    StepTimeScenario(
        name="intermittent_metrics",
        profiles={0: _profile(), 1: _profile()},
        steps=tuple(range(170, 194)),
        # Gradient accumulation over four steps, and H2D on every other
        # step: occurrence-driven metrics.
        metric_every={"optimizer_step": 4, "h2d": 2},
        # A phase that should occur on every step is missing once.
        dropped_events={1: {"forward": (180,)}},
    ),
)

ALL_SCENARIOS: tuple[StepTimeScenario, ...] = SCENARIOS + REALISM_SCENARIOS

SCENARIOS_BY_NAME: Mapping[str, StepTimeScenario] = {
    scenario.name: scenario for scenario in ALL_SCENARIOS
}


def _event_payload(profile: MetricProfile, clock: str) -> dict:
    """Encode one rank profile as the sampler's wire ``events`` mapping."""
    events = {}
    for metric, value in profile.items():
        event_name = EVENT_NAMES[metric]
        selected = float(value)
        cpu_ms = selected * 2.0 if clock == "gpu" else selected
        events[event_name] = {
            "cuda:0" if clock == "gpu" else "cpu": {
                "is_gpu": clock == "gpu",
                "duration_ms": cpu_ms,
                "cpu_ms": cpu_ms,
                "gpu_ms": selected if clock == "gpu" else None,
                "n_calls": 1,
            }
        }
    return events


def _step_profile(
    scenario: StepTimeScenario,
    global_rank: int,
    step: int,
) -> dict[str, float]:
    """One rank's metric values on one step, after the scenario's rules."""
    dropped = scenario.dropped_events.get(global_rank, {})
    values = {}
    for metric, value in scenario.profiles[global_rank].items():
        every = scenario.metric_every.get(metric)
        if every is not None and step % every != 0:
            continue
        if step in dropped.get(metric, ()):
            continue
        if scenario.jitter:
            draw = random.Random(
                f"{scenario.seed}:{global_rank}:{step}:{metric}"
            )
            value = value * (1.0 + draw.uniform(-1.0, 1.0) * scenario.jitter)
        values[metric] = value
    return values


def create_step_time_database(
    path: str | Path,
    scenario: StepTimeScenario,
) -> None:
    """Persist a scenario through the production projection schema.

    Parameters
    ----------
    path:
        SQLite database path to create.
    scenario:
        Canonical scenario whose rows and runtime strategy are persisted.
    """
    world_size = len(scenario.profiles)
    with sqlite_database(path, init_step_time_schema) as conn:
        insert_training_strategy(conn, scenario.training_strategy)
        sequence = 0
        for global_rank in sorted(scenario.profiles):
            missing = scenario.missing_steps.get(global_rank, ())
            for step in scenario.steps:
                if step in missing:
                    continue
                events = _event_payload(
                    _step_profile(scenario, global_rank, step),
                    scenario.clock,
                )
                sequence += 1
                insert_step_time_sample(
                    conn,
                    row_id=sequence,
                    rank=global_rank,
                    step=step,
                    events=events,
                    local_rank=global_rank,
                    world_size=world_size,
                    local_world_size=world_size,
                    node_rank=0,
                    hostname="worker-0",
                    ts=float(step),
                    seq=sequence,
                )


__all__ = [
    "ALL_SCENARIOS",
    "BALANCED_PROFILE",
    "REALISM_SCENARIOS",
    "SCENARIOS",
    "SCENARIOS_BY_NAME",
    "SQLiteSelectRecorder",
    "StepTimeScenario",
    "create_step_time_database",
]
