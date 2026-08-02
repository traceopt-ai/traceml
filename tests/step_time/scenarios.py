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

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

MetricProfile = Mapping[str, float]
RankProfiles = Mapping[int, MetricProfile]

EVENT_NAMES: Mapping[str, str] = {
    "input_wait": "_traceml_internal:dataloader_next",
    "h2d": "_traceml_internal:h2d_time",
    "forward": "_traceml_internal:forward_time",
    "backward": "_traceml_internal:backward_time",
    "optimizer_step": "_traceml_internal:optimizer_step",
    "step_time": "_traceml_internal:step_time",
}

BALANCED_PROFILE: Mapping[str, float] = {
    "input_wait": 5.0,
    "h2d": 5.0,
    "forward": 30.0,
    "backward": 45.0,
    "optimizer_step": 10.0,
    "step_time": 95.0,
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
    """

    name: str
    profiles: RankProfiles
    steps: tuple[int, ...]
    clock: str = "cpu"
    training_strategy: str = "ddp"


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
                step_time=40.0,
            ),
            1: _profile(
                input_wait=0.0,
                h2d=0.0,
                forward=20.0,
                backward=120.0,
                optimizer_step=0.0,
                step_time=140.0,
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
                step_time=40.0,
            ),
            1: _profile(
                input_wait=0.0,
                h2d=0.0,
                forward=80.0,
                backward=80.0,
                optimizer_step=0.0,
                step_time=160.0,
            ),
        },
        steps=tuple(range(80, 104)),
        training_strategy="fsdp",
    ),
)

SCENARIOS_BY_NAME: Mapping[str, StepTimeScenario] = {
    scenario.name: scenario for scenario in SCENARIOS
}


def _event_payload(profile: MetricProfile, clock: str) -> dict:
    """Encode one rank profile in the production ``events_json`` shape."""
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


def create_step_time_database(
    path: str | Path,
    scenario: StepTimeScenario,
) -> None:
    """Persist a scenario using the columns read by all Step Time surfaces.

    Parameters
    ----------
    path:
        SQLite database path to create.
    scenario:
        Canonical scenario whose rows and runtime strategy are persisted.
    """
    db_path = str(path)
    world_size = len(scenario.profiles)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE step_time_samples (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                recv_ts_ns         INTEGER NOT NULL,
                rank               INTEGER,
                global_rank        INTEGER,
                local_rank         INTEGER,
                world_size         INTEGER,
                local_world_size   INTEGER,
                node_rank          INTEGER,
                hostname           TEXT,
                sample_ts_s        REAL,
                seq                INTEGER,
                step               INTEGER,
                events_json        TEXT NOT NULL
            );

            CREATE TABLE runtime_environment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                training_strategy TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO runtime_environment(training_strategy) VALUES (?);",
            (scenario.training_strategy,),
        )

        rows = []
        sequence = 0
        for global_rank, profile in sorted(scenario.profiles.items()):
            payload = json.dumps(
                _event_payload(profile, scenario.clock),
                sort_keys=True,
            )
            for step in scenario.steps:
                sequence += 1
                rows.append(
                    (
                        sequence,
                        global_rank,
                        global_rank,
                        global_rank,
                        world_size,
                        world_size,
                        0,
                        "worker-0",
                        float(step),
                        sequence,
                        step,
                        payload,
                    )
                )

        conn.executemany(
            """
            INSERT INTO step_time_samples(
                recv_ts_ns,
                rank,
                global_rank,
                local_rank,
                world_size,
                local_world_size,
                node_rank,
                hostname,
                sample_ts_s,
                seq,
                step,
                events_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


__all__ = [
    "BALANCED_PROFILE",
    "SCENARIOS",
    "SCENARIOS_BY_NAME",
    "SQLiteSelectRecorder",
    "StepTimeScenario",
    "create_step_time_database",
]
