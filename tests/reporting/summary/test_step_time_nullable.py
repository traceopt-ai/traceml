# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Nullable public Step Time metrics in final summaries (issue #258).

Missing signals surface as JSON ``null`` / text ``n/a`` and stay out of
rollups and rank selection; measured zeros stay ``0.0``.
"""

import json
import sqlite3

from traceml_ai.reporting.compare.core import build_compare_payload
from traceml_ai.reporting.sections.step_time import StepTimeSummarySection

_EVENT_NAMES = {
    "input_wait": "_traceml_internal:dataloader_next",
    "h2d": "_traceml_internal:h2d_time",
    "forward": "_traceml_internal:forward_time",
    "backward": "_traceml_internal:backward_time",
    "optimizer_step": "_traceml_internal:optimizer_step",
    "step_time": "_traceml_internal:step_time",
}

_FULL_RANK_MS = {
    "input_wait": 1.0,
    "forward": 5.0,
    "backward": 10.0,
    "optimizer_step": 4.0,
    "step_time": 30.0,
}


def _events_json(values_ms: dict) -> str:
    return json.dumps(
        {
            _EVENT_NAMES[key]: {
                "cpu": {
                    "is_gpu": False,
                    "duration_ms": value,
                    "cpu_ms": value,
                    "gpu_ms": None,
                    "n_calls": 1,
                }
            }
            for key, value in values_ms.items()
        }
    )


def _create_db(path: str, per_rank_events: dict) -> None:
    """Create a step_time db; per_rank_events maps rank -> metric ms map."""
    conn = sqlite3.connect(path)
    try:
        conn.execute(
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
            """
        )
        conn.execute(
            """
            CREATE TABLE runtime_environment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                training_strategy TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO runtime_environment(training_strategy) VALUES (?);",
            ("ddp",),
        )
        world_size = len(per_rank_events)
        rows = []
        row_id = 0
        for rank, values_ms in per_rank_events.items():
            for step in (1, 2):
                row_id += 1
                rows.append(
                    (
                        row_id,
                        rank,
                        rank,
                        rank,
                        world_size,
                        world_size,
                        0,
                        f"worker-{rank}",
                        float(step),
                        step,
                        step,
                        _events_json(values_ms),
                    )
                )
        conn.executemany(
            """
            INSERT INTO step_time_samples(
                recv_ts_ns, rank, global_rank, local_rank, world_size,
                local_world_size, node_rank, hostname, sample_ts_s,
                seq, step, events_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def test_missing_h2d_is_null_and_measured_zero_stays_zero(tmp_path) -> None:
    missing_db = tmp_path / "missing"
    _create_db(str(missing_db), {0: dict(_FULL_RANK_MS)})
    missing = StepTimeSummarySection().build(str(missing_db)).payload

    zero_db = tmp_path / "zero"
    _create_db(str(zero_db), {0: dict(_FULL_RANK_MS, h2d=0.0)})
    zero = StepTimeSummarySection().build(str(zero_db)).payload

    assert missing["global"]["average"]["h2d_ms"] is None
    assert missing["global"]["median"]["h2d_ms"] == {
        "value": None,
        "idx": None,
    }
    assert missing["global"]["worst"]["h2d_ms"] == {
        "value": None,
        "idx": None,
    }
    assert missing["groups"]["rows"]["0"]["metrics"]["h2d_ms"] is None
    # Measured metrics on the same row stay plain floats.
    assert missing["groups"]["rows"]["0"]["metrics"]["step_time_ms"] == 30.0
    assert "H2D n/a" in missing["card"]

    assert zero["global"]["average"]["h2d_ms"] == 0.0
    assert zero["groups"]["rows"]["0"]["metrics"]["h2d_ms"] == 0.0
    assert "H2D 0.0ms" in zero["card"]


def test_h2d_only_rank_keeps_row_and_stays_out_of_rank_selection(
    tmp_path,
) -> None:
    db = tmp_path / "telemetry"
    _create_db(
        str(db),
        {
            0: dict(_FULL_RANK_MS, h2d=2.0),
            1: {"h2d": 6.0},
        },
    )
    summary = StepTimeSummarySection().build(str(db)).payload

    rows = summary["groups"]["rows"]
    assert set(rows) == {"0", "1"}
    # The H2D-only rank keeps its H2D observation and nulls elsewhere.
    assert rows["1"]["metrics"]["h2d_ms"] == 6.0
    assert rows["1"]["metrics"]["step_time_ms"] is None
    assert rows["1"]["metrics"]["total_step_ms"] is None

    # Null metrics stay out of averages and rank picks: the H2D average
    # spans both ranks, total-step rollups only the measured rank.
    assert summary["global"]["average"]["h2d_ms"] == 4.0
    assert summary["global"]["average"]["total_step_ms"] == 31.0
    assert summary["global"]["worst"]["total_step_ms"]["idx"] == "0"
    assert summary["global"]["worst"]["h2d_ms"]["idx"] == "1"


def test_compare_treats_null_metric_as_unavailable(tmp_path) -> None:
    def _summary_payload(compute_ms):
        return {
            "schema_version": 1.7,
            "duration_s": 10.0,
            "system": {},
            "process": {},
            "step_memory": {},
            "step_time": {
                "diagnosis": {
                    "kind": "INCOMPLETE_DATA",
                    "status": "INCOMPLETE DATA",
                },
                "global": {
                    "average": {
                        "total_step_ms": 31.0,
                        "input_wait_ms": 1.0,
                        "h2d_ms": None,
                        "compute_ms": compute_ms,
                        "residual_ms": None,
                        "forward_ms": 5.0,
                        "backward_ms": 10.0,
                        "optimizer_ms": compute_ms,
                    }
                },
            },
        }

    payload = build_compare_payload(
        lhs_payload=_summary_payload(19.0),
        rhs_payload=_summary_payload(None),
        lhs_path=tmp_path / "a" / "final_summary.json",
        rhs_path=tmp_path / "b" / "final_summary.json",
    )

    metrics = payload["sections"]["step_time"]["metrics"]
    # A null side yields no delta: unavailable, never "went to zero".
    assert metrics["compute_ms"]["lhs"] == 19.0
    assert metrics["compute_ms"]["rhs"] is None
    assert metrics["compute_ms"]["delta"] is None
    assert metrics["compute_ms"]["pct_change"] is None
    # Null on both sides is equally quiet.
    assert metrics["residual_ms"]["delta"] is None


def test_single_rank_scope_wording_survives_null_total(tmp_path) -> None:
    db = tmp_path / "telemetry"
    _create_db(str(db), {0: {"h2d": 3.0}})
    summary = StepTimeSummarySection().build(str(db)).payload

    # A single-rank run keeps single-rank wording even when its total
    # step is unmeasured and it cannot win a median/worst pick.
    assert "aligned steps on global rank r0" in summary["card"]
    assert "across 1 global ranks" not in summary["card"]


def test_all_ranks_without_total_step_null_the_rank_picks(tmp_path) -> None:
    db = tmp_path / "telemetry"
    _create_db(str(db), {0: {"h2d": 2.0}, 1: {"h2d": 6.0}})
    summary = StepTimeSummarySection().build(str(db)).payload

    assert summary["global"]["worst"]["total_step_ms"] == {
        "value": None,
        "idx": None,
    }
    assert summary["global"]["average"]["total_step_ms"] is None
    assert summary["global"]["average"]["h2d_ms"] == 4.0
    assert summary["global"]["worst"]["h2d_ms"]["idx"] == "1"
    assert "total n/a" in summary["card"]


def test_partial_compute_triplet_nulls_only_derived_metrics(
    tmp_path,
) -> None:
    db = tmp_path / "telemetry"
    values = {k: v for k, v in _FULL_RANK_MS.items() if k != "optimizer_step"}
    _create_db(str(db), {0: values})
    summary = StepTimeSummarySection().build(str(db)).payload

    metrics = summary["groups"]["rows"]["0"]["metrics"]
    # Measured phases keep their values; only the metrics that need the
    # unmeasured optimizer become null.
    assert metrics["forward_ms"] == 5.0
    assert metrics["backward_ms"] == 10.0
    assert metrics["optimizer_ms"] is None
    assert metrics["compute_ms"] is None
    assert metrics["residual_ms"] is None
    assert metrics["total_step_ms"] == 31.0
