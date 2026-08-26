# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""SQLite fixture coverage for final-report summary sections.

These tests keep section assertions schema-oriented instead of snapshotting
full cards. The goal is to make contributor changes safe while leaving
copy/layout free to evolve intentionally.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.sqlite_fixtures import (
    insert_process_sample as _insert_process_sample,
    insert_step_memory_sample as _insert_step_memory_sample,
    insert_step_time_sample as _insert_step_time_sample,
    insert_system_sample as _insert_system_sample,
    summary_database,
)
from traceml_ai.reporting.schema import BaseSectionPayload
from traceml_ai.reporting.sections.process import ProcessSummarySection
from traceml_ai.reporting.sections.step_memory import StepMemorySummarySection
from traceml_ai.reporting.sections.step_time import StepTimeSummarySection
from traceml_ai.reporting.sections.system import SystemSummarySection

SECTION_KEYS = {
    "metadata",
    "diagnosis",
    "issues",
    "global",
    "groups",
    "units",
    "card",
}

GLOBAL_KEYS = {
    "index_by",
    "window",
    "average",
    "median",
    "worst",
}

GLOBAL_WINDOW_KEYS = {
    "kind",
    "alignment",
    "samples",
    "steps_analyzed",
    "start_step",
    "end_step",
    "completed_step",
}
OPTIONAL_GLOBAL_WINDOW_KEYS = {"diagnosis_clock"}

METADATA_KEYS = {
    "mode",
    "duration_s",
    "samples",
    "nodes_expected",
    "nodes_observed",
    "nodes_coverage",
    "nodes_partial",
    "gpus_observed",
    "global_ranks_seen",
    "global_ranks_used",
    "training_total_steps",
    "training_latest_step",
    "section_metric_names",
}

GROUP_ROW_KEYS = {
    "identity",
    "metrics",
}

IDENTITY_KEYS = {
    "global_rank",
    "local_rank",
    "node_rank",
    "hostname",
    "local_world_size",
    "world_size",
}

ISSUE_KEYS = {
    "kind",
    "status",
    "severity",
    "summary",
    "action",
    "metric",
    "phase",
    "score",
    "share_pct",
    "skew_pct",
    "ranks",
    "evidence",
}


def _assert_section_shape(payload: dict, *, group_by: str) -> None:
    assert set(payload) == SECTION_KEYS
    assert set(payload["metadata"]) == METADATA_KEYS
    assert set(payload["diagnosis"]) == ISSUE_KEYS
    assert isinstance(payload["issues"], list)
    assert payload["issues"]
    assert payload["diagnosis"] == payload["issues"][0]
    for issue in payload["issues"]:
        assert set(issue) == ISSUE_KEYS
    assert set(payload["global"]) == GLOBAL_KEYS
    assert payload["global"]["index_by"] in {"node_rank", "global_rank"}
    assert payload["global"]["index_by"] == group_by
    window_keys = set(payload["global"]["window"])
    assert GLOBAL_WINDOW_KEYS <= window_keys
    assert window_keys <= GLOBAL_WINDOW_KEYS | OPTIONAL_GLOBAL_WINDOW_KEYS
    metric_names = payload["metadata"]["section_metric_names"]
    if metric_names is not None:
        expected_metrics = set(metric_names)
        assert set(payload["global"]["average"]) == expected_metrics
        assert set(payload["global"]["median"]) == expected_metrics
        assert set(payload["global"]["worst"]) == expected_metrics
        for metric in expected_metrics:
            assert set(payload["global"]["median"][metric]) == {
                "value",
                "idx",
            }
            assert set(payload["global"]["worst"][metric]) == {
                "value",
                "idx",
            }
    assert payload["groups"]["by"] == group_by
    assert isinstance(payload["groups"]["rows"], dict)
    for row in payload["groups"]["rows"].values():
        assert set(row) == GROUP_ROW_KEYS
        assert set(row["identity"]) == IDENTITY_KEYS
        if metric_names is not None:
            assert set(row["metrics"]) == set(metric_names)


def _valid_base_section_kwargs() -> dict:
    point = {"value": 1.0, "idx": "0"}
    return {
        "metadata": {"section_metric_names": ["cpu_percent"]},
        "diagnosis": None,
        "issues": [],
        "global_summary": {
            "index_by": "global_rank",
            "window": {},
            "average": {"cpu_percent": 1.0},
            "median": {"cpu_percent": point},
            "worst": {"cpu_percent": point},
        },
        "groups": {"by": "global_rank", "rows": {}},
        "units": {},
        "card": "",
    }


@pytest.mark.parametrize(
    ("case", "messages"),
    [
        ("global-metric", ("section_metric_names",)),
        ("group-index", ("groups.by",)),
        ("group-metric", ("groups.rows", "section_metric_names")),
        ("group-field", ("identity and metrics",)),
    ],
)
def test_base_section_payload_rejects_contract_mismatch(
    case: str,
    messages: tuple[str, ...],
) -> None:
    kwargs = _valid_base_section_kwargs()
    if case == "global-metric":
        kwargs["global_summary"]["average"]["extra_metric"] = 2.0
    elif case == "group-index":
        kwargs["groups"]["by"] = "node_rank"
    else:
        row = {"identity": {}, "metrics": {"cpu_percent": 1.0}}
        if case == "group-metric":
            row["metrics"]["extra_metric"] = 2.0
        else:
            row["diagnosis"] = None
        kwargs["groups"]["rows"]["0"] = row

    with pytest.raises(ValueError) as caught:
        BaseSectionPayload(**kwargs).to_json()

    for message in messages:
        assert message in str(caught.value)


def test_summary_sections_handle_empty_tables_with_stable_schema(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "empty.db"
    with summary_database(db_path):
        pass

    sections = (
        SystemSummarySection(),
        ProcessSummarySection(),
        StepTimeSummarySection(),
        StepMemorySummarySection(),
    )

    results = {
        section.name: section.build(str(db_path)) for section in sections
    }

    assert results["system"].payload["metadata"]["samples"] == 0
    assert results["process"].payload["metadata"]["samples"] == 0
    assert results["step_time"].payload["metadata"]["mode"] == "no_data"
    assert (
        results["step_memory"].payload["global"]["window"]["steps_analyzed"]
        == 0
    )
    assert "steps_used" not in results["step_memory"].payload["metadata"]
    _assert_section_shape(results["system"].payload, group_by="node_rank")
    _assert_section_shape(
        results["process"].payload,
        group_by="global_rank",
    )
    _assert_section_shape(
        results["step_time"].payload,
        group_by="global_rank",
    )
    _assert_section_shape(
        results["step_memory"].payload,
        group_by="global_rank",
    )
    for name, result in results.items():
        assert result.section == name
        assert "card" in result.payload
        assert "diagnosis" in result.payload
        assert isinstance(result.payload["issues"], list)
        assert result.text == result.payload["card"]


def test_summary_sections_cover_single_rank_gpu_run(tmp_path: Path) -> None:
    db_path = tmp_path / "single_rank.db"
    with summary_database(db_path) as conn:
        _insert_system_sample(
            conn,
            row_id=1,
            rank=0,
            ts=1.0,
            gpu_available=True,
            gpu_count=1,
            gpu_util=70.0,
        )
        _insert_process_sample(
            conn,
            row_id=1,
            rank=0,
            ts=1.0,
            gpu_available=True,
            gpu_count=1,
        )
        for step in range(1, 5):
            _insert_step_time_sample(
                conn,
                row_id=step,
                rank=0,
                step=step,
                traced_step_time=10.0,
            )
            _insert_step_memory_sample(
                conn,
                row_id=step,
                rank=0,
                step=step,
                alloc=100.0 + step,
                reserved=200.0 + step,
            )

    system = SystemSummarySection().build(str(db_path)).payload
    process = ProcessSummarySection().build(str(db_path)).payload
    step_time = StepTimeSummarySection().build(str(db_path)).payload
    step_memory = StepMemorySummarySection().build(str(db_path)).payload

    assert system["metadata"]["nodes_coverage"] == "1/1"
    assert system["metadata"]["gpus_observed"] == 1
    assert system["metadata"]["mode"] == "single_node"
    assert system["global"]["average"]["gpu_util_percent"] is not None
    _assert_section_shape(system, group_by="node_rank")
    _assert_section_shape(process, group_by="global_rank")
    _assert_section_shape(step_time, group_by="global_rank")
    _assert_section_shape(step_memory, group_by="global_rank")
    assert set(system["groups"]["rows"]) == {"0"}
    assert process["metadata"]["global_ranks_seen"] == 1
    assert step_time["metadata"]["mode"] == "single_node"
    assert step_time["metadata"]["global_ranks_seen"] == 1
    assert step_time["global"]["median"]["step_time_ms"]["value"] == 11.0
    assert step_time["units"] == {"time": "ms"}
    assert step_memory["metadata"]["global_ranks_seen"] == 1
    assert step_memory["metadata"]["global_ranks_used"] == 1
    assert "aligned_steps_analyzed" not in step_memory["metadata"]
    assert "steps_used" not in step_memory["metadata"]
    assert step_memory["global"]["window"]["steps_analyzed"] == 4
    assert set(step_memory["global"]["average"]) == {
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    }


def test_summary_sections_cover_multi_rank_aligned_run(tmp_path: Path) -> None:
    db_path = tmp_path / "multi_rank.db"
    with summary_database(db_path) as conn:
        gpu_mem_by_rank = {0: 2_500.0, 1: 5_000.0}
        for rank in (0, 1):
            _insert_system_sample(
                conn,
                row_id=10 + rank,
                rank=rank,
                ts=1.0 + rank,
                gpu_available=True,
                gpu_count=2,
                gpu_util=80.0 - rank * 20.0,
                world_size=2,
                local_world_size=1,
            )
            conn.execute(
                """
                UPDATE system_gpu_samples
                SET mem_used_bytes = ?
                WHERE recv_ts_ns = ? AND node_rank = ?;
                """,
                (gpu_mem_by_rank[rank], 10 + rank, rank),
            )
            _insert_process_sample(
                conn,
                row_id=10 + rank,
                rank=rank,
                ts=1.0 + rank,
                gpu_available=True,
                gpu_count=2,
            )
            for step in range(1, 6):
                row_id = rank * 100 + step
                _insert_step_time_sample(
                    conn,
                    row_id=row_id,
                    rank=rank,
                    step=step,
                    traced_step_time=10.0 + rank,
                )
                _insert_step_memory_sample(
                    conn,
                    row_id=row_id,
                    rank=rank,
                    step=step,
                    alloc=100.0 + rank * 20.0 + step,
                    reserved=200.0 + rank * 30.0 + step,
                    device=f"cuda:{rank}",
                    world_size=2,
                    local_world_size=1,
                )

    step_time = StepTimeSummarySection().build(str(db_path)).payload
    step_memory = StepMemorySummarySection().build(str(db_path)).payload
    process = ProcessSummarySection().build(str(db_path)).payload

    system = SystemSummarySection().build(str(db_path)).payload
    assert system["global"]["average"]["gpu_headroom_bytes"] == 6_250.0
    assert system["global"]["worst"]["gpu_headroom_bytes"]["value"] == 5_000.0
    assert step_time["metadata"]["mode"] == "multi_node"
    assert step_time["metadata"]["global_ranks_seen"] == 2
    assert step_time["global"]["window"]["steps_analyzed"] == 5
    assert "window_size" not in step_time["global"]["window"]
    assert "aligned_steps_analyzed" not in step_time["metadata"]
    assert set(step_time["groups"]["rows"]) == {"0", "1"}
    assert step_memory["metadata"]["global_ranks_seen"] == 2
    assert step_memory["metadata"]["global_ranks_used"] == 2
    assert "aligned_steps_analyzed" not in step_memory["metadata"]
    assert "steps_used" not in step_memory["metadata"]
    assert step_memory["global"]["window"]["steps_analyzed"] == 5
    assert set(step_memory["groups"]["rows"]) == {"0", "1"}
    assert (
        step_memory["global"]["median"]["peak_allocated_bytes"]["idx"]
        in step_memory["groups"]["rows"]
    )
    assert (
        step_memory["global"]["median"]["peak_reserved_bytes"]["idx"]
        in step_memory["groups"]["rows"]
    )
    assert process["metadata"]["global_ranks_seen"] == 2
    assert set(process["groups"]["rows"]) == {"0", "1"}


def test_step_memory_section_reports_no_gpu_without_throwing(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "no_gpu.db"
    with summary_database(db_path) as conn:
        _insert_system_sample(
            conn,
            row_id=1,
            rank=0,
            ts=1.0,
            gpu_available=False,
            gpu_count=0,
            gpu_util=None,
        )
        _insert_process_sample(
            conn,
            row_id=1,
            rank=0,
            ts=1.0,
            gpu_available=False,
            gpu_count=0,
        )
        _insert_step_memory_sample(
            conn,
            row_id=1,
            rank=0,
            step=1,
            alloc=None,
            reserved=None,
            device=None,
        )

    payload = StepMemorySummarySection().build(str(db_path)).payload

    assert payload["metadata"]["training_total_steps"] == 2
    assert payload["diagnosis"]["status"] == "NO GPU"
    assert payload["diagnosis"]["action"]
    assert payload["diagnosis"] == payload["issues"][0]
    assert "- Next:" not in payload["card"]
    assert payload["card"].index("- Diagnosis:") < payload["card"].index(
        "- Scope:"
    )
    assert payload["card"].index("- Scope:") < payload["card"].index(
        "- Stats:"
    )
    assert payload["card"].index("- Stats:") < payload["card"].index("- Why:")
    assert payload["global"]["window"]["steps_analyzed"] == 0
    assert "steps_used" not in payload["global"]["window"]
    assert payload["issues"][0]["kind"] == "NO_GPU"
