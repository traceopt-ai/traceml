# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from tests.sqlite_fixtures import (
    insert_process_sample,
    insert_step_memory_sample,
    insert_step_time_sample,
    insert_system_sample,
    summary_database,
)
from traceml_ai.core.summaries import SummaryResult
from traceml_ai.reporting.final import (
    FinalReportGenerator,
    build_summary_payload,
)


@dataclass(frozen=True)
class _StaticSection:
    name: str
    duration_s: float | None = None

    def build(self, db_path: str) -> SummaryResult:
        title = self.name.replace("_", " ").title()
        payload = {"card": f"TraceML {title} Summary\n- Status: OK"}
        if self.duration_s is not None:
            payload["duration_s"] = self.duration_s
        return SummaryResult(
            section=self.name,
            payload=payload,
            text=payload["card"],
        )


@dataclass(frozen=True)
class _BrokenSection:
    name: str

    def build(self, db_path: str) -> SummaryResult:
        raise RuntimeError("section failed")


@dataclass(frozen=True)
class _PayloadSection:
    name: str
    payload: dict

    def build(self, db_path: str) -> SummaryResult:
        return SummaryResult(
            section=self.name,
            payload=self.payload,
            text=str(self.payload.get("card", "")),
        )


def _generator(*sections) -> FinalReportGenerator:
    return FinalReportGenerator(sections=sections)


def _diagnosis(
    kind: str,
    status: str,
    *,
    severity: str = "info",
    summary: str = "summary",
    action: str = "action",
    phase: str | None = None,
    **extra,
) -> dict:
    diagnosis = {
        "kind": kind,
        "status": status,
        "severity": severity,
        "summary": summary,
        "action": action,
        "phase": phase,
    }
    diagnosis.update(extra)
    return diagnosis


def _payload(
    *,
    metadata: dict,
    diagnosis: dict,
    global_summary: dict | None = None,
    groups: dict | None = None,
    card: str = "ORIGINAL SECTION CARD",
) -> dict:
    return {
        "metadata": metadata,
        "diagnosis": diagnosis,
        "issues": [diagnosis],
        "global": global_summary or {},
        "groups": groups or {"by": "global_rank", "rows": {}},
        "units": {},
        "card": card,
    }


def _point(value: float, idx: int) -> dict:
    return {"value": value, "idx": str(idx)}


def _status_payload(status: str) -> dict:
    return _payload(
        metadata={},
        diagnosis=_diagnosis(status, status),
        card=f"{status} SECTION CARD",
    )


def _final_payload(step_time: dict, *, system: dict | None = None) -> dict:
    return build_summary_payload(
        "fake.db",
        generator=_generator(
            _PayloadSection("system", system or _status_payload("NORMAL")),
            _PayloadSection("process", _status_payload("NORMAL")),
            _PayloadSection("step_time", step_time),
            _PayloadSection("step_memory", _status_payload("BALANCED")),
        ),
    )


def test_final_summary_fixture_schema_contains_all_sections(tmp_path) -> None:
    db_path = tmp_path / "final.db"
    with summary_database(db_path) as conn:
        insert_system_sample(
            conn,
            row_id=1,
            rank=0,
            ts=1.0,
            gpu_available=False,
            gpu_count=0,
            gpu_util=None,
        )
        insert_process_sample(
            conn,
            row_id=1,
            rank=0,
            ts=1.0,
            gpu_available=False,
            gpu_count=0,
        )

    payload = build_summary_payload(str(db_path))

    assert payload["schema_version"] == 1.8
    assert set(payload) == {
        "schema_version",
        "generated_at",
        "duration_s",
        "analysis_window",
        "meta",
        "primary_diagnosis",
        "system",
        "process",
        "step_time",
        "step_memory",
        "text",
    }
    assert set(payload["meta"]) == {
        "run_name",
        "mode",
        "world_size",
        "nodes_observed",
        "gpus_observed",
    }
    assert payload["primary_diagnosis"]["kind"] == (
        "INSUFFICIENT_STEP_TIME_DATA"
    )
    assert payload["meta"] == {
        "run_name": None,
        "mode": "single_node",
        "world_size": 1,
        "nodes_observed": 1,
        "gpus_observed": 0,
    }
    for key in ("system", "process", "step_time", "step_memory"):
        assert "metadata" in payload[key]
        assert "card" in payload[key]
        assert "diagnosis" in payload[key]
        assert payload[key]["issues"]
        assert payload[key]["diagnosis"] == payload[key]["issues"][0]
        assert "- Next:" not in payload[key]["card"]
    assert payload["system"]["diagnosis"]["status"] == "NORMAL"
    assert "NO GPU" not in payload["system"]["card"]
    assert "TraceML Run Summary" in payload["text"]
    assert "Verdict: INSUFFICIENT STEP-TIME DATA" in payload["text"]
    assert "Next:" in payload["text"]


def test_final_summary_aligns_sections_to_step_derived_time_window(
    tmp_path,
) -> None:
    db_path = tmp_path / "aligned.db"
    with summary_database(db_path) as conn:
        for step in range(1, 501):
            ts = step * 6.0
            insert_step_time_sample(
                conn,
                row_id=step,
                rank=0,
                step=step,
                ts=ts,
                traced_step_time=10.0,
            )
            insert_step_memory_sample(
                conn,
                row_id=step,
                rank=0,
                step=step,
                ts=ts,
                alloc=100.0,
                reserved=200.0,
            )

        for index in range(500):
            ts = 1200.0 + index * (1800.0 / 499.0)
            insert_system_sample(
                conn,
                row_id=index + 1,
                rank=0,
                ts=ts,
                gpu_available=False,
                gpu_count=0,
            )
        for index in range(400):
            ts = 1200.0 + index * (1800.0 / 399.0)
            insert_process_sample(
                conn,
                row_id=index + 1,
                rank=0,
                ts=ts,
                gpu_available=False,
                gpu_count=0,
            )

        conn.execute(
            """
            CREATE TABLE history_retention_state (
                table_name TEXT PRIMARY KEY,
                max_deleted_sample_ts_s REAL,
                max_deleted_step INTEGER
            );
            """
        )
        conn.executemany(
            """
            INSERT INTO history_retention_state(
                table_name, max_deleted_sample_ts_s, max_deleted_step
            ) VALUES (?, ?, ?);
            """,
            [
                ("system_samples", 1199.0, None),
                ("process_samples", 1300.0, None),
                ("step_time_samples", 1199.0, 199),
                ("step_memory_samples", 1199.0, 199),
            ],
        )

    payload = build_summary_payload(str(db_path), history_retention_s=1800.0)
    window = payload["analysis_window"]

    assert window["start_step"] == 200
    assert window["end_step"] == 500
    assert window["start_ts_s"] == 1200.0
    assert window["end_ts_s"] == 3000.0
    assert window["sections"]["step_time"]["samples"] == 301
    assert window["sections"]["step_memory"]["samples"] == 301
    assert window["sections"]["system"]["samples"] == 500
    assert window["sections"]["process"]["samples"] == 400
    assert window["sections"]["process"]["coverage"] == "partial"
    assert payload["duration_s"] is None
    for section in ("system", "process", "step_time", "step_memory"):
        metadata = payload[section]["metadata"]
        assert metadata["analysis_start_ts_s"] == 1200.0
        assert metadata["analysis_end_ts_s"] == 3000.0


def test_final_report_generator_preserves_summary_schema_and_order():
    payload = build_summary_payload(
        "fake.db",
        generator=_generator(
            _StaticSection("system"),
            _StaticSection("process", duration_s=12.5),
            _StaticSection("step_time", duration_s=10.0),
            _StaticSection("step_memory"),
        ),
    )

    assert payload["schema_version"] == 1.8
    assert payload["duration_s"] is None
    assert list(payload.keys()) == [
        "schema_version",
        "generated_at",
        "duration_s",
        "analysis_window",
        "meta",
        "primary_diagnosis",
        "system",
        "process",
        "step_time",
        "step_memory",
        "text",
    ]
    assert payload["meta"] == {
        "run_name": None,
        "mode": "no_data",
        "world_size": None,
        "nodes_observed": None,
        "gpus_observed": None,
    }
    assert payload["primary_diagnosis"]["kind"] == (
        "INSUFFICIENT_STEP_TIME_DATA"
    )
    text = payload["text"]
    assert "TraceML Run Summary" in text
    assert "10.0s" not in text
    assert "Verdict: INSUFFICIENT STEP-TIME DATA" in text
    # The verdict card replaced the old section-status and evidence tables.
    assert "Section Status" not in text
    assert "System Evidence" not in text
    assert "Step Time Evidence" not in text


def test_final_report_generator_fails_open_for_one_section():
    payload = build_summary_payload(
        "fake.db",
        generator=_generator(
            _StaticSection("system"),
            _BrokenSection("process"),
            _StaticSection("step_time"),
            _StaticSection("step_memory"),
        ),
    )

    assert payload["process"]["metadata"]["mode"] == "no_data"
    assert payload["process"]["diagnosis"]["status"] == "NO DATA"
    assert payload["process"]["diagnosis"] == payload["process"]["issues"][0]
    assert payload["process"]["global"]["index_by"] == "global_rank"
    assert payload["process"]["groups"] == {
        "by": "global_rank",
        "rows": {},
    }
    assert payload["process"]["units"] == {}
    assert payload["primary_diagnosis"]["kind"] == (
        "INSUFFICIENT_STEP_TIME_DATA"
    )
    assert "TraceML Run Summary" in payload["text"]
    assert "Verdict: INSUFFICIENT STEP-TIME DATA" in payload["text"]


def test_final_text_uses_single_process_average_layout():
    step_diag = _diagnosis(
        "INPUT_BOUND",
        "INPUT-BOUND",
        severity="crit",
        summary="Input wait is 48.5% of the typical GPU Step Time.",
        action="Increase workers, prefetch, or storage throughput.",
    )
    step_time = _payload(
        metadata={"global_ranks_used": 1},
        diagnosis=step_diag,
        global_summary={
            "window": {"steps_analyzed": 60, "diagnosis_clock": "gpu"},
            "average": {
                "dataloader_fetch_cpu_ms": 120.0,
                "input_wait_ms": 130.8,
                "step_time_ms": 269.9,
                "traced_step_time_ms": 139.1,
                "compute_ms": 6.9,
                "residual_ms": 1.3,
                "h2d_ms": 0.2,
            },
        },
        card="STEP TIME ORIGINAL CARD",
    )
    system = _payload(
        metadata={"mode": "single_node", "gpus_observed": 1},
        diagnosis=_diagnosis("LOW_GPU_UTILIZATION", "LOW GPU UTIL"),
        global_summary={
            "average": {
                "cpu_percent": 18.4,
                "gpu_util_percent": 0.0,
                "gpu_mem_bytes": 570_000_000.0,
                "gpu_temp_c": 30.0,
            }
        },
        groups={"by": "node_rank", "rows": {}},
        card="SYSTEM ORIGINAL CARD",
    )

    payload = _final_payload(step_time, system=system)

    text = payload["text"]
    assert "Verdict: INPUT-BOUND  (CRITICAL)" in text
    # The terminal card presents the stored primary summary and action.
    assert "Why: Input wait is 48.5% of the typical GPU Step Time." in text
    assert "Next: Increase workers, prefetch, or storage throughput." in text
    assert "STEP TIMING (Window Average), GPU Clock" in text
    assert "Step Time           269.9 ms  100%" in text
    assert "├─ Input Wait       130.8 ms   48%" in text
    assert "Traced Step Time" not in text
    assert "├─ Compute            6.9 ms    3%" in text
    assert "SYSTEM METRICS: LOW GPU UTIL" in text
    assert "GPU util               0%" in text
    # Single-process cards carry no distributed comparison tables.
    assert "Section Status" not in text
    assert "Median" not in text
    assert "Worst" not in text
    assert "Skew" not in text
    assert "node" not in text
    assert "1 rank" in text
    # DataLoader fetch is supplemental; it is never a timing-tree row.
    assert "DataLoader Fetch" not in text
    assert "DataLoader fetch: 120.0 ms (CPU, supplemental)" in text
    assert payload["step_time"]["card"] == "STEP TIME ORIGINAL CARD"
    assert payload["system"]["card"] == "SYSTEM ORIGINAL CARD"


def test_final_text_omits_never_measured_step_metrics():
    step_diag = _diagnosis(
        "INCOMPLETE_DATA",
        "INCOMPLETE DATA",
        summary="Missing timing signals prevent a reliable diagnosis: h2d.",
        action="Instrument the missing phases.",
    )
    step_time = _payload(
        metadata={"global_ranks_used": 1},
        diagnosis=step_diag,
        global_summary={
            "window": {"steps_analyzed": 60, "diagnosis_clock": "cpu"},
            "average": {
                "dataloader_fetch_cpu_ms": 120.0,
                "input_wait_ms": 130.8,
                "step_time_ms": 269.9,
                "traced_step_time_ms": 139.1,
                "compute_ms": None,
                "residual_ms": None,
                "h2d_ms": None,
            },
        },
        card="STEP TIME ORIGINAL CARD",
    )

    payload = _final_payload(step_time)

    text = payload["text"]

    # Never-measured signals are omitted and explained in words. The card
    # never prints a placeholder value for a signal that was not measured.
    assert "n/a" not in text
    assert "H2D" not in text
    assert "Compute" not in text
    assert "Residual" not in text
    assert "Verdict: INSUFFICIENT STEP-TIME DATA" in text


def test_final_text_uses_selected_step_time_for_phase_shares():
    step_time = _payload(
        metadata={"global_ranks_used": 1},
        diagnosis=_diagnosis("COMPUTE_BOUND", "COMPUTE-BOUND"),
        global_summary={
            "window": {"steps_analyzed": 60, "diagnosis_clock": "gpu"},
            "average": {
                "dataloader_fetch_cpu_ms": 0.5,
                "input_wait_ms": 2.0,
                "step_time_ms": 52.0,
                "traced_step_time_ms": 50.0,
                "compute_ms": 48.0,
                "residual_ms": 1.0,
                "h2d_ms": 1.0,
            },
        },
    )

    payload = _final_payload(step_time)

    text = payload["text"]
    # Shares use the selected-clock step_time_ms denominator.
    assert "Step Time            52.0 ms  100%" in text
    assert "Traced Step Time" not in text
    assert "├─ Compute           48.0 ms   92%" in text
    assert "DataLoader Fetch" not in text
    assert "DataLoader fetch: 0.5 ms (CPU, supplemental)" in text
    assert "Total" not in text


def test_final_text_includes_h2d_bound_diagnosis():
    step_time = _payload(
        metadata={"global_ranks_used": 1},
        diagnosis=_diagnosis(
            "H2D_BOUND",
            "H2D-BOUND",
            severity="crit",
            summary="H2D transfer is 14.3% of the typical GPU Step Time.",
            action="Inspect pinned memory and batch transfers.",
            share_pct=0.143,
        ),
        global_summary={
            "window": {"steps_analyzed": 60, "diagnosis_clock": "gpu"},
            "average": {
                "dataloader_fetch_cpu_ms": 40.0,
                "input_wait_ms": 40.0,
                "step_time_ms": 140.0,
                "traced_step_time_ms": 100.0,
                "h2d_ms": 20.0,
                "compute_ms": 70.0,
                "residual_ms": 10.0,
            },
        },
    )

    payload = _final_payload(step_time)

    assert "Verdict: H2D-BOUND  (CRITICAL)" in payload["text"]
    assert "Why: H2D transfers took 14% of Step Time." in payload["text"]
    assert "├─ H2D               20.0 ms   14%" in payload["text"]


def test_final_text_uses_diagnosed_straggler_rank_rows():
    step_diag = _diagnosis(
        "INPUT_STRAGGLER",
        "INPUT STRAGGLER",
        severity="crit",
        summary=("r0 waited 264.5 ms for input, compared with 13.8 ms on r1."),
        phase="input",
        action=(
            "Inspect input wait, collate_fn, preprocessing, and storage "
            "on the slow rank."
        ),
        evidence={
            "culprit_rank": 0,
            "victim_rank": 1,
            "visible_metric": "backward",
            "visible_cost_ms": 250.7,
        },
    )
    step_time = _payload(
        metadata={"global_ranks_used": 2},
        diagnosis=step_diag,
        global_summary={
            "window": {"steps_analyzed": 60, "diagnosis_clock": "gpu"},
            "median": {
                "dataloader_fetch_cpu_ms": _point(3.8, 1),
                "input_wait_ms": _point(13.8, 1),
                "step_time_ms": _point(303.7, 1),
                "traced_step_time_ms": _point(299.9, 1),
                "compute_ms": _point(259.5, 1),
                "residual_ms": _point(40.5, 1),
                "h2d_ms": _point(0.2, 1),
            },
            "worst": {
                "dataloader_fetch_cpu_ms": _point(254.5, 0),
                "input_wait_ms": _point(264.5, 0),
                "step_time_ms": _point(304.1, 0),
                "traced_step_time_ms": _point(49.6, 0),
                "compute_ms": _point(261.0, 0),
                "residual_ms": _point(42.1, 0),
                "h2d_ms": _point(0.4, 0),
            },
        },
        groups={
            "by": "global_rank",
            "rows": {
                "0": {
                    "identity": {"global_rank": 0, "node_rank": 0},
                    "metrics": {"input_wait_ms": 264.5},
                },
                "1": {
                    "identity": {"global_rank": 1, "node_rank": 1},
                    "metrics": {
                        "dataloader_fetch_cpu_ms": 3.8,
                        "input_wait_ms": 13.8,
                        "step_time_ms": 303.7,
                        "traced_step_time_ms": 299.9,
                        "compute_ms": 259.5,
                        "residual_ms": 40.5,
                        "h2d_ms": 0.2,
                    },
                },
            },
        },
    )
    system = _payload(
        metadata={"mode": "multi_node", "nodes_observed": 2},
        diagnosis=_diagnosis("LOW_GPU_UTILIZATION", "LOW GPU UTIL"),
        global_summary={
            "median": {
                "cpu_percent": _point(18.4, 0),
                "gpu_util_percent": _point(14.0, 0),
                "gpu_mem_bytes": _point(6_200_000_000.0, 0),
                "gpu_temp_c": _point(42.0, 0),
            },
            "worst": {
                "cpu_percent": _point(71.2, 1),
                "gpu_util_percent": _point(0.0, 0),
                "gpu_mem_bytes": _point(8_900_000_000.0, 1),
                "gpu_temp_c": _point(58.0, 1),
            },
        },
        groups={
            "by": "node_rank",
            "rows": {
                "0": {"identity": {"node_rank": 0}, "metrics": {}},
                "1": {"identity": {"node_rank": 1}, "metrics": {}},
            },
        },
    )

    payload = _final_payload(step_time, system=system)

    text = payload["text"]
    assert "Verdict: INPUT STRAGGLER  (CRITICAL)" in text
    assert payload["primary_diagnosis"]["summary"] == (
        "r0 waited 264.5 ms for input, compared with 13.8 ms on r1."
    )
    assert "Why: R0/N0 waited 264.5 ms for input; R1/N1" in text
    assert "13.8 ms for input." in text
    assert (
        "Next: Inspect input wait, collate_fn, preprocessing, and storage "
        "on the" in text
    )
    assert "STEP TIMING (Median R1/N1), GPU Clock" in text
    assert "├─ Input Wait        13.8 ms    5%" in text
    assert "Step Time           303.7 ms  100%" in text
    assert "Input comparison:" not in text
    assert "x median" not in text
    assert "◀ cause" not in text
    # The old wide median/worst/skew/scope table is gone.
    assert "Skew" not in text
    assert "rank=r" not in text
    assert "node=n" not in text


def test_reporting_final_is_the_summary_orchestration_owner():
    import traceml_ai.reporting.final as reporting_final

    assert reporting_final.generate_summary is not None
    assert reporting_final.build_summary_payload is build_summary_payload
