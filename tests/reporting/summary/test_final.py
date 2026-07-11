# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

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
) -> dict:
    return {
        "kind": kind,
        "status": status,
        "severity": severity,
        "summary": summary,
        "action": action,
        "phase": phase,
    }


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

    assert payload["schema_version"] == 1.6
    assert payload["duration_s"] == 10.0
    assert list(payload.keys()) == [
        "schema_version",
        "generated_at",
        "duration_s",
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
    assert "TraceML Run Summary | duration 10.0s" in payload["text"]
    assert "TraceML Verdict:" in payload["text"]
    assert "Section Status" in payload["text"]
    assert "System Evidence" in payload["text"]
    assert "Step Time Evidence" in payload["text"]


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
    assert "Process" in payload["text"]


def test_final_text_uses_single_process_average_layout():
    step_diag = _diagnosis(
        "INPUT_BOUND",
        "INPUT-BOUND",
        severity="crit",
        action="Increase workers, prefetch, or storage throughput.",
    )
    step_time = _payload(
        metadata={"global_ranks_used": 1},
        diagnosis=step_diag,
        global_summary={
            "window": {"steps_analyzed": 60, "diagnosis_clock": "gpu"},
            "average": {
                "total_step_ms": 139.1,
                "dataloader_ms": 120.0,
                "input_wait_ms": 130.8,
                "step_time_ms": 139.1,
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

    payload = build_summary_payload(
        "fake.db",
        generator=_generator(
            _PayloadSection("system", system),
            _PayloadSection("process", _status_payload("NORMAL")),
            _PayloadSection("step_time", step_time),
            _PayloadSection("step_memory", _status_payload("BALANCED")),
        ),
    )

    text = payload["text"]
    assert "TraceML Verdict: INPUT-BOUND / CRITICAL" in text
    assert "Why: Input wait was 130.8ms before a 139.1ms traced step." in text
    assert "Next: Increase workers, prefetch, or storage throughput." in text
    assert "System Evidence" in text
    assert "Metric            Average" in text
    assert "Step Time Evidence" in text
    assert "Phase             Average           Share" in text
    assert "Total             139.1ms           compat" in text
    assert "Input Wait        130.8ms           94.0%" in text
    assert "Dataloader        120.0ms           compat" in text
    assert "Step Time         139.1ms           100.0%" in text
    assert "Median" not in text
    assert "Worst" not in text
    assert "Skew" not in text
    assert "rank=r" not in text
    assert "node=n" not in text
    assert payload["step_time"]["card"] == "STEP TIME ORIGINAL CARD"
    assert payload["system"]["card"] == "SYSTEM ORIGINAL CARD"


def test_final_text_uses_selected_step_time_for_phase_shares():
    step_time = _payload(
        metadata={"global_ranks_used": 1},
        diagnosis=_diagnosis("COMPUTE_BOUND", "COMPUTE-BOUND"),
        global_summary={
            "window": {"steps_analyzed": 60, "diagnosis_clock": "gpu"},
            "average": {
                "total_step_ms": 10.5,
                "dataloader_ms": 0.5,
                "input_wait_ms": 2.0,
                "step_time_ms": 50.0,
                "compute_ms": 48.0,
                "residual_ms": 1.0,
                "h2d_ms": 1.0,
            },
        },
    )

    payload = build_summary_payload(
        "fake.db",
        generator=_generator(
            _PayloadSection("system", _status_payload("NORMAL")),
            _PayloadSection("process", _status_payload("NORMAL")),
            _PayloadSection("step_time", step_time),
            _PayloadSection("step_memory", _status_payload("BALANCED")),
        ),
    )

    text = payload["text"]
    assert "Total             10.5ms            compat" in text
    assert "Dataloader        0.5ms             compat" in text
    assert "Step Time         50.0ms            100.0%" in text
    assert "Compute           48.0ms            96.0%" in text
    assert "457.1%" not in text
    assert "476.2%" not in text


def test_final_text_uses_multi_process_comparison_layout():
    step_diag = _diagnosis(
        "INPUT_STRAGGLER",
        "INPUT STRAGGLER",
        severity="crit",
        phase="input",
        action=(
            "Inspect input wait, collate_fn, preprocessing, and storage "
            "on the slow rank."
        ),
    )
    step_time = _payload(
        metadata={"global_ranks_used": 2},
        diagnosis=step_diag,
        global_summary={
            "window": {"steps_analyzed": 60, "diagnosis_clock": "gpu"},
            "median": {
                "total_step_ms": _point(303.7, 1),
                "dataloader_ms": _point(3.8, 1),
                "input_wait_ms": _point(13.8, 1),
                "step_time_ms": _point(299.9, 1),
                "compute_ms": _point(259.5, 1),
                "residual_ms": _point(40.5, 1),
                "h2d_ms": _point(0.2, 1),
            },
            "worst": {
                "total_step_ms": _point(304.1, 0),
                "dataloader_ms": _point(254.5, 0),
                "input_wait_ms": _point(264.5, 0),
                "step_time_ms": _point(49.6, 0),
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
                    "metrics": {},
                },
                "1": {
                    "identity": {"global_rank": 1, "node_rank": 1},
                    "metrics": {},
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

    payload = build_summary_payload(
        "fake.db",
        generator=_generator(
            _PayloadSection("system", system),
            _PayloadSection("process", _status_payload("NORMAL")),
            _PayloadSection("step_time", step_time),
            _PayloadSection("step_memory", _status_payload("BALANCED")),
        ),
    )

    text = payload["text"]
    assert "TraceML Verdict: INPUT STRAGGLER / CRITICAL" in text
    assert (
        "Rank r0 input wait was 264.5ms vs median rank r1 at 13.8ms." in text
    )
    assert (
        "Metric          Median        Worst         Skew        Scope" in text
    )
    assert "GPU Util        14.0%         0.0%          14.0pp" in text
    assert "node=n1" in text
    assert (
        "Phase           Median        Worst         Skew        Scope" in text
    )
    assert "Input Wait      13.8ms        264.5ms       1816.7%" in text
    assert "rank=r0 node=n0" in text
    assert "Average" not in text


def test_reporting_final_is_the_summary_orchestration_owner():
    import traceml_ai.reporting.final as reporting_final

    assert reporting_final.generate_summary is not None
    assert reporting_final.build_summary_payload is build_summary_payload
