"""Shared payload builders for HTML-report renderer tests.

These construct minimal-but-realistic ``final_summary.json`` payloads
(the dict shaped at reporting/final.py) so renderer tests do not depend
on any sibling repo's sample artifacts.
"""

from typing import Any, Dict, Optional

import pytest


def _make_section(
    *,
    metric_names,
    average: Dict[str, float],
    severity: str = "info",
    kind: str = "NORMAL",
    status: str = "NORMAL",
    summary: str = "No issues.",
    action: Optional[str] = None,
    issues=None,
    rows: Optional[Dict[str, Any]] = None,
    by: str = "global_rank",
    units: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    diagnosis = {
        "kind": kind,
        "status": status,
        "severity": severity,
        "summary": summary,
        "action": action,
    }
    return {
        "metadata": {"section_metric_names": list(metric_names)},
        "diagnosis": diagnosis,
        "issues": [diagnosis] if issues is None else issues,
        "global": {
            "index_by": by,
            "average": dict(average),
            "median": {
                m: {"value": v, "idx": "0"} for m, v in average.items()
            },
            "worst": {m: {"value": v, "idx": "0"} for m, v in average.items()},
            "window": {},
        },
        "groups": {"by": by, "rows": rows or {}},
        "units": units or {},
        "card": "",
    }


def _make_payload(
    *,
    meta: Any = "default",
    schema_version: Any = 1.4,
    primary_diagnosis: Optional[Any] = None,
    step_time: Optional[Dict[str, Any]] = None,
    step_memory: Optional[Dict[str, Any]] = None,
    system: Optional[Dict[str, Any]] = None,
    process: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if meta == "default":
        meta = {
            "run_name": "demo-run",
            "mode": "single_node",
            "world_size": 2,
            "nodes_observed": 1,
            "gpus_observed": 2,
        }
    st_metrics = ["total_step_ms", "dataloader_ms", "compute_ms"]
    sm_metrics = ["peak_allocated_bytes", "peak_reserved_bytes"]
    payload = {
        "schema_version": schema_version,
        "generated_at": "2026-06-10T12:00:00+00:00",
        "duration_s": 120.0,
        "meta": meta,
        "step_time": (
            step_time
            if step_time is not None
            else _make_section(
                metric_names=st_metrics,
                average={
                    "total_step_ms": 100.0,
                    "dataloader_ms": 20.0,
                    "compute_ms": 75.0,
                },
                kind="BALANCED",
                status="BALANCED",
                summary="Balanced step.",
                units={"time": "ms"},
                rows={
                    "0": {
                        "identity": {"global_rank": 0, "hostname": "h0"},
                        "metrics": {
                            "total_step_ms": 99.0,
                            "dataloader_ms": 19.0,
                            "compute_ms": 75.0,
                        },
                    },
                    "1": {
                        "identity": {"global_rank": 1, "hostname": "h0"},
                        "metrics": {
                            "total_step_ms": 101.0,
                            "dataloader_ms": 21.0,
                            "compute_ms": 75.0,
                        },
                    },
                },
            )
        ),
        "step_memory": (
            step_memory
            if step_memory is not None
            else _make_section(
                metric_names=sm_metrics,
                average={
                    "peak_allocated_bytes": 2.0e9,
                    "peak_reserved_bytes": 2.3e9,
                },
                kind="BALANCED",
                status="BALANCED",
                summary="Balanced memory.",
                units={"memory": "bytes"},
                rows={
                    "0": {
                        "identity": {"global_rank": 0, "hostname": "h0"},
                        "metrics": {
                            "peak_allocated_bytes": 2.0e9,
                            "peak_reserved_bytes": 2.3e9,
                        },
                    },
                },
            )
        ),
        "system": (
            system
            if system is not None
            else _make_section(
                metric_names=["gpu_util_percent", "gpu_mem_percent"],
                average={"gpu_util_percent": 90.0, "gpu_mem_percent": 40.0},
                by="node_rank",
                units={"util": "%"},
                rows={
                    "0": {
                        "identity": {"node_rank": 0, "hostname": "h0"},
                        "metrics": {
                            "gpu_util_percent": 90.0,
                            "gpu_mem_percent": 40.0,
                        },
                    },
                },
            )
        ),
        "process": (
            process
            if process is not None
            else _make_section(
                metric_names=[
                    "gpu_mem_reserved_bytes",
                    "gpu_mem_headroom_bytes",
                ],
                average={
                    "gpu_mem_reserved_bytes": 2.3e9,
                    "gpu_mem_headroom_bytes": 22.0e9,
                },
                rows={
                    "0": {
                        "identity": {"global_rank": 0, "hostname": "h0"},
                        "metrics": {
                            "gpu_mem_reserved_bytes": 2.3e9,
                            "gpu_mem_headroom_bytes": 22.0e9,
                        },
                    },
                },
            )
        ),
        "text": "TraceML summary card",
    }
    if primary_diagnosis is not None:
        payload["primary_diagnosis"] = primary_diagnosis
    return payload


@pytest.fixture
def make_section():
    return _make_section


@pytest.fixture
def make_payload():
    return _make_payload
