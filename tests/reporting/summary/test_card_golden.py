# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Golden-output lock-down for the terminal run/watch summary card.

Each case builds a final-summary shaped payload and asserts the full rendered
card, byte for byte. Layout changes are intentional changes: update the golden
in the same commit as the renderer.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pytest

from traceml_ai.reporting.primary_diagnosis import build_primary_diagnosis
from traceml_ai.reporting.summary_card import (
    CardDoc,
    build_summary_card,
    card_to_ansi,
    card_to_plain,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _issue(
    kind: str,
    status: str,
    *,
    severity: str = "info",
    summary: str = "",
    action: str = "",
    **extra: Any,
) -> Dict[str, Any]:
    """Build one section issue/diagnosis block."""
    issue: Dict[str, Any] = {
        "kind": kind,
        "status": status,
        "severity": severity,
        "summary": summary,
        "action": action,
        "metric": None,
        "phase": None,
        "score": None,
        "share_pct": None,
        "skew_pct": None,
        "ranks": [],
        "evidence": {},
    }
    issue.update(extra)
    return issue


def _section(
    *,
    diagnosis: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    window: Optional[Dict[str, Any]] = None,
    average: Optional[Dict[str, Any]] = None,
    median: Optional[Dict[str, Any]] = None,
    worst: Optional[Dict[str, Any]] = None,
    rows: Optional[Dict[str, Any]] = None,
    issues: Optional[List[Dict[str, Any]]] = None,
    by: str = "global_rank",
) -> Dict[str, Any]:
    """Build one final-summary section payload."""
    return {
        "metadata": dict(metadata or {}),
        "diagnosis": diagnosis,
        "issues": list(issues if issues is not None else [diagnosis]),
        "global": {
            "index_by": by,
            "window": dict(window or {}),
            "average": dict(average or {}),
            "median": dict(median or {}),
            "worst": dict(worst or {}),
        },
        "groups": {"by": by, "rows": dict(rows or {})},
        "units": {},
        "card": "SECTION CARD",
    }


def _point(value: Optional[float], idx: str) -> Dict[str, Any]:
    """Build one median/worst point."""
    return {"value": value, "idx": idx}


def _rank_rows(*ranks: int, node_rank: int = 0) -> Dict[str, Any]:
    """Build grouped rows indexed by global rank."""
    return {
        str(rank): {
            "identity": {
                "global_rank": rank,
                "local_rank": rank,
                "node_rank": node_rank,
                "hostname": None,
                "local_world_size": None,
                "world_size": None,
            },
            "metrics": {},
        }
        for rank in ranks
    }


def _node_rows(*nodes: int) -> Dict[str, Any]:
    """Build grouped rows indexed by node rank."""
    return {
        str(node): {
            "identity": {
                "global_rank": None,
                "local_rank": None,
                "node_rank": node,
                "hostname": None,
                "local_world_size": None,
                "world_size": None,
            },
            "metrics": {},
        }
        for node in nodes
    }


def _card(
    *,
    profile: str,
    system: Dict[str, Any],
    process: Dict[str, Any],
    step_time: Dict[str, Any],
    step_memory: Dict[str, Any],
    meta: Dict[str, Any],
    duration_s: Optional[float],
    artifact_hint: str,
    html_hint: Optional[str] = None,
) -> CardDoc:
    """Build one card document from already-built section payloads."""
    primary = build_primary_diagnosis(
        system_summary=system,
        process_summary=process,
        step_time_summary=step_time,
        step_memory_summary=step_memory,
    )
    return build_summary_card(
        profile=profile,
        primary_diagnosis=primary,
        system_summary=system,
        process_summary=process,
        step_time_summary=step_time,
        step_memory_summary=step_memory,
        duration_s=duration_s,
        meta=meta,
        artifact_hint=artifact_hint,
        html_hint=html_hint,
    )


_NORMAL_PROCESS = _section(diagnosis=_issue("NORMAL", "NORMAL"))


def _meta(
    *,
    run_name: Optional[str] = None,
    mode: str = "single_node",
    world_size: Optional[int] = 1,
    nodes_observed: Optional[int] = 1,
    gpus_observed: Optional[int] = 1,
) -> Dict[str, Any]:
    """Build the run-level meta block."""
    return {
        "run_name": run_name,
        "mode": mode,
        "world_size": world_size,
        "nodes_observed": nodes_observed,
        "gpus_observed": gpus_observed,
    }


def _system_single(
    *,
    diagnosis: Optional[Dict[str, Any]] = None,
    cpu_percent: Optional[float] = None,
    gpu_util_percent: Optional[float] = None,
    ram_bytes: Optional[float] = None,
    ram_percent: Optional[float] = None,
    gpu_mem_bytes: Optional[float] = None,
    gpu_mem_percent: Optional[float] = None,
    gpu_temp_c: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a single-node System section payload."""
    return _section(
        diagnosis=diagnosis or _issue("NORMAL", "NORMAL"),
        metadata={"mode": "single_node", "gpus_observed": 1},
        average={
            "cpu_percent": cpu_percent,
            "ram_bytes": ram_bytes,
            "ram_percent": ram_percent,
            "gpu_util_percent": gpu_util_percent,
            "gpu_mem_bytes": gpu_mem_bytes,
            "gpu_mem_percent": gpu_mem_percent,
            "gpu_temp_c": gpu_temp_c,
        },
        worst={"gpu_temp_c": _point(gpu_temp_c, "0")},
        rows=_node_rows(0),
        by="node_rank",
    )


def _step_memory_single(
    allocated: Optional[float],
    reserved: Optional[float],
    *,
    diagnosis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a single-rank Step Memory section payload."""
    return _section(
        diagnosis=diagnosis or _issue("BALANCED", "BALANCED"),
        metadata={"global_ranks_used": 1},
        worst={
            "peak_allocated_bytes": _point(allocated, "0"),
            "peak_reserved_bytes": _point(reserved, "0"),
        },
        median={
            "peak_allocated_bytes": _point(allocated, "0"),
            "peak_reserved_bytes": _point(reserved, "0"),
        },
        rows=_rank_rows(0),
    )


def _step_time_single(
    *,
    diagnosis: Dict[str, Any],
    steps_analyzed: Optional[int],
    clock: str = "gpu",
    step_time_ms: Optional[float] = None,
    input_wait_ms: Optional[float] = None,
    traced_step_time_ms: Optional[float] = None,
    compute_ms: Optional[float] = None,
    h2d_ms: Optional[float] = None,
    residual_ms: Optional[float] = None,
    dataloader_fetch_cpu_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a single-rank Step Time section payload."""
    return _section(
        diagnosis=diagnosis,
        metadata={"mode": "single_node", "global_ranks_used": 1},
        window={"steps_analyzed": steps_analyzed, "diagnosis_clock": clock},
        average={
            "step_time_ms": step_time_ms,
            "input_wait_ms": input_wait_ms,
            "traced_step_time_ms": traced_step_time_ms,
            "compute_ms": compute_ms,
            "h2d_ms": h2d_ms,
            "residual_ms": residual_ms,
            "dataloader_fetch_cpu_ms": dataloader_fetch_cpu_ms,
        },
        rows=_rank_rows(0),
    )


def run_input_bound_critical() -> CardDoc:
    """run x single GPU, INPUT-BOUND critical (the flagship card)."""
    return _card(
        profile="run",
        system=_system_single(
            diagnosis=_issue("LOW_GPU_UTILIZATION", "LOW GPU UTIL"),
            cpu_percent=18.4,
            gpu_util_percent=24.0,
            gpu_mem_bytes=3.33e9,
            gpu_temp_c=42.0,
        ),
        process=_NORMAL_PROCESS,
        step_time=_step_time_single(
            diagnosis=_issue(
                "INPUT_BOUND",
                "INPUT-BOUND",
                severity="crit",
                summary="Input Wait is 64.0% of the typical GPU Step Time.",
                action="Increase workers, prefetch, or storage throughput.",
                score=0.64,
            ),
            steps_analyzed=256,
            step_time_ms=200.4,
            input_wait_ms=128.0,
            traced_step_time_ms=72.0,
            compute_ms=68.0,
            h2d_ms=0.4,
            residual_ms=3.6,
            dataloader_fetch_cpu_ms=120.0,
        ),
        step_memory=_step_memory_single(2.9e9, 3.2e9),
        meta=_meta(run_name="bert_finetune"),
        duration_s=52.4,
        artifact_hint="logs/bert_finetune/final_summary.json",
    )


def run_healthy() -> CardDoc:
    """run x single GPU, no clear bottleneck."""
    return _card(
        profile="run",
        system=_system_single(
            cpu_percent=42.0,
            gpu_util_percent=92.0,
            gpu_temp_c=61.0,
        ),
        process=_NORMAL_PROCESS,
        step_time=_step_time_single(
            diagnosis=_issue("BALANCED", "BALANCED"),
            steps_analyzed=256,
            step_time_ms=191.0,
            input_wait_ms=2.1,
            traced_step_time_ms=188.9,
            compute_ms=185.0,
            h2d_ms=1.9,
            residual_ms=2.0,
        ),
        step_memory=_step_memory_single(2.9e9, 3.2e9),
        meta=_meta(run_name="bert_finetune"),
        duration_s=48.9,
        artifact_hint="logs/bert_finetune/final_summary.json",
    )


def run_input_bound_with_also() -> CardDoc:
    """run x single GPU, INPUT-BOUND critical plus a step-memory warning."""
    creep = _issue(
        "MEMORY_CREEP",
        "MEMORY CREEP",
        severity="warn",
        summary="Step memory grew 1.2 GB over the run -- possible leak",
        action="Check for retained tensors between steps.",
    )
    return _card(
        profile="run",
        system=_system_single(
            cpu_percent=18.4,
            gpu_util_percent=88.0,
            gpu_temp_c=42.0,
        ),
        process=_NORMAL_PROCESS,
        step_time=_step_time_single(
            diagnosis=_issue(
                "INPUT_BOUND",
                "INPUT-BOUND",
                severity="crit",
                summary="Input Wait is 64.0% of the typical GPU Step Time.",
                action="Increase workers, prefetch, or storage throughput.",
                score=0.64,
            ),
            steps_analyzed=256,
            step_time_ms=200.4,
            input_wait_ms=128.0,
            traced_step_time_ms=72.0,
            compute_ms=68.0,
            h2d_ms=0.4,
            residual_ms=3.6,
        ),
        step_memory=_step_memory_single(3.9e9, 4.6e9, diagnosis=creep),
        meta=_meta(run_name="bert_finetune"),
        duration_s=52.4,
        artifact_hint="logs/bert_finetune/final_summary.json",
    )


def run_low_gpu_utilization() -> CardDoc:
    """run x single GPU, balanced timing with unexplained low GPU util."""
    return _card(
        profile="run",
        system=_system_single(
            diagnosis=_issue("LOW_GPU_UTILIZATION", "LOW GPU UTIL"),
            cpu_percent=17.0,
            gpu_util_percent=22.0,
            gpu_temp_c=44.0,
        ),
        process=_NORMAL_PROCESS,
        step_time=_step_time_single(
            diagnosis=_issue("BALANCED", "BALANCED"),
            steps_analyzed=256,
            step_time_ms=240.0,
            input_wait_ms=3.1,
            traced_step_time_ms=236.9,
            compute_ms=120.3,
            h2d_ms=1.2,
            residual_ms=115.4,
        ),
        step_memory=_step_memory_single(2.9e9, 3.2e9),
        meta=_meta(run_name="bert_finetune"),
        duration_s=61.2,
        artifact_hint="logs/bert_finetune/final_summary.json",
    )


def run_cpu_only_input_bound() -> CardDoc:
    """run x single CPU-only machine, INPUT-BOUND warning."""
    return _card(
        profile="run",
        system=_system_single(cpu_percent=25.4),
        process=_NORMAL_PROCESS,
        step_time=_step_time_single(
            diagnosis=_issue(
                "INPUT_BOUND",
                "INPUT-BOUND",
                severity="warn",
                summary="Input Wait is 14.0% of the typical CPU Step Time.",
                action="Increase workers or prefetch.",
                score=0.14,
            ),
            steps_analyzed=100,
            clock="cpu",
            step_time_ms=1.36,
            input_wait_ms=0.19,
            traced_step_time_ms=1.17,
            compute_ms=1.02,
            h2d_ms=None,
            residual_ms=0.149,
            dataloader_fetch_cpu_ms=0.19,
        ),
        step_memory=_step_memory_single(
            None,
            None,
            diagnosis=_issue("NO_GPU", "NO GPU"),
        ),
        meta=_meta(run_name="quickstart", gpus_observed=0),
        duration_s=2.0,
        artifact_hint="logs/session_20260806_090618/final_summary.json",
    )


def run_not_enough_step_data() -> CardDoc:
    """run x single GPU, not enough completed steps for a diagnosis."""
    return _card(
        profile="run",
        system=_system_single(cpu_percent=17.0, gpu_util_percent=31.0),
        process=_NORMAL_PROCESS,
        step_time=_step_time_single(
            diagnosis=_issue("NO_DATA", "NO DATA"),
            steps_analyzed=12,
        ),
        step_memory=_step_memory_single(None, None),
        meta=_meta(run_name="quickstart"),
        duration_s=4.0,
        artifact_hint="logs/quickstart/final_summary.json",
    )


def run_step_timing_incomplete() -> CardDoc:
    """run x single GPU, step timing missing phase signals."""
    incomplete = _issue(
        "INCOMPLETE_DATA",
        "INCOMPLETE DATA",
        summary="Missing timing signals prevent a reliable diagnosis.",
        action="Instrument the missing phases.",
        evidence={"missing_signals": ["backward", "optimizer"]},
    )
    return _card(
        profile="run",
        system=_system_single(),
        process=_NORMAL_PROCESS,
        step_time=_step_time_single(
            diagnosis=incomplete,
            steps_analyzed=None,
        ),
        step_memory=_step_memory_single(None, None),
        meta=_meta(run_name="quickstart"),
        duration_s=4.0,
        artifact_hint="logs/quickstart/final_summary.json",
    )


def run_multi_input_straggler() -> CardDoc:
    """run x multi rank, input straggler on rank 0."""
    system = _section(
        diagnosis=_issue("LOW_GPU_UTILIZATION", "LOW GPU UTIL"),
        metadata={"mode": "multi_node", "nodes_observed": 2},
        average={"cpu_percent": 22.0, "gpu_util_percent": 14.0},
        median={"gpu_util_percent": _point(14.0, "0")},
        worst={"gpu_util_percent": _point(9.0, "1")},
        rows=_node_rows(0, 1),
        by="node_rank",
    )
    step_time = _section(
        diagnosis=_issue(
            "INPUT_STRAGGLER",
            "INPUT STRAGGLER",
            severity="crit",
            summary="r0 has excess input wait burden relative to victim r1.",
            action="Inspect input wait on the slow rank.",
            phase="input",
            score=0.83,
        ),
        metadata={"mode": "multi_node", "global_ranks_used": 4},
        window={"steps_analyzed": 250, "diagnosis_clock": "gpu"},
        median={
            "step_time_ms": _point(303.7, "1"),
            "input_wait_ms": _point(3.8, "1"),
            "traced_step_time_ms": _point(299.9, "1"),
            "compute_ms": _point(259.5, "1"),
            "h2d_ms": _point(1.1, "1"),
            "residual_ms": _point(39.3, "1"),
        },
        worst={
            "step_time_ms": _point(304.1, "0"),
            "input_wait_ms": _point(254.5, "0"),
            "traced_step_time_ms": _point(300.5, "0"),
            "compute_ms": _point(261.0, "0"),
            "h2d_ms": _point(1.2, "0"),
            "residual_ms": _point(39.5, "0"),
        },
        rows=_rank_rows(0, 1, 2, 3),
    )
    step_memory = _section(
        diagnosis=_issue("BALANCED", "BALANCED"),
        metadata={"global_ranks_used": 4},
        median={"peak_reserved_bytes": _point(8.9e9, "1")},
        worst={"peak_reserved_bytes": _point(9.8e9, "2")},
        rows=_rank_rows(0, 1, 2, 3),
    )
    return _card(
        profile="run",
        system=system,
        process=_NORMAL_PROCESS,
        step_time=step_time,
        step_memory=step_memory,
        meta=_meta(
            run_name="ddp_pretrain",
            mode="multi_node",
            world_size=4,
            nodes_observed=2,
            gpus_observed=4,
        ),
        duration_s=40.1,
        artifact_hint="logs/ddp_pretrain/final_summary.json",
    )


def run_multi_healthy() -> CardDoc:
    """run x multi rank, balanced and even across ranks."""
    system = _section(
        diagnosis=_issue("NORMAL", "NORMAL"),
        metadata={"mode": "single_node", "nodes_observed": 1},
        average={"cpu_percent": 38.0, "gpu_util_percent": 94.0},
        rows=_node_rows(0),
        by="node_rank",
    )
    step_time = _section(
        diagnosis=_issue("BALANCED", "BALANCED"),
        metadata={"mode": "single_node", "global_ranks_used": 4},
        window={"steps_analyzed": 250, "diagnosis_clock": "gpu"},
        median={
            "step_time_ms": _point(152.1, "1"),
            "input_wait_ms": _point(2.0, "1"),
            "traced_step_time_ms": _point(150.1, "1"),
            "compute_ms": _point(145.2, "1"),
            "h2d_ms": _point(1.6, "1"),
            "residual_ms": _point(3.3, "1"),
        },
        worst={
            "step_time_ms": _point(152.7, "2"),
            "input_wait_ms": _point(2.1, "2"),
            "traced_step_time_ms": _point(150.6, "2"),
            "compute_ms": _point(145.6, "2"),
            "h2d_ms": _point(1.7, "2"),
            "residual_ms": _point(3.4, "2"),
        },
        rows=_rank_rows(0, 1, 2, 3),
    )
    step_memory = _section(
        diagnosis=_issue("BALANCED", "BALANCED"),
        metadata={"global_ranks_used": 4},
        median={"peak_reserved_bytes": _point(9.0e9, "1")},
        worst={"peak_reserved_bytes": _point(9.1e9, "2")},
        rows=_rank_rows(0, 1, 2, 3),
    )
    return _card(
        profile="run",
        system=system,
        process=_NORMAL_PROCESS,
        step_time=step_time,
        step_memory=step_memory,
        meta=_meta(
            run_name="ddp_pretrain",
            world_size=4,
            nodes_observed=1,
            gpus_observed=4,
        ),
        duration_s=38.7,
        artifact_hint="logs/ddp_pretrain/final_summary.json",
    )


def run_single_gpu_on_shared_host() -> CardDoc:
    """run x 1 process on a 4-GPU host (real 4xT4 capture)."""
    return _card(
        profile="run",
        system=_system_single(
            diagnosis=_issue("LOW_GPU_UTILIZATION", "LOW GPU UTIL"),
            cpu_percent=0.2285714285714286,
            gpu_util_percent=0.21428571428571427,
            ram_bytes=3986740955.428571,
            ram_percent=1.988849429010849,
            gpu_mem_bytes=498930249.14285713,
            gpu_mem_percent=3.0977666945684526,
            gpu_temp_c=34.5,
        ),
        process=_NORMAL_PROCESS,
        step_time=_step_time_single(
            diagnosis=_issue(
                "BALANCED",
                "BALANCED",
                summary="No dominant bottleneck is visible in this window.",
                action="No action needed.",
            ),
            steps_analyzed=128,
            step_time_ms=34.1507115913555,
            input_wait_ms=2.4584102761000395,
            traced_step_time_ms=31.692301315255463,
            compute_ms=29.361732091289014,
            h2d_ms=None,
            residual_ms=2.3305692239664495,
            dataloader_fetch_cpu_ms=2.4595465511083603,
        ),
        step_memory=_step_memory_single(18838528.0, 23068672.0),
        meta=_meta(
            run_name="session_20260806_101725_5580d3",
            world_size=1,
            nodes_observed=1,
            gpus_observed=4,
        ),
        duration_s=12.133176565170288,
        artifact_hint=(
            "logs/session_20260806_101725_5580d3/final_summary.json"
        ),
    )


def run_multi_residual_heavy() -> CardDoc:
    """run x 4-rank DDP, residual-heavy with a competing step-time issue."""
    system = _section(
        diagnosis=_issue("LOW_GPU_UTILIZATION", "LOW GPU UTIL"),
        metadata={"mode": "single_node", "nodes_observed": 1},
        average={
            "cpu_percent": 2.4,
            "gpu_util_percent": 6.833333333333333,
        },
        rows=_node_rows(0),
        by="node_rank",
    )
    residual_heavy = _issue(
        "RESIDUAL_HEAVY",
        "RESIDUAL-HEAVY",
        severity="warn",
        summary="Residual time is 14.0% of the typical GPU Step Time.",
        action="Investigate untraced work between steps.",
        score=0.13980705864843995,
    )
    # A competing Step Time finding: it must never reach the Also block,
    # which is scoped to resource sections.
    input_bound = _issue(
        "INPUT_BOUND",
        "INPUT-BOUND",
        severity="warn",
        summary="Input wait is 11.0% of the typical GPU Step Time.",
        action="Increase workers, prefetch, or storage throughput.",
        score=0.11,
    )
    step_time = _section(
        diagnosis=residual_heavy,
        issues=[residual_heavy, input_bound],
        metadata={"mode": "single_node", "global_ranks_used": 4},
        window={"steps_analyzed": 128, "diagnosis_clock": "gpu"},
        median={
            "step_time_ms": _point(5.203032052493654, "1"),
            "input_wait_ms": _point(0.575069501879625, "1"),
            "traced_step_time_ms": _point(4.6259870529174805, "1"),
            "compute_ms": _point(3.83626828956767, "1"),
            "h2d_ms": _point(0.08052700001280755, "1"),
            "residual_ms": _point(0.7217999144340865, "1"),
        },
        worst={
            "step_time_ms": _point(5.229208162869327, "2"),
            "input_wait_ms": _point(0.5804139995016158, "1"),
            "traced_step_time_ms": _point(4.6567043997347355, "0"),
            "compute_ms": _point(3.8683359728602227, "3"),
            "h2d_ms": _point(0.08191550013725646, "1"),
            "residual_ms": _point(0.7465767902322114, "1"),
        },
        rows=_rank_rows(0, 1, 2, 3),
    )
    step_memory = _section(
        diagnosis=_issue("BALANCED", "BALANCED"),
        metadata={"global_ranks_used": 4},
        median={"peak_reserved_bytes": _point(23068672.0, "1")},
        worst={"peak_reserved_bytes": _point(25149440.0, "0")},
        rows=_rank_rows(0, 1, 2, 3),
    )
    return _card(
        profile="run",
        system=system,
        process=_NORMAL_PROCESS,
        step_time=step_time,
        step_memory=step_memory,
        meta=_meta(
            run_name="session_20260806_101837_1b81f7",
            world_size=4,
            nodes_observed=1,
            gpus_observed=4,
        ),
        duration_s=2.744549512863159,
        artifact_hint=(
            "logs/session_20260806_101837_1b81f7/final_summary.json"
        ),
    )


def _watch_system(
    *,
    diagnosis: Optional[Dict[str, Any]] = None,
    cpu_percent: float,
    ram_bytes: float,
    ram_percent: float,
    gpu_util_percent: Optional[float] = None,
    gpu_mem_bytes: Optional[float] = None,
    gpu_mem_percent: Optional[float] = None,
    gpu_temp_c: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a single-node System payload for watch cards."""
    return _section(
        diagnosis=diagnosis or _issue("NORMAL", "NORMAL"),
        metadata={"mode": "single_node", "gpus_observed": 1},
        average={
            "cpu_percent": cpu_percent,
            "ram_bytes": ram_bytes,
            "ram_percent": ram_percent,
            "gpu_util_percent": gpu_util_percent,
            "gpu_mem_bytes": gpu_mem_bytes,
            "gpu_mem_percent": gpu_mem_percent,
            "gpu_temp_c": gpu_temp_c,
        },
        worst={"gpu_temp_c": _point(gpu_temp_c, "0")},
        rows=_node_rows(0),
        by="node_rank",
    )


def _watch_process(rss_bytes: float) -> Dict[str, Any]:
    """Build a Process payload for watch cards."""
    return _section(
        diagnosis=_issue("NORMAL", "NORMAL"),
        metadata={"global_ranks_used": 1},
        worst={"ram_bytes": _point(rss_bytes, "0")},
        rows=_rank_rows(0),
    )


_WATCH_STEP_TIME = _section(
    diagnosis=_issue("NO_DATA", "NO DATA"),
    metadata={"global_ranks_used": 1},
    window={"steps_analyzed": 0},
)
_WATCH_STEP_MEMORY = _section(diagnosis=_issue("NO_GPU", "NO GPU"))


def watch_healthy() -> CardDoc:
    """watch x single machine, healthy host."""
    return _card(
        profile="watch",
        system=_watch_system(
            cpu_percent=18.0,
            ram_bytes=6.2e9,
            ram_percent=19.375,
            gpu_util_percent=76.0,
            gpu_mem_bytes=10.3e9,
            gpu_mem_percent=64.375,
            gpu_temp_c=61.0,
        ),
        process=_watch_process(4.1e9),
        step_time=_WATCH_STEP_TIME,
        step_memory=_WATCH_STEP_MEMORY,
        meta=_meta(),
        duration_s=312.0,
        artifact_hint="logs/watch_20260806/final_summary.json",
    )


def watch_low_gpu_utilization() -> CardDoc:
    """watch x single machine, low GPU utilization observation."""
    return _card(
        profile="watch",
        system=_watch_system(
            diagnosis=_issue("LOW_GPU_UTILIZATION", "LOW GPU UTIL"),
            cpu_percent=18.0,
            ram_bytes=6.2e9,
            ram_percent=19.375,
            gpu_util_percent=14.0,
            gpu_mem_bytes=3.3e9,
            gpu_mem_percent=20.625,
            gpu_temp_c=42.0,
        ),
        process=_watch_process(4.1e9),
        step_time=_WATCH_STEP_TIME,
        step_memory=_WATCH_STEP_MEMORY,
        meta=_meta(),
        duration_s=312.0,
        artifact_hint="logs/watch_20260806/final_summary.json",
    )


def watch_memory_pressure() -> CardDoc:
    """watch x single machine, host memory pressure."""
    pressure = _issue(
        "MEMORY_PRESSURE",
        "MEMORY PRESSURE",
        severity="warn",
        summary=(
            "RAM 30.1 / 32.0 GB (94%) -- the host is close to memory "
            "exhaustion."
        ),
        action="reduce DataLoader workers or caching, or move work off "
        "this host.",
    )
    return _card(
        profile="watch",
        system=_watch_system(
            diagnosis=pressure,
            cpu_percent=64.0,
            ram_bytes=30.1e9,
            ram_percent=94.0625,
            gpu_util_percent=81.0,
            gpu_mem_bytes=14.9e9,
            gpu_mem_percent=93.125,
            gpu_temp_c=79.0,
        ),
        process=_watch_process(28.6e9),
        step_time=_WATCH_STEP_TIME,
        step_memory=_WATCH_STEP_MEMORY,
        meta=_meta(),
        duration_s=312.0,
        artifact_hint="logs/watch_20260806/final_summary.json",
    )


def watch_multi_node() -> CardDoc:
    """watch x multi node, healthy hosts."""
    system = _section(
        diagnosis=_issue("NORMAL", "NORMAL"),
        metadata={"mode": "multi_node", "nodes_observed": 3},
        median={
            "cpu_percent": _point(24.0, "0"),
            "ram_percent": _point(31.0, "0"),
            "gpu_util_percent": _point(88.0, "0"),
            "gpu_mem_percent": _point(71.0, "0"),
            "gpu_temp_c": _point(64.0, "0"),
        },
        worst={
            "cpu_percent": _point(71.0, "1"),
            "ram_percent": _point(64.0, "1"),
            "gpu_util_percent": _point(62.0, "2"),
            "gpu_mem_percent": _point(93.0, "0"),
            "gpu_temp_c": _point(81.0, "0"),
        },
        rows=_node_rows(0, 1, 2),
        by="node_rank",
    )
    return _card(
        profile="watch",
        system=system,
        process=_watch_process(4.1e9),
        step_time=_WATCH_STEP_TIME,
        step_memory=_WATCH_STEP_MEMORY,
        meta=_meta(
            mode="multi_node",
            world_size=12,
            nodes_observed=3,
            gpus_observed=12,
        ),
        duration_s=760.0,
        artifact_hint="logs/watch_20260806/final_summary.json",
    )


CASES = {
    "run_input_bound_critical": run_input_bound_critical,
    "run_healthy": run_healthy,
    "run_input_bound_with_also": run_input_bound_with_also,
    "run_low_gpu_utilization": run_low_gpu_utilization,
    "run_cpu_only_input_bound": run_cpu_only_input_bound,
    "run_not_enough_step_data": run_not_enough_step_data,
    "run_step_timing_incomplete": run_step_timing_incomplete,
    "run_multi_input_straggler": run_multi_input_straggler,
    "run_multi_healthy": run_multi_healthy,
    "run_single_gpu_on_shared_host": run_single_gpu_on_shared_host,
    "run_multi_residual_heavy": run_multi_residual_heavy,
    "watch_healthy": watch_healthy,
    "watch_low_gpu_utilization": watch_low_gpu_utilization,
    "watch_memory_pressure": watch_memory_pressure,
    "watch_multi_node": watch_multi_node,
}

WATCH_CASES = tuple(name for name in CASES if name.startswith("watch_"))
SINGLE_MACHINE_CASES = (
    "run_input_bound_critical",
    "run_healthy",
    "run_input_bound_with_also",
    "run_low_gpu_utilization",
    "run_cpu_only_input_bound",
    "run_not_enough_step_data",
    "run_step_timing_incomplete",
    "run_single_gpu_on_shared_host",
    "watch_healthy",
    "watch_low_gpu_utilization",
    "watch_memory_pressure",
)


GOLDENS = {
    "run_input_bound_critical": """\
+----------------------------------------------------------------------------+
|  TraceML Run Summary                                                       |
|  bert_finetune · 1 GPU · 256 steps analyzed · 52.4s                        |
+----------------------------------------------------------------------------+
|                                                                            |
|  Verdict: INPUT-BOUND  (CRITICAL)                                          |
|  Why: input wait is 64% of the typical step; the GPU sits idle while       |
|  batches arrive.                                                           |
|                                                                            |
|  Where a step goes (average, GPU clock)                                    |
|  Step Time           200.4 ms  100%  ████████████████████████████          |
|  ├─ Input Wait       128.0 ms   64%  ██████████████████░░░░░░░░░░  ◀  cause|
|  └─ Traced Step Time  72.0 ms   36%  ██████████░░░░░░░░░░░░░░░░░░          |
|     ├─ Compute        68.0 ms   34%                                        |
|     ├─ H2D             0.4 ms   <1%                                        |
|     └─ Residual        3.6 ms    2%                                        |
|                                                                            |
|  Next: raise DataLoader num_workers, enable pin_memory / prefetch, or check|
|  storage read throughput.                                                  |
|                                                                            |
|  Supporting: GPU util 24% avg -- consistent with input starvation.         |
|  Peak step memory: 2.9 GB allocated · 3.2 GB reserved                      |
|                                                                            |
|  Full evidence: logs/bert_finetune/final_summary.json  (--html-report)     |
+----------------------------------------------------------------------------+""",
    "run_healthy": """\
+----------------------------------------------------------------------------+
|  TraceML Run Summary                                                       |
|  bert_finetune · 1 GPU · 256 steps analyzed · 48.9s                        |
+----------------------------------------------------------------------------+
|                                                                            |
|  Verdict: NO CLEAR BOTTLENECK                                              |
|  Why: step timing is balanced -- no input, transfer, or residual issue.    |
|                                                                            |
|  Where a step goes (average, GPU clock)                                    |
|  Step Time           191.0 ms  100%                                        |
|  ├─ Input Wait         2.1 ms    1%                                        |
|  └─ Traced Step Time 188.9 ms   99%                                        |
|     ├─ Compute       185.0 ms   97%                                        |
|     ├─ H2D             1.9 ms    1%                                        |
|     └─ Residual        2.0 ms    1%                                        |
|                                                                            |
|  System, process, and memory: all normal. GPU util 92% avg.                |
|  Peak step memory: 2.9 GB allocated · 3.2 GB reserved                      |
|                                                                            |
|  Next: training is compute-dominated; for more speed profile kernels       |
|  (torch.profiler / Nsight).                                                |
|                                                                            |
|  Full evidence: logs/bert_finetune/final_summary.json  (--html-report)     |
+----------------------------------------------------------------------------+""",
    "run_input_bound_with_also": """\
+----------------------------------------------------------------------------+
|  TraceML Run Summary                                                       |
|  bert_finetune · 1 GPU · 256 steps analyzed · 52.4s                        |
+----------------------------------------------------------------------------+
|                                                                            |
|  Verdict: INPUT-BOUND  (CRITICAL)                                          |
|  Why: input wait is 64% of the typical step; the GPU sits idle while       |
|  batches arrive.                                                           |
|                                                                            |
|  Where a step goes (average, GPU clock)                                    |
|  Step Time           200.4 ms  100%  ████████████████████████████          |
|  ├─ Input Wait       128.0 ms   64%  ██████████████████░░░░░░░░░░  ◀  cause|
|  └─ Traced Step Time  72.0 ms   36%  ██████████░░░░░░░░░░░░░░░░░░          |
|     ├─ Compute        68.0 ms   34%                                        |
|     ├─ H2D             0.4 ms   <1%                                        |
|     └─ Residual        3.6 ms    2%                                        |
|                                                                            |
|  Next: raise DataLoader num_workers, enable pin_memory / prefetch, or check|
|  storage read throughput.                                                  |
|                                                                            |
|  Also, not the cause of slow steps:                                        |
|  ! Step memory grew 1.2 GB over the run -- possible leak  (WARNING)        |
|  Peak step memory: 3.9 GB allocated · 4.6 GB reserved                      |
|                                                                            |
|  Full evidence: logs/bert_finetune/final_summary.json  (--html-report)     |
+----------------------------------------------------------------------------+""",
    "run_low_gpu_utilization": """\
+----------------------------------------------------------------------------+
|  TraceML Run Summary                                                       |
|  bert_finetune · 1 GPU · 256 steps analyzed · 1m 1s                        |
+----------------------------------------------------------------------------+
|                                                                            |
|  Verdict: LOW GPU UTILIZATION -- cause unclear                             |
|  Why: step timing is balanced, but GPU util averaged 22%. The idle time is |
|  not explained by input or transfer.                                       |
|                                                                            |
|  Where a step goes (average, GPU clock)                                    |
|  Step Time           240.0 ms  100%                                        |
|  ├─ Input Wait         3.1 ms    1%                                        |
|  └─ Traced Step Time 236.9 ms   99%                                        |
|     ├─ Compute       120.3 ms   50%                                        |
|     ├─ H2D             1.2 ms   <1%                                        |
|     └─ Residual      115.4 ms   48%                                        |
|                                                                            |
|  Next: look for untraced work between steps -- validation, checkpointing,  |
|  logging -- or inefficient kernels (torch.profiler).                       |
|                                                                            |
|  Peak step memory: 2.9 GB allocated · 3.2 GB reserved                      |
|                                                                            |
|  Full evidence: logs/bert_finetune/final_summary.json  (--html-report)     |
+----------------------------------------------------------------------------+""",
    "run_cpu_only_input_bound": """\
+----------------------------------------------------------------------------+
|  TraceML Run Summary                                                       |
|  quickstart · CPU only (no GPU detected) · 100 steps analyzed · 2.0s       |
+----------------------------------------------------------------------------+
|                                                                            |
|  Verdict: INPUT-BOUND  (WARNING)                                           |
|  Why: input wait is 14% of the typical step (CPU clock).                   |
|                                                                            |
|  Where a step goes (average, CPU clock)                                    |
|  Step Time             1.4 ms  100%  ████████████████████████████          |
|  ├─ Input Wait         0.2 ms   14%  ████░░░░░░░░░░░░░░░░░░░░░░░░  ◀  cause|
|  └─ Traced Step Time   1.2 ms   86%  ████████████████████████░░░░          |
|     ├─ Compute         1.0 ms   75%                                        |
|     └─ Residual        0.1 ms   11%                                        |
|                                                                            |
|  H2D and step memory not measured (no GPU).                                |
|  DataLoader fetch: 0.2 ms (supplemental).                                  |
|                                                                            |
|  Next: raise DataLoader num_workers or prefetch, or check storage read     |
|  throughput.                                                               |
|                                                                            |
|  Full evidence: logs/session_20260806_090618/final_summary.json            |
|  (--html-report)                                                           |
+----------------------------------------------------------------------------+""",
    "run_not_enough_step_data": """\
+----------------------------------------------------------------------------+
|  TraceML Run Summary                                                       |
|  quickstart · 1 GPU · 12 steps analyzed · 4.0s                             |
+----------------------------------------------------------------------------+
|                                                                            |
|  Verdict: NOT ENOUGH STEP DATA                                             |
|  Why: only 12 completed steps were captured; a stable diagnosis needs a    |
|  larger sample.                                                            |
|                                                                            |
|  Observed anyway: GPU util 31% avg · CPU util 17% avg                      |
|                                                                            |
|  Next: run more steps, or check that traceml.trace_step(...) wraps the     |
|  training loop.                                                            |
|                                                                            |
|  Full evidence: logs/quickstart/final_summary.json  (--html-report)        |
+----------------------------------------------------------------------------+""",
    "run_step_timing_incomplete": """\
+----------------------------------------------------------------------------+
|  TraceML Run Summary                                                       |
|  quickstart · 1 GPU · 4.0s                                                 |
+----------------------------------------------------------------------------+
|                                                                            |
|  Verdict: STEP TIMING INCOMPLETE                                           |
|  Why: some timing signals were never measured (missing: backward,          |
|  optimizer), so phases don't add up to a reliable step time.               |
|                                                                            |
|  Next: check the integration wiring for the missing phases; per-signal     |
|  coverage is listed in the JSON step_time section.                         |
|                                                                            |
|  Full evidence: logs/quickstart/final_summary.json  (--html-report)        |
+----------------------------------------------------------------------------+""",
    "run_multi_input_straggler": """\
+----------------------------------------------------------------------------+
|  TraceML Run Summary                                                       |
|  ddp_pretrain · 4 GPUs · 2 nodes · 250 steps analyzed · 40.1s              |
+----------------------------------------------------------------------------+
|                                                                            |
|  Verdict: INPUT STRAGGLER  (CRITICAL)                                      |
|  Why: rank 0 (node n0) waits 254.5 ms per step for input; the median rank  |
|  waits 3.8 ms. All 4 ranks then advance at rank 0's pace.                  |
|                                                                            |
|  Where a step goes (median rank, GPU clock)         worst rank             |
|  Step Time           303.7 ms  100%                 304.1 ms (r0)          |
|  ├─ Input Wait         3.8 ms    1%                 254.5 ms (r0)  ◀  67x  |
|  └─ Traced Step Time 299.9 ms   99%                                        |
|     ├─ Compute       259.5 ms   85%                                        |
|     ├─ H2D             1.1 ms   <1%                                        |
|     └─ Residual       39.3 ms   13%                                        |
|                                                                            |
|  Next: inspect dataloader, collate_fn, preprocessing, and storage on rank 0|
|  (node n0).                                                                |
|                                                                            |
|  Supporting: GPU util median 14% -- ranks idle at the step barrier.        |
|  Peak step memory: 9.8 GB reserved · worst rank r2                         |
|                                                                            |
|  Full evidence: logs/ddp_pretrain/final_summary.json  (--html-report)      |
+----------------------------------------------------------------------------+""",
    "run_multi_healthy": """\
+----------------------------------------------------------------------------+
|  TraceML Run Summary                                                       |
|  ddp_pretrain · 4 GPUs · 1 node · 250 steps analyzed · 38.7s               |
+----------------------------------------------------------------------------+
|                                                                            |
|  Verdict: NO CLEAR BOTTLENECK                                              |
|  Why: step timing is balanced and ranks are even (worst step time +0.4% vs |
|  median).                                                                  |
|                                                                            |
|  Where a step goes (median rank, GPU clock)                                |
|  Step Time           152.1 ms  100%                                        |
|  ├─ Input Wait         2.0 ms    1%                                        |
|  └─ Traced Step Time 150.1 ms   99%                                        |
|     ├─ Compute       145.2 ms   95%                                        |
|     ├─ H2D             1.6 ms    1%                                        |
|     └─ Residual        3.3 ms    2%                                        |
|                                                                            |
|  System, process, and memory: all normal on every rank. GPU util 94% avg.  |
|  Peak step memory: 9.1 GB reserved · even across ranks                     |
|                                                                            |
|  Next: training is compute-dominated; for more speed profile kernels       |
|  (torch.profiler / Nsight).                                                |
|                                                                            |
|  Full evidence: logs/ddp_pretrain/final_summary.json  (--html-report)      |
+----------------------------------------------------------------------------+""",
    "run_single_gpu_on_shared_host": """\
+----------------------------------------------------------------------------+
|  TraceML Run Summary                                                       |
|  session_20260806_101725_5580d3 · 1 GPU · 128 steps analyzed · 12.1s       |
+----------------------------------------------------------------------------+
|                                                                            |
|  Verdict: LOW GPU UTILIZATION -- cause unclear                             |
|  Why: step timing is balanced, but GPU util averaged 0%. The idle time is  |
|  not explained by input or transfer.                                       |
|                                                                            |
|  Where a step goes (average, GPU clock)                                    |
|  Step Time            34.2 ms  100%                                        |
|  ├─ Input Wait         2.5 ms    7%                                        |
|  └─ Traced Step Time  31.7 ms   93%                                        |
|     ├─ Compute        29.4 ms   86%                                        |
|     └─ Residual        2.3 ms    7%                                        |
|                                                                            |
|  Next: look for untraced work between steps -- validation, checkpointing,  |
|  logging -- or inefficient kernels (torch.profiler).                       |
|                                                                            |
|  Peak step memory: 18.8 MB allocated · 23.1 MB reserved                    |
|                                                                            |
|  Full evidence: logs/session_20260806_101725_5580d3/final_summary.json     |
|  (--html-report)                                                           |
+----------------------------------------------------------------------------+""",
    "run_multi_residual_heavy": """\
+----------------------------------------------------------------------------+
|  TraceML Run Summary                                                       |
|  session_20260806_101837_1b81f7 · 4 GPUs · 1 node · 128 steps analyzed ·   |
|  2.7s                                                                      |
+----------------------------------------------------------------------------+
|                                                                            |
|  Verdict: RESIDUAL-HEAVY  (WARNING)                                        |
|  Why: 14% of the typical step is time outside the traced phases.           |
|                                                                            |
|  Where a step goes (median rank, GPU clock)                                |
|  Step Time             5.2 ms  100%                                        |
|  ├─ Input Wait         0.6 ms   11%                                        |
|  └─ Traced Step Time   4.6 ms   89%                                        |
|     ├─ Compute         3.8 ms   74%                                        |
|     ├─ H2D             0.1 ms    2%                                        |
|     └─ Residual        0.7 ms   14%  ◀  cause                              |
|                                                                            |
|  Next: look for untraced work between steps -- validation, checkpointing,  |
|  logging.                                                                  |
|                                                                            |
|  Supporting: GPU util 7% avg.                                              |
|  Peak step memory: 25.1 MB reserved · even across ranks                    |
|                                                                            |
|  Full evidence: logs/session_20260806_101837_1b81f7/final_summary.json     |
|  (--html-report)                                                           |
+----------------------------------------------------------------------------+""",
    "watch_healthy": """\
+----------------------------------------------------------------------------+
|  TraceML Watch Summary                                                     |
|  1 machine · 1 GPU · observed for 5m 12s                                   |
+----------------------------------------------------------------------------+
|                                                                            |
|  Host health: NORMAL                                                       |
|                                                                            |
|  CPU Util   18% avg          RAM       6.2 / 32.0 GB  (19%)                |
|  GPU Util   76% avg          GPU Mem  10.3 / 16.0 GB  (64%)                |
|  GPU Temp   61C max          Proc RSS  4.1 GB                              |
|                                                                            |
|  Watch monitors host and process health only; it does not measure step     |
|  time. To diagnose training speed: traceml run <your-script>.py            |
|                                                                            |
|  Full evidence: logs/watch_20260806/final_summary.json  (--html-report)    |
+----------------------------------------------------------------------------+""",
    "watch_low_gpu_utilization": """\
+----------------------------------------------------------------------------+
|  TraceML Watch Summary                                                     |
|  1 machine · 1 GPU · observed for 5m 12s                                   |
+----------------------------------------------------------------------------+
|                                                                            |
|  Host health: NORMAL                                                       |
|                                                                            |
|  CPU Util   18% avg          RAM       6.2 / 32.0 GB  (19%)                |
|  GPU Util   14% avg          GPU Mem   3.3 / 16.0 GB  (21%)                |
|  GPU Temp   42C max          Proc RSS  4.1 GB                              |
|                                                                            |
|  Observation: GPU utilization stayed low (14% avg). Watch cannot tell      |
|  whether that is input, transfer, sync, or idle time.                      |
|  Next: traceml run <your-script>.py -- measures step time and finds the    |
|  cause.                                                                    |
|                                                                            |
|  Full evidence: logs/watch_20260806/final_summary.json  (--html-report)    |
+----------------------------------------------------------------------------+""",
    "watch_memory_pressure": """\
+----------------------------------------------------------------------------+
|  TraceML Watch Summary                                                     |
|  1 machine · 1 GPU · observed for 5m 12s                                   |
+----------------------------------------------------------------------------+
|                                                                            |
|  Host health: MEMORY PRESSURE  (WARNING)                                   |
|  RAM 30.1 / 32.0 GB (94%) -- the host is close to memory exhaustion.       |
|                                                                            |
|  CPU Util   64% avg          RAM      30.1 / 32.0 GB  (94%)                |
|  GPU Util   81% avg          GPU Mem  14.9 / 16.0 GB  (93%)                |
|  GPU Temp   79C max          Proc RSS 28.6 GB                              |
|                                                                            |
|  Next: reduce DataLoader workers or caching, or move work off this host.   |
|                                                                            |
|  Watch monitors host and process health only; it does not measure step     |
|  time. To diagnose training speed: traceml run <your-script>.py            |
|                                                                            |
|  Full evidence: logs/watch_20260806/final_summary.json  (--html-report)    |
+----------------------------------------------------------------------------+""",
    "watch_multi_node": """\
+----------------------------------------------------------------------------+
|  TraceML Watch Summary                                                     |
|  3 nodes · 12 GPUs · observed for 12m 40s                                  |
+----------------------------------------------------------------------------+
|                                                                            |
|  Host health: NORMAL on all 3 nodes                                        |
|                                                                            |
|              median        worst node                                      |
|  CPU Util    24% avg       71%  (n1)                                       |
|  RAM         31% used      64%  (n1)                                       |
|  GPU Util    88% avg       62%  (n2)                                       |
|  GPU Mem     71% used      93%  (n0)                                       |
|  GPU Temp    64C max       81C  (n0)                                       |
|                                                                            |
|  Watch monitors host and process health only; it does not measure step     |
|  time. To diagnose training speed: traceml run <your-script>.py            |
|                                                                            |
|  Full evidence: logs/watch_20260806/final_summary.json  (--html-report)    |
+----------------------------------------------------------------------------+""",
}


def plain(name: str) -> str:
    """Render one named case as plain text."""
    return card_to_plain(CASES[name]())


@pytest.mark.parametrize("name", sorted(CASES))
def test_card_matches_golden(name: str) -> None:
    assert plain(name) == GOLDENS[name]


@pytest.mark.parametrize("name", sorted(CASES))
def test_card_lines_are_exactly_card_width(name: str) -> None:
    for line in plain(name).splitlines():
        assert len(line) == 78, (name, len(line), line)


def test_run_header_gpu_count_is_capped_by_world_size() -> None:
    # A one-process run on a four-GPU host describes the run, not the host.
    assert " · 1 GPU · " in plain("run_single_gpu_on_shared_host")
    # A four-rank run on the same host still reports every GPU it used.
    assert " · 4 GPUs · " in plain("run_multi_residual_heavy")


def test_watch_header_keeps_the_host_gpu_count() -> None:
    # watch describes the host, so it is not capped by world size.
    doc = _card(
        profile="watch",
        system=_watch_system(
            cpu_percent=1.0,
            ram_bytes=4.0e9,
            ram_percent=25.0,
            gpu_util_percent=0.2,
            gpu_mem_bytes=0.5e9,
            gpu_mem_percent=3.1,
            gpu_temp_c=34.0,
        ),
        process=_watch_process(2.0e8),
        step_time=_WATCH_STEP_TIME,
        step_memory=_WATCH_STEP_MEMORY,
        meta=_meta(world_size=1, nodes_observed=1, gpus_observed=4),
        duration_s=2.0,
        artifact_hint="logs/watch/final_summary.json",
    )
    assert "1 machine · 4 GPUs · observed for 2.0s" in card_to_plain(doc)


def test_also_block_skips_step_time_and_keeps_resource_findings() -> None:
    # A competing Step Time warning may well be a cause of slow steps, so it
    # never appears under "not the cause of slow steps"; a resource finding
    # from the same run still does.
    creep = _issue(
        "MEMORY_CREEP",
        "MEMORY CREEP",
        severity="warn",
        summary="Step memory grew 1.2 GB over the run -- possible leak",
        action="Check for retained tensors between steps.",
    )
    residual_heavy = _issue(
        "RESIDUAL_HEAVY",
        "RESIDUAL-HEAVY",
        severity="warn",
        summary="Residual time is 14.0% of the typical GPU Step Time.",
        action="Investigate untraced work between steps.",
        score=0.14,
    )
    input_bound = _issue(
        "INPUT_BOUND",
        "INPUT-BOUND",
        severity="warn",
        summary="Input wait is 11.0% of the typical GPU Step Time.",
        action="Increase workers, prefetch, or storage throughput.",
        score=0.11,
    )
    step_time = _step_time_single(
        diagnosis=residual_heavy,
        steps_analyzed=128,
        step_time_ms=5.2,
        input_wait_ms=0.6,
        traced_step_time_ms=4.6,
        compute_ms=3.8,
        h2d_ms=0.1,
        residual_ms=0.7,
    )
    step_time["issues"] = [residual_heavy, input_bound]

    text = card_to_plain(
        _card(
            profile="run",
            system=_system_single(gpu_util_percent=88.0),
            process=_NORMAL_PROCESS,
            step_time=step_time,
            step_memory=_step_memory_single(3.9e9, 4.6e9, diagnosis=creep),
            meta=_meta(run_name="ddp_balanced"),
            duration_s=2.7,
            artifact_hint="logs/ddp_balanced/final_summary.json",
        )
    )

    assert "Also, not the cause of slow steps:" in text
    assert "! Step memory grew 1.2 GB over the run" in text
    assert "Input wait is 11.0%" not in text


def test_peak_memory_falls_back_to_megabytes() -> None:
    # A sub-0.1 GB footprint must not render as "0.0 GB".
    text = plain("run_single_gpu_on_shared_host")
    assert "Peak step memory: 18.8 MB allocated · 23.1 MB reserved" in text
    assert "0.0 GB" not in text
    multi = plain("run_multi_residual_heavy")
    assert "Peak step memory: 25.1 MB reserved · even across ranks" in multi


def test_peak_memory_keeps_gigabytes_above_the_threshold() -> None:
    text = plain("run_healthy")
    assert "Peak step memory: 2.9 GB allocated · 3.2 GB reserved" in text


@pytest.mark.parametrize("name", sorted(CASES))
def test_card_never_prints_na(name: str) -> None:
    assert "n/a" not in plain(name)


@pytest.mark.parametrize("name", sorted(CASES))
def test_ansi_rendering_has_the_same_visible_text(name: str) -> None:
    doc = CASES[name]()
    assert _ANSI_RE.sub("", card_to_ansi(doc)) == card_to_plain(doc)


@pytest.mark.parametrize("name", sorted(CASES))
def test_ansi_rendering_uses_only_the_allowed_codes(name: str) -> None:
    codes = set(_ANSI_RE.findall(card_to_ansi(CASES[name]())))
    assert codes <= {
        "\x1b[0m",
        "\x1b[1m",
        "\x1b[2m",
        "\x1b[32m",
        "\x1b[1;31m",
        "\x1b[1;33m",
    }


@pytest.mark.parametrize("name", WATCH_CASES)
def test_watch_cards_never_mention_step_timing(name: str) -> None:
    text = plain(name)
    for banned in (
        "Step Time",
        "Step Memory",
        "INSUFFICIENT",
        "steps analyzed",
    ):
        assert banned not in text


@pytest.mark.parametrize("name", SINGLE_MACHINE_CASES)
def test_single_machine_cards_have_no_distributed_vocabulary(
    name: str,
) -> None:
    text = plain(name)
    for banned in ("rank", "node", "skew"):
        assert not re.search(rf"\b{banned}", text, re.IGNORECASE), banned
