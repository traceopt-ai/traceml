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

import copy
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from traceml_ai.reporting.primary_diagnosis import build_primary_diagnosis
from traceml_ai.reporting.terminal_card.common import format_scope
from traceml_ai.reporting.terminal_card.card import (
    CardDoc,
    Span,
    STYLE_BOLD,
    STYLE_CRIT,
    STYLE_DIM,
    STYLE_NEXT,
    STYLE_OK,
    STYLE_PLAIN,
    STYLE_WARN,
    build_summary_card,
    card_to_ansi,
    card_to_plain,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"rank": 2, "node": 1, "gpu": 0}, "R2/N1/G0"),
        ({"rank": 2, "node": 1}, "R2/N1"),
        ({"node": 1, "gpu": 0}, "N1/G0"),
        ({"node": 1}, "N1"),
        ({"gpu": 0}, "G0"),
        ({}, None),
    ],
)
def test_format_scope_uses_only_present_stored_identity_parts(
    kwargs: Dict[str, int], expected: Optional[str]
) -> None:
    """Compact scope labels never infer missing rank, node, or GPU fields."""
    assert format_scope(**kwargs) == expected


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


def _rank_rows(
    *ranks: int,
    node_rank: int = 0,
    node_ranks: Optional[Dict[int, int]] = None,
    metrics_by_rank: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build grouped rows indexed by global rank, with optional metrics."""
    node_ranks = node_ranks or {}
    metrics_by_rank = metrics_by_rank or {}
    return {
        str(rank): {
            "identity": {
                "global_rank": rank,
                "local_rank": rank,
                "node_rank": node_ranks.get(rank, node_rank),
                "hostname": None,
                "local_world_size": None,
                "world_size": None,
            },
            "metrics": dict(metrics_by_rank.get(rank, {})),
        }
        for rank in ranks
    }


def _node_rows(
    *nodes: int,
    metrics_by_node: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build grouped rows indexed by node rank, with optional metrics."""
    metrics_by_node = metrics_by_node or {}
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
            "metrics": dict(metrics_by_node.get(node, {})),
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


_NORMAL_PROCESS = _section(
    diagnosis=_issue("NORMAL", "NORMAL"),
    metadata={"global_ranks_used": 1},
    average={
        "cpu_capacity_percent": 14.0,
        "ram_bytes": 3.1e9,
        "ram_percent": 10.0,
        "gpu_mem_used_bytes": 2.9e9,
        "gpu_mem_reserved_bytes": 3.2e9,
        "gpu_mem_reserved_percent": 20.0,
    },
    rows=_rank_rows(
        0,
        metrics_by_rank={
            0: {
                "cpu_capacity_percent": 14.0,
                "ram_bytes": 3.1e9,
                "ram_percent": 10.0,
                "gpu_mem_used_bytes": 2.9e9,
                "gpu_mem_reserved_bytes": 3.2e9,
                "gpu_mem_reserved_percent": 20.0,
            }
        },
    ),
)


_CPU_ONLY_PROCESS = _section(
    diagnosis=_issue("NORMAL", "NORMAL"),
    metadata={"global_ranks_used": 1},
    average={
        "cpu_capacity_percent": 14.0,
        "ram_bytes": 3.1e9,
        "ram_percent": 10.0,
    },
    rows=_rank_rows(
        0,
        metrics_by_rank={
            0: {
                "cpu_capacity_percent": 14.0,
                "ram_bytes": 3.1e9,
                "ram_percent": 10.0,
            }
        },
    ),
)


def _multi_process_fixture(*, node_ranks: Dict[int, int]) -> Dict[str, Any]:
    """Build four-rank Process data with topology-consistent identities."""
    return _section(
        diagnosis=_issue("NORMAL", "NORMAL"),
        metadata={"global_ranks_seen": 4, "global_ranks_used": 4},
        average={
            "cpu_capacity_percent": 32.0,
            "ram_bytes": 4.2e9,
            "ram_percent": 13.5,
            "gpu_mem_used_bytes": 3.6e9,
            "gpu_mem_reserved_bytes": 4.5e9,
            "gpu_mem_reserved_percent": 28.25,
        },
        median={
            "cpu_capacity_percent": _point(12.0, "0"),
            # Crossed byte indexes prove percentage anchors select one row.
            "ram_bytes": _point(999.0e9, "2"),
            "ram_percent": _point(10.0, "0"),
            "gpu_mem_used_bytes": _point(2.9e9, "0"),
            "gpu_mem_reserved_bytes": _point(777.0e9, "2"),
            "gpu_mem_reserved_percent": _point(20.0, "0"),
        },
        worst={
            "cpu_capacity_percent": _point(81.0, "2"),
            "ram_bytes": _point(888.0e9, "3"),
            "ram_percent": _point(17.0, "1"),
            "gpu_mem_used_bytes": _point(4.6e9, "3"),
            "gpu_mem_reserved_bytes": _point(666.0e9, "1"),
            "gpu_mem_reserved_percent": _point(43.0, "3"),
        },
        rows=_rank_rows(
            0,
            1,
            2,
            3,
            node_ranks=node_ranks,
            metrics_by_rank={
                0: {
                    "cpu_capacity_percent": 12.0,
                    "ram_bytes": 3.1e9,
                    "ram_percent": 10.0,
                    "gpu_mem_used_bytes": 2.9e9,
                    "gpu_mem_reserved_bytes": 3.2e9,
                    "gpu_mem_reserved_percent": 20.0,
                },
                1: {
                    "cpu_capacity_percent": 20.0,
                    "ram_bytes": 5.4e9,
                    "ram_percent": 17.0,
                    "gpu_mem_used_bytes": 3.2e9,
                    "gpu_mem_reserved_bytes": 3.8e9,
                    "gpu_mem_reserved_percent": 24.0,
                },
                2: {
                    "cpu_capacity_percent": 81.0,
                    "ram_bytes": 4.2e9,
                    "ram_percent": 14.0,
                    "gpu_mem_used_bytes": 3.6e9,
                    "gpu_mem_reserved_bytes": 4.2e9,
                    "gpu_mem_reserved_percent": 26.0,
                },
                3: {
                    "cpu_capacity_percent": 15.0,
                    "ram_bytes": 4.0e9,
                    "ram_percent": 13.0,
                    "gpu_mem_used_bytes": 4.6e9,
                    "gpu_mem_reserved_bytes": 6.8e9,
                    "gpu_mem_reserved_percent": 43.0,
                },
            },
        ),
    )


_ONE_NODE_MULTI_PROCESS = _multi_process_fixture(
    node_ranks={0: 0, 1: 0, 2: 0, 3: 0}
)
_TWO_NODE_MULTI_PROCESS = _multi_process_fixture(
    node_ranks={0: 0, 1: 0, 2: 1, 3: 1}
)


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
    gpu_power_w: Optional[float] = None,
    gpus_observed: Optional[int] = 1,
) -> Dict[str, Any]:
    """Build a single-node System section payload."""
    return _section(
        diagnosis=diagnosis or _issue("NORMAL", "NORMAL"),
        metadata={
            "mode": "single_node",
            "nodes_observed": 1,
            "nodes_expected": 1,
            "gpus_observed": gpus_observed,
        },
        average={
            "cpu_percent": cpu_percent,
            "ram_bytes": ram_bytes,
            "ram_percent": ram_percent,
            "gpu_util_percent": gpu_util_percent,
            "gpu_mem_bytes": gpu_mem_bytes,
            "gpu_mem_percent": gpu_mem_percent,
            "gpu_temp_c": gpu_temp_c,
            "gpu_power_w": gpu_power_w,
        },
        worst={"gpu_temp_c": _point(gpu_temp_c, "0")},
        rows=_node_rows(
            0,
            metrics_by_node={
                0: {
                    "cpu_percent": cpu_percent,
                    "ram_bytes": ram_bytes,
                    "ram_percent": ram_percent,
                    "gpu_util_percent": gpu_util_percent,
                    "gpu_mem_bytes": gpu_mem_bytes,
                    "gpu_mem_percent": gpu_mem_percent,
                    "gpu_temp_c": gpu_temp_c,
                    "gpu_power_w": gpu_power_w,
                }
            },
        ),
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
        average={
            "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved,
        },
        worst={
            "peak_allocated_bytes": _point(allocated, "0"),
            "peak_reserved_bytes": _point(reserved, "0"),
        },
        median={
            "peak_allocated_bytes": _point(allocated, "0"),
            "peak_reserved_bytes": _point(reserved, "0"),
        },
        rows=_rank_rows(
            0,
            metrics_by_rank={
                0: {
                    "peak_allocated_bytes": allocated,
                    "peak_reserved_bytes": reserved,
                }
            },
        ),
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
    forward_ms: Optional[float] = None,
    backward_ms: Optional[float] = None,
    optimizer_ms: Optional[float] = None,
    alignment: Optional[str] = "common_steps",
) -> Dict[str, Any]:
    """Build a single-rank Step Time section payload."""
    return _section(
        diagnosis=diagnosis,
        metadata={"mode": "single_node", "global_ranks_used": 1},
        window={
            "steps_analyzed": steps_analyzed,
            "diagnosis_clock": clock,
            "alignment": alignment,
        },
        average={
            "step_time_ms": step_time_ms,
            "input_wait_ms": input_wait_ms,
            "traced_step_time_ms": traced_step_time_ms,
            "compute_ms": compute_ms,
            "h2d_ms": h2d_ms,
            "residual_ms": residual_ms,
            "dataloader_fetch_cpu_ms": dataloader_fetch_cpu_ms,
            "forward_ms": forward_ms,
            "backward_ms": backward_ms,
            "optimizer_ms": optimizer_ms,
        },
        rows=_rank_rows(
            0,
            metrics_by_rank={
                0: {
                    "step_time_ms": step_time_ms,
                    "input_wait_ms": input_wait_ms,
                    "traced_step_time_ms": traced_step_time_ms,
                    "compute_ms": compute_ms,
                    "h2d_ms": h2d_ms,
                    "residual_ms": residual_ms,
                    "dataloader_fetch_cpu_ms": dataloader_fetch_cpu_ms,
                    "forward_ms": forward_ms,
                    "backward_ms": backward_ms,
                    "optimizer_ms": optimizer_ms,
                }
            },
        ),
    )


def _system_multi_node(
    *,
    diagnosis: Dict[str, Any],
    average: Dict[str, Any],
    median: Dict[str, Any],
    worst: Dict[str, Any],
    metrics_by_node: Dict[int, Dict[str, Any]],
    gpus_observed: int = 4,
    nodes_expected: Optional[int] = None,
) -> Dict[str, Any]:
    """Build multi-node System data from explicit stored node values."""
    nodes = tuple(metrics_by_node)
    return _section(
        diagnosis=diagnosis,
        metadata={
            "mode": "multi_node",
            "nodes_observed": len(nodes),
            "nodes_expected": (
                len(nodes) if nodes_expected is None else nodes_expected
            ),
            "gpus_observed": gpus_observed,
        },
        average=average,
        median=median,
        worst=worst,
        rows=_node_rows(*nodes, metrics_by_node=metrics_by_node),
        by="node_rank",
    )


def _step_time_multi_rank(
    *,
    diagnosis: Dict[str, Any],
    steps_analyzed: Optional[int],
    median: Dict[str, Any],
    worst: Dict[str, Any],
    metrics_by_rank: Dict[int, Dict[str, Any]],
    node_ranks: Optional[Dict[int, int]] = None,
    ranks: Tuple[int, ...] = (0, 1, 2, 3),
    mode: str = "multi_node",
    issues: Optional[List[Dict[str, Any]]] = None,
    clock: str = "gpu",
    alignment: str = "common_steps",
) -> Dict[str, Any]:
    """Build multi-rank Step Time data with explicit stored points and rows."""
    return _section(
        diagnosis=diagnosis,
        issues=issues,
        metadata={"mode": mode, "global_ranks_used": len(ranks)},
        window={
            "steps_analyzed": steps_analyzed,
            "diagnosis_clock": clock,
            "alignment": alignment,
        },
        median=median,
        worst=worst,
        rows=_rank_rows(
            *ranks,
            node_ranks=node_ranks,
            metrics_by_rank=metrics_by_rank,
        ),
    )


def _step_memory_multi_rank(
    *,
    diagnosis: Dict[str, Any],
    median: Dict[str, Any],
    worst: Dict[str, Any],
    metrics_by_rank: Dict[int, Dict[str, Any]],
    node_ranks: Optional[Dict[int, int]] = None,
    ranks: Tuple[int, ...] = (0, 1, 2, 3),
) -> Dict[str, Any]:
    """Build multi-rank Step Memory data with explicit points and rank rows."""
    return _section(
        diagnosis=diagnosis,
        metadata={"global_ranks_used": len(ranks)},
        median=median,
        worst=worst,
        rows=_rank_rows(
            *ranks,
            node_ranks=node_ranks,
            metrics_by_rank=metrics_by_rank,
        ),
    )


def _render_single_run(
    *,
    run_name: str,
    step_time: Dict[str, Any],
    system: Optional[Dict[str, Any]] = None,
    process: Optional[Dict[str, Any]] = None,
    step_memory: Optional[Dict[str, Any]] = None,
    duration_s: float = 1.0,
) -> str:
    """Render a behavior-only single-process Run with neutral defaults."""
    return card_to_plain(
        _card(
            profile="run",
            system=system if system is not None else _system_single(),
            process=process if process is not None else _NORMAL_PROCESS,
            step_time=step_time,
            step_memory=(
                step_memory
                if step_memory is not None
                else _step_memory_single(None, None)
            ),
            meta=_meta(run_name=run_name),
            duration_s=duration_s,
            artifact_hint=f"logs/{run_name}/final_summary.json",
        )
    )


def run_input_bound_critical() -> CardDoc:
    """run x single GPU, INPUT-BOUND critical (the flagship card)."""
    return _card(
        profile="run",
        system=_system_single(
            diagnosis=_issue(
                "LOW_GPU_UTILIZATION",
                "LOW GPU UTIL",
                summary="GPU utilization averaged 24%.",
            ),
            cpu_percent=18.4,
            gpu_util_percent=24.0,
            ram_bytes=6.2e9,
            ram_percent=19.0,
            gpu_mem_bytes=3.33e9,
            gpu_mem_percent=21.0,
            gpu_temp_c=42.0,
            gpu_power_w=58.0,
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
                share_pct=0.64,
            ),
            steps_analyzed=256,
            step_time_ms=200.4,
            input_wait_ms=128.0,
            traced_step_time_ms=72.0,
            compute_ms=68.0,
            h2d_ms=0.4,
            residual_ms=3.6,
            dataloader_fetch_cpu_ms=120.0,
            forward_ms=24.0,
            backward_ms=38.0,
            optimizer_ms=6.0,
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
            ram_bytes=6.2e9,
            ram_percent=19.0,
            gpu_mem_bytes=2.8e9,
            gpu_mem_percent=17.0,
            gpu_temp_c=61.0,
            gpu_power_w=210.0,
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
            dataloader_fetch_cpu_ms=1.8,
            forward_ms=65.0,
            backward_ms=110.0,
            optimizer_ms=10.0,
        ),
        step_memory=_step_memory_single(2.9e9, 3.2e9),
        meta=_meta(run_name="bert_finetune"),
        duration_s=48.9,
        artifact_hint="logs/bert_finetune/final_summary.json",
    )


def run_input_bound_with_also() -> CardDoc:
    """run x single GPU, INPUT-BOUND critical plus a step-memory warning."""
    creep = _issue(
        "CREEP_CONFIRMED",
        "MEMORY CREEP",
        severity="warn",
        summary="Peak reserved memory is rising across the window.",
        action="Check for retained tensors between steps.",
        metric="peak_reserved",
        ranks=[0],
        evidence={
            "overall_abs_delta_bytes": 1.2e9,
            "overall_worst_growth_pct": 0.18,
        },
    )
    fragmentation = _issue(
        "ALLOCATOR_FRAGMENTATION",
        "ALLOCATOR FRAGMENTATION",
        severity="warn",
        summary="Allocator reserved memory is 2.4x live tensor memory.",
        action="Inspect retained blocks and allocation sizes.",
    )
    step_memory = _step_memory_single(3.9e9, 4.6e9, diagnosis=creep)
    step_memory["issues"] = [creep, fragmentation]
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
                share_pct=0.64,
            ),
            steps_analyzed=256,
            step_time_ms=200.4,
            input_wait_ms=128.0,
            traced_step_time_ms=72.0,
            compute_ms=68.0,
            h2d_ms=0.4,
            residual_ms=3.6,
        ),
        step_memory=step_memory,
        meta=_meta(run_name="bert_finetune"),
        duration_s=52.4,
        artifact_hint="logs/bert_finetune/final_summary.json",
    )


def run_low_gpu_utilization() -> CardDoc:
    """run x single GPU, balanced timing with unexplained low GPU util."""
    return _card(
        profile="run",
        system=_system_single(
            diagnosis=_issue(
                "LOW_GPU_UTILIZATION",
                "LOW GPU UTIL",
                summary="GPU utilization averaged 22%.",
            ),
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
        system=_system_single(cpu_percent=25.4, gpus_observed=0),
        process=_CPU_ONLY_PROCESS,
        step_time=_step_time_single(
            diagnosis=_issue(
                "INPUT_BOUND",
                "INPUT-BOUND",
                severity="warn",
                summary="Input Wait is 14.0% of the typical CPU Step Time.",
                action="Increase workers or prefetch.",
                score=0.14,
                share_pct=0.14,
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
    node_ranks = {0: 0, 1: 0, 2: 1, 3: 1}
    system = _system_multi_node(
        diagnosis=_issue(
            "LOW_GPU_UTILIZATION",
            "LOW GPU UTIL",
            summary="GPU utilization averaged 14%.",
        ),
        average={
            "cpu_percent": 22.0,
            "ram_bytes": 18.4e9,
            "ram_percent": 31.0,
            "gpu_util_percent": 14.0,
            "gpu_mem_bytes": 6.0e9,
            "gpu_mem_percent": 37.5,
            "gpu_temp_c": 64.0,
            "gpu_power_w": 250.0,
        },
        median={
            "cpu_percent": _point(18.0, "0"),
            "ram_bytes": _point(16.0e9, "0"),
            "ram_percent": _point(27.0, "0"),
            "gpu_util_percent": _point(9.0, "1"),
            "gpu_mem_bytes": _point(5.0e9, "0"),
            "gpu_mem_percent": _point(31.0, "0"),
            "gpu_temp_c": _point(58.0, "0"),
            "gpu_power_w": _point(220.0, "0"),
        },
        worst={
            "cpu_percent": _point(26.0, "1"),
            "ram_bytes": _point(20.8e9, "1"),
            "ram_percent": _point(35.0, "1"),
            "gpu_util_percent": _point(9.0, "1"),
            "gpu_mem_bytes": _point(7.0e9, "1"),
            "gpu_mem_percent": _point(44.0, "1"),
            "gpu_temp_c": _point(70.0, "1"),
            "gpu_power_w": _point(280.0, "1"),
        },
        metrics_by_node={
            0: {
                "cpu_percent": 18.0,
                "ram_bytes": 16.0e9,
                "ram_percent": 27.0,
                "gpu_util_percent": 19.0,
                "gpu_mem_bytes": 5.0e9,
                "gpu_mem_percent": 31.0,
                "gpu_temp_c": 58.0,
                "gpu_power_w": 220.0,
            },
            1: {
                "cpu_percent": 26.0,
                "ram_bytes": 20.8e9,
                "ram_percent": 35.0,
                "gpu_util_percent": 9.0,
                "gpu_mem_bytes": 7.0e9,
                "gpu_mem_percent": 44.0,
                "gpu_temp_c": 70.0,
                "gpu_power_w": 280.0,
            },
        },
    )
    step_time = _step_time_multi_rank(
        diagnosis=_issue(
            "INPUT_STRAGGLER",
            "INPUT STRAGGLER",
            severity="crit",
            summary=(
                "r0 waited 254.5 ms for input, compared with 3.8 ms on r1."
            ),
            action="Inspect input wait on the slow rank.",
            phase="input",
            score=0.83,
            evidence={
                "culprit_rank": 0,
                "victim_rank": 1,
                "visible_metric": "backward",
                "visible_culprit_ms": 20.0,
                "visible_victim_ms": 120.0,
                "visible_cost_ms": 100.0,
            },
        ),
        steps_analyzed=250,
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
        node_ranks=node_ranks,
        metrics_by_rank={
            0: {
                "step_time_ms": 304.1,
                "input_wait_ms": 254.5,
                "traced_step_time_ms": 300.5,
                "compute_ms": 261.0,
                "h2d_ms": 1.2,
                "residual_ms": 39.5,
                "dataloader_fetch_cpu_ms": 249.0,
                "forward_ms": 81.0,
                "backward_ms": 170.0,
                "optimizer_ms": 10.0,
            },
            1: {
                "step_time_ms": 303.7,
                "input_wait_ms": 3.8,
                "traced_step_time_ms": 299.9,
                "compute_ms": 259.5,
                "h2d_ms": 1.1,
                "residual_ms": 39.3,
                "dataloader_fetch_cpu_ms": 3.7,
                "forward_ms": 80.0,
                "backward_ms": 169.5,
                "optimizer_ms": 10.0,
            },
        },
    )
    step_memory = _step_memory_multi_rank(
        diagnosis=_issue("BALANCED", "BALANCED"),
        median={
            "peak_allocated_bytes": _point(8.5e9, "1"),
            "peak_reserved_bytes": _point(8.9e9, "1"),
        },
        worst={
            "peak_allocated_bytes": _point(9.4e9, "2"),
            "peak_reserved_bytes": _point(9.8e9, "2"),
        },
        node_ranks=node_ranks,
        metrics_by_rank={
            1: {
                "peak_allocated_bytes": 8.5e9,
                "peak_reserved_bytes": 8.9e9,
            },
            2: {
                "peak_allocated_bytes": 9.4e9,
                "peak_reserved_bytes": 9.8e9,
            },
        },
    )
    return _card(
        profile="run",
        system=system,
        process=_TWO_NODE_MULTI_PROCESS,
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
    system = _system_single(
        cpu_percent=38.0,
        ram_bytes=18.4e9,
        ram_percent=31.0,
        gpu_util_percent=94.0,
        gpu_mem_bytes=6.0e9,
        gpu_mem_percent=38.0,
        gpu_temp_c=64.0,
        gpu_power_w=240.0,
        gpus_observed=4,
    )
    step_time = _step_time_multi_rank(
        diagnosis=_issue("BALANCED", "BALANCED"),
        steps_analyzed=250,
        mode="single_node",
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
        metrics_by_rank={
            1: {
                "step_time_ms": 152.1,
                "input_wait_ms": 2.0,
                "traced_step_time_ms": 150.1,
                "compute_ms": 145.2,
                "h2d_ms": 1.6,
                "residual_ms": 3.3,
                "dataloader_fetch_cpu_ms": 1.8,
                "forward_ms": 50.0,
                "backward_ms": 85.2,
                "optimizer_ms": 10.0,
            },
            2: {
                "step_time_ms": 152.7,
                "input_wait_ms": 2.1,
                "traced_step_time_ms": 150.6,
                "compute_ms": 145.6,
                "h2d_ms": 1.7,
                "residual_ms": 3.4,
            },
        },
    )
    step_memory = _step_memory_multi_rank(
        diagnosis=_issue("BALANCED", "BALANCED"),
        median={
            "peak_allocated_bytes": _point(8.7e9, "1"),
            "peak_reserved_bytes": _point(9.0e9, "1"),
        },
        worst={
            "peak_allocated_bytes": _point(8.8e9, "2"),
            "peak_reserved_bytes": _point(9.1e9, "2"),
        },
        metrics_by_rank={
            1: {
                "peak_allocated_bytes": 8.7e9,
                "peak_reserved_bytes": 9.0e9,
            },
            2: {
                "peak_allocated_bytes": 8.8e9,
                "peak_reserved_bytes": 9.1e9,
            },
        },
    )
    return _card(
        profile="run",
        system=system,
        process=_ONE_NODE_MULTI_PROCESS,
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
            diagnosis=_issue(
                "LOW_GPU_UTILIZATION",
                "LOW GPU UTIL",
                summary="GPU utilization averaged 0.2%.",
            ),
            cpu_percent=0.2285714285714286,
            gpu_util_percent=0.21428571428571427,
            ram_bytes=3986740955.428571,
            ram_percent=1.988849429010849,
            gpu_mem_bytes=498930249.14285713,
            gpu_mem_percent=3.0977666945684526,
            gpu_temp_c=34.5,
            gpus_observed=4,
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
    system = _system_single(
        diagnosis=_issue(
            "LOW_GPU_UTILIZATION",
            "LOW GPU UTIL",
            summary="GPU utilization averaged 6.8%.",
        ),
        cpu_percent=2.4,
        gpu_util_percent=6.833333333333333,
        gpus_observed=4,
    )
    residual_heavy = _issue(
        "RESIDUAL_HEAVY",
        "RESIDUAL-HEAVY",
        severity="warn",
        summary="Residual time is 14.0% of the typical GPU Step Time.",
        action="Investigate untraced work between steps.",
        score=0.13980705864843995,
        share_pct=0.13980705864843995,
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
    step_time = _step_time_multi_rank(
        diagnosis=residual_heavy,
        issues=[residual_heavy, input_bound],
        steps_analyzed=128,
        mode="single_node",
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
        metrics_by_rank={
            0: {
                "step_time_ms": 5.201,
                "input_wait_ms": 0.573,
                "traced_step_time_ms": 4.657,
                "compute_ms": 3.84,
                "h2d_ms": 0.08,
                "residual_ms": 0.737,
            },
            1: {
                "step_time_ms": 5.203032052493654,
                "input_wait_ms": 0.575069501879625,
                "traced_step_time_ms": 4.6259870529174805,
                "compute_ms": 3.83626828956767,
                "h2d_ms": 0.08052700001280755,
                "residual_ms": 0.7217999144340865,
                "dataloader_fetch_cpu_ms": 0.57,
                "forward_ms": 1.3,
                "backward_ms": 2.2,
                "optimizer_ms": 0.33626828956767,
            },
            2: {
                "step_time_ms": 5.229208162869327,
                "input_wait_ms": 0.579,
                "traced_step_time_ms": 4.65,
                "compute_ms": 3.86,
                "h2d_ms": 0.081,
                "residual_ms": 0.74,
            },
            3: {
                "step_time_ms": 5.21,
                "input_wait_ms": 0.576,
                "traced_step_time_ms": 4.63,
                "compute_ms": 3.8683359728602227,
                "h2d_ms": 0.081,
                "residual_ms": 0.73,
            },
        },
    )
    step_memory = _step_memory_multi_rank(
        diagnosis=_issue("BALANCED", "BALANCED"),
        median={
            "peak_allocated_bytes": _point(18838528.0, "1"),
            "peak_reserved_bytes": _point(23068672.0, "1"),
        },
        worst={
            "peak_allocated_bytes": _point(20971520.0, "0"),
            "peak_reserved_bytes": _point(25149440.0, "0"),
        },
        metrics_by_rank={
            0: {
                "peak_allocated_bytes": 20971520.0,
                "peak_reserved_bytes": 25149440.0,
            },
            1: {
                "peak_allocated_bytes": 18838528.0,
                "peak_reserved_bytes": 23068672.0,
            },
        },
    )
    return _card(
        profile="run",
        system=system,
        process=_ONE_NODE_MULTI_PROCESS,
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


def run_partial_rank_payload() -> CardDoc:
    """run x partial distributed payload with missing coherent rows."""
    inputs = _coherent_multi_inputs()
    inputs["step_time"]["global"]["median"]["step_time_ms"] = _point(
        100.0,
        "missing",
    )
    inputs["step_memory"]["global"]["median"]["peak_reserved_bytes"] = _point(
        4.0e9, "missing"
    )
    return _card(**inputs)


def run_multi_duplicate_memory_pair() -> CardDoc:
    """run x DDP where median and worst memory select the same row."""
    inputs = _coherent_multi_inputs()
    step_memory = inputs["step_memory"]
    step_memory["global"]["worst"]["peak_allocated_bytes"] = _point(
        3.0e9,
        "3",
    )
    step_memory["global"]["worst"]["peak_reserved_bytes"] = _point(
        4.0e9,
        "3",
    )
    return _card(**inputs)


def run_multi_high_power_partial() -> CardDoc:
    """run x partial multi-node System coverage with high GPU power."""
    inputs = _coherent_multi_inputs()
    diagnosis = _issue(
        "HIGH_GPU_POWER",
        "HIGH GPU PWR",
        severity="warn",
        summary="GPU power was high, averaging 83.7% of limit.",
        evidence={
            "gpu_power_avg_limit_percent": 83.7,
            "gpu_idx": 0,
            "scope": {"level": "gpu", "node_rank": 0, "gpu_idx": 0},
        },
    )
    system = inputs["system"]
    system["diagnosis"] = diagnosis
    system["issues"] = [diagnosis]
    system["metadata"].update(
        {
            "nodes_expected": 4,
            "nodes_coverage": "2/4",
            "nodes_partial": True,
        }
    )
    inputs["meta"]["run_name"] = "multi-high-power"
    inputs["artifact_hint"] = "logs/multi-high-power/final_summary.json"
    return _card(**inputs)


def run_measured_zero_and_null() -> CardDoc:
    """run x measured zero values with adjacent unavailable metrics."""
    system = _system_single(
        cpu_percent=0.0,
        ram_bytes=0.0,
        ram_percent=0.0,
        gpu_util_percent=0.0,
        gpu_mem_bytes=None,
        gpu_mem_percent=None,
        gpu_temp_c=0.0,
    )
    process = _section(
        diagnosis=_issue("NORMAL", "NORMAL"),
        metadata={"global_ranks_used": 1},
        average={
            "cpu_capacity_percent": 0.0,
            "ram_bytes": 0.0,
            "ram_percent": 0.0,
            "gpu_mem_used_bytes": None,
            "gpu_mem_reserved_bytes": None,
            "gpu_mem_reserved_percent": None,
        },
        rows=_rank_rows(0),
    )
    return _card(
        profile="run",
        system=system,
        process=process,
        step_time=_step_time_single(
            diagnosis=_issue("BALANCED", "BALANCED"),
            steps_analyzed=20,
            step_time_ms=10.0,
            input_wait_ms=0.0,
            traced_step_time_ms=10.0,
            compute_ms=10.0,
            h2d_ms=0.0,
            residual_ms=0.0,
            dataloader_fetch_cpu_ms=0.0,
            forward_ms=0.0,
            backward_ms=10.0,
            optimizer_ms=0.0,
        ),
        step_memory=_step_memory_single(0.0, 0.0),
        meta=_meta(run_name="measured-zero"),
        duration_s=1.0,
        artifact_hint="logs/measured-zero/final_summary.json",
    )


def run_unmeasured_resource_sections() -> CardDoc:
    """run x complete timing with unavailable resource sections."""
    system_diagnosis = _issue(
        "NO_DATA",
        "NO DATA",
        summary="System sampler produced no rows.",
    )
    process_diagnosis = _issue(
        "NO_DATA",
        "NO DATA",
        summary="Process telemetry was not measured.",
    )
    memory_diagnosis = _issue(
        "NO_GPU",
        "NO GPU",
        summary="Step memory was not measured because no GPU was detected.",
    )
    return _card(
        profile="run",
        system=_section(
            diagnosis=system_diagnosis,
            metadata={"gpus_observed": 0, "nodes_observed": 1},
            by="node_rank",
        ),
        process=_section(diagnosis=process_diagnosis),
        step_time=_step_time_single(
            diagnosis=_issue("BALANCED", "BALANCED"),
            steps_analyzed=20,
            clock="cpu",
            step_time_ms=10.0,
            input_wait_ms=1.0,
            traced_step_time_ms=9.0,
            compute_ms=8.0,
            residual_ms=1.0,
        ),
        step_memory=_section(diagnosis=memory_diagnosis),
        meta=_meta(run_name="unmeasured", gpus_observed=0),
        duration_s=1.0,
        artifact_hint="logs/unmeasured/final_summary.json",
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


def run_h2d_bound() -> CardDoc:
    """run x single GPU, H2D-BOUND warning."""
    return _card(
        profile="run",
        system=_system_single(gpu_util_percent=72.0),
        process=_NORMAL_PROCESS,
        step_time=_step_time_single(
            diagnosis=_issue(
                "H2D_BOUND",
                "H2D-BOUND",
                severity="warn",
                summary="Stored H2D summary.",
                action="Inspect pinned memory and batch transfers.",
                phase="h2d",
                share_pct=0.20,
            ),
            steps_analyzed=40,
            step_time_ms=100.0,
            input_wait_ms=5.0,
            traced_step_time_ms=95.0,
            compute_ms=70.0,
            h2d_ms=20.0,
            residual_ms=5.0,
            forward_ms=20.0,
            backward_ms=45.0,
            optimizer_ms=5.0,
        ),
        step_memory=_step_memory_single(2.0e9, 2.4e9),
        meta=_meta(run_name="h2d-bound"),
        duration_s=5.0,
        artifact_hint="logs/h2d-bound/final_summary.json",
    )


def run_compute_bound() -> CardDoc:
    """run x single GPU, COMPUTE-BOUND with a stored largest phase."""
    return _card(
        profile="run",
        system=_system_single(gpu_util_percent=95.0),
        process=_NORMAL_PROCESS,
        step_time=_step_time_single(
            diagnosis=_issue(
                "COMPUTE_BOUND",
                "COMPUTE-BOUND",
                summary="Stored compute summary.",
                action="Optimize model compute or reduce step cost.",
                phase="backward",
                share_pct=0.95,
            ),
            steps_analyzed=40,
            step_time_ms=100.0,
            input_wait_ms=1.0,
            traced_step_time_ms=99.0,
            compute_ms=95.0,
            h2d_ms=1.0,
            residual_ms=3.0,
            forward_ms=20.0,
            backward_ms=65.0,
            optimizer_ms=10.0,
        ),
        step_memory=_step_memory_single(2.0e9, 2.4e9),
        meta=_meta(run_name="compute-bound"),
        duration_s=5.0,
        artifact_hint="logs/compute-bound/final_summary.json",
    )


def _run_straggler_case(
    *,
    kind: str,
    status: str,
    phase: str,
    evidence: Dict[str, Any],
    culprit_metrics: Dict[str, Any],
    victim_metrics: Dict[str, Any],
    run_name: str,
) -> CardDoc:
    """Build a distributed Run case with diagnosis-specific rank rows."""
    inputs = _coherent_multi_inputs()
    diagnosis = _issue(
        kind,
        status,
        severity="warn",
        summary=f"Stored {status.lower()} summary.",
        action="Inspect the diagnosed ranks.",
        phase=phase,
        score=0.5,
        evidence=evidence,
    )
    step_time = inputs["step_time"]
    step_time["diagnosis"] = diagnosis
    step_time["issues"] = [diagnosis]
    rows = step_time["groups"]["rows"]
    rows["0"]["metrics"].update(culprit_metrics)
    rows["1"]["metrics"].update(victim_metrics)
    inputs["meta"]["run_name"] = run_name
    inputs["artifact_hint"] = f"logs/{run_name}/final_summary.json"
    return _card(**inputs)


def run_multi_h2d_straggler() -> CardDoc:
    """run x distributed, H2D straggler with exact diagnosed-rank values."""
    return _run_straggler_case(
        kind="H2D_STRAGGLER",
        status="H2D STRAGGLER",
        phase="h2d",
        evidence={
            "culprit_rank": 0,
            "victim_rank": 1,
            "visible_metric": "backward",
            "visible_cost_ms": 100.0,
        },
        culprit_metrics={"h2d_ms": 84.0},
        victim_metrics={"h2d_ms": 4.0},
        run_name="h2d-straggler",
    )


def run_multi_compute_straggler() -> CardDoc:
    """run x distributed, Compute straggler with stored Forward phase."""
    return _run_straggler_case(
        kind="COMPUTE_STRAGGLER",
        status="COMPUTE STRAGGLER",
        phase="forward",
        evidence={
            "culprit_rank": 0,
            "victim_rank": 1,
            "visible_metric": "backward",
            "visible_cost_ms": 100.0,
        },
        culprit_metrics={"forward_ms": 100.0},
        victim_metrics={"forward_ms": 20.0},
        run_name="compute-straggler",
    )


def run_multi_generic_straggler() -> CardDoc:
    """run x DDP, generic straggler with a stored Backward sync gap."""
    return _run_straggler_case(
        kind="STRAGGLER",
        status="STRAGGLER",
        phase="sync",
        evidence={
            "culprit_rank": 0,
            "victim_rank": 1,
            "visible_metric": "backward",
            "visible_cost_ms": 100.0,
        },
        culprit_metrics={"backward_ms": 20.0},
        victim_metrics={"backward_ms": 120.0},
        run_name="generic-straggler",
    )


def run_multi_fsdp_straggler() -> CardDoc:
    """run x FSDP, generic straggler with stored Forward + Backward gap."""
    return _run_straggler_case(
        kind="STRAGGLER",
        status="STRAGGLER",
        phase="sync",
        evidence={
            "culprit_rank": 0,
            "victim_rank": 1,
            "visible_metric": "forward_backward",
            "visible_cost_ms": 100.0,
        },
        culprit_metrics={"forward_ms": 10.0, "backward_ms": 30.0},
        victim_metrics={"forward_ms": 40.0, "backward_ms": 100.0},
        run_name="fsdp-straggler",
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
    "run_partial_rank_payload": run_partial_rank_payload,
    "run_multi_duplicate_memory_pair": run_multi_duplicate_memory_pair,
    "run_multi_high_power_partial": run_multi_high_power_partial,
    "run_measured_zero_and_null": run_measured_zero_and_null,
    "run_unmeasured_resource_sections": run_unmeasured_resource_sections,
    "run_h2d_bound": run_h2d_bound,
    "run_compute_bound": run_compute_bound,
    "run_multi_h2d_straggler": run_multi_h2d_straggler,
    "run_multi_compute_straggler": run_multi_compute_straggler,
    "run_multi_generic_straggler": run_multi_generic_straggler,
    "run_multi_fsdp_straggler": run_multi_fsdp_straggler,
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
    "run_measured_zero_and_null",
    "run_unmeasured_resource_sections",
    "run_h2d_bound",
    "run_compute_bound",
    "watch_healthy",
    "watch_low_gpu_utilization",
    "watch_memory_pressure",
)


GOLDENS = {
    "run_input_bound_critical": """\
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|  TraceML Run Summary                                                                                                                                     |
|  bert_finetune · 1 rank · 1 GPU observed · 256 common steps · 52.4s                                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                                                          |
|  Verdict: INPUT-BOUND  (CRITICAL)                                                                                                                        |
|  Why: Input Wait took 64% of Step Time.                                                                                                                  |
|  Next: Increase workers, prefetch, or storage throughput.                                                                                                |
|                                                                                                                                                          |
|  STEP TIMING (Window Average), GPU Clock                      ||  STEP MEMORY: BALANCED                                                                  |
|  Step Time           200.4 ms  100%                           ||                                                                                         |
|  ├─ Input Wait       128.0 ms   64%  ◀  cause                 ||                                                                                         |
|  ├─ Compute           68.0 ms   34%                           ||  avg per-step peak           avg                                                        |
|  │  ├─ Forward        24.0 ms   12%                           ||  Allocated                   2.9 GB                                                     |
|  │  ├─ Backward       38.0 ms   19%                           ||  Reserved                    3.2 GB                                                     |
|  │  └─ Optimizer       6.0 ms    3%                           ||                                                                                         |
|  ├─ H2D                0.4 ms   <1%                           ||                                                                                         |
|  └─ Residual           3.6 ms    2%                           ||                                                                                         |
|  DataLoader fetch: 120.0 ms (CPU, supplemental)               ||                                                                                         |
|                                                                                                                                                          |
|  SYSTEM METRICS: LOW GPU UTIL                                 ||  PROCESS METRICS: NORMAL                                                                |
|  Evidence: GPU utilization averaged 24%.                      ||                                                                                         |
|                                                               ||                                                                                         |
|                         avg                                   ||                       avg                                                               |
|  CPU                    18%                                   ||  CPU capacity         14%                                                               |
|  RAM used               6.2 GB (19%)                          ||  RSS used             3.1 GB (10%)                                                      |
|  GPU util               24%                                   ||  CUDA used            2.9 GB                                                            |
|  GPU memory/device      3.3 GB (21%)                          ||  CUDA reserved        3.2 GB (20%)                                                      |
|  GPU temperature        42C                                   ||                                                                                         |
|  GPU power              58W                                   ||                                                                                         |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Full evidence: logs/bert_finetune/final_summary.json  (--html-report)                                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+""",
    "run_healthy": """\
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|  TraceML Run Summary                                                                                                                                     |
|  bert_finetune · 1 rank · 1 GPU observed · 256 common steps · 48.9s                                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                                                          |
|  Verdict: NO CLEAR PERFORMANCE BOTTLENECK                                                                                                                |
|  Why: The measured Step Time phases did not identify a material bottleneck.                                                                              |
|  Next: No data-pipeline or rank-skew bottleneck was detected; use model/kernel-level profiling if more speed is needed.                                  |
|                                                                                                                                                          |
|  STEP TIMING (Window Average), GPU Clock                      ||  STEP MEMORY: BALANCED                                                                  |
|  Step Time           191.0 ms  100%                           ||                                                                                         |
|  ├─ Input Wait         2.1 ms    1%                           ||                                                                                         |
|  ├─ Compute          185.0 ms   97%                           ||  avg per-step peak           avg                                                        |
|  │  ├─ Forward        65.0 ms   34%                           ||  Allocated                   2.9 GB                                                     |
|  │  ├─ Backward      110.0 ms   58%                           ||  Reserved                    3.2 GB                                                     |
|  │  └─ Optimizer      10.0 ms    5%                           ||                                                                                         |
|  ├─ H2D                1.9 ms    1%                           ||                                                                                         |
|  └─ Residual           2.0 ms    1%                           ||                                                                                         |
|  DataLoader fetch: 1.8 ms (CPU, supplemental)                 ||                                                                                         |
|                                                                                                                                                          |
|  SYSTEM METRICS: NORMAL                                       ||  PROCESS METRICS: NORMAL                                                                |
|                                                               ||                                                                                         |
|                                                               ||                                                                                         |
|                         avg                                   ||                       avg                                                               |
|  CPU                    42%                                   ||  CPU capacity         14%                                                               |
|  RAM used               6.2 GB (19%)                          ||  RSS used             3.1 GB (10%)                                                      |
|  GPU util               92%                                   ||  CUDA used            2.9 GB                                                            |
|  GPU memory/device      2.8 GB (17%)                          ||  CUDA reserved        3.2 GB (20%)                                                      |
|  GPU temperature        61C                                   ||                                                                                         |
|  GPU power              210W                                  ||                                                                                         |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Full evidence: logs/bert_finetune/final_summary.json  (--html-report)                                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+""",
    "run_input_bound_with_also": """\
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|  TraceML Run Summary                                                                                                                                     |
|  bert_finetune · 1 rank · 1 GPU observed · 256 common steps · 52.4s                                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                                                          |
|  Verdict: INPUT-BOUND  (CRITICAL)                                                                                                                        |
|  Why: Input Wait took 64% of Step Time.                                                                                                                  |
|  Next: Increase workers, prefetch, or storage throughput.                                                                                                |
|                                                                                                                                                          |
|  STEP TIMING (Window Average), GPU Clock                      ||  STEP MEMORY: MEMORY CREEP  (WARNING)                                                   |
|  Step Time           200.4 ms  100%                           ||  Evidence: Memory creep +1.2 GB (18.0%) · R0/N0                                         |
|  ├─ Input Wait       128.0 ms   64%  ◀  cause                 ||                                                                                         |
|  ├─ Compute           68.0 ms   34%                           ||  avg per-step peak           avg                                                        |
|  ├─ H2D                0.4 ms   <1%                           ||  Allocated                   3.9 GB                                                     |
|  └─ Residual           3.6 ms    2%                           ||  Reserved                    4.6 GB                                                     |
|                                                                                                                                                          |
|  SYSTEM METRICS: NORMAL                                       ||  PROCESS METRICS: NORMAL                                                                |
|                                                               ||                                                                                         |
|                                                               ||                                                                                         |
|                         avg                                   ||                       avg                                                               |
|  CPU                    18%                                   ||  CPU capacity         14%                                                               |
|  GPU util               88%                                   ||  RSS used             3.1 GB (10%)                                                      |
|  GPU temperature        42C                                   ||  CUDA used            2.9 GB                                                            |
|                                                               ||  CUDA reserved        3.2 GB (20%)                                                      |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Other findings:                                                                                                                                         |
|  ! Allocator reserved memory is 2.4x live tensor memory.  (WARNING)                                                                                      |
|                                                                                                                                                          |
|  Full evidence: logs/bert_finetune/final_summary.json  (--html-report)                                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+""",
    "run_low_gpu_utilization": """\
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|  TraceML Run Summary                                                                                                                                     |
|  bert_finetune · 1 rank · 1 GPU observed · 256 common steps · 1m 1s                                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                                                          |
|  Verdict: LOW GPU UTILIZATION                                                                                                                            |
|  Why: GPU utilization averaged 22%; the measured Step Time phases did not identify why.                                                                  |
|  Next: Inspect untraced work, validation/checkpointing, kernel efficiency, or missing instrumentation.                                                   |
|                                                                                                                                                          |
|  STEP TIMING (Window Average), GPU Clock                      ||  STEP MEMORY: BALANCED                                                                  |
|  Step Time           240.0 ms  100%                           ||                                                                                         |
|  ├─ Input Wait         3.1 ms    1%                           ||                                                                                         |
|  ├─ Compute          120.3 ms   50%                           ||  avg per-step peak           avg                                                        |
|  ├─ H2D                1.2 ms   <1%                           ||  Allocated                   2.9 GB                                                     |
|  └─ Residual         115.4 ms   48%                           ||  Reserved                    3.2 GB                                                     |
|                                                                                                                                                          |
|  SYSTEM METRICS: LOW GPU UTIL                                 ||  PROCESS METRICS: NORMAL                                                                |
|  Evidence: GPU utilization averaged 22%.                      ||                                                                                         |
|                                                               ||                                                                                         |
|                         avg                                   ||                       avg                                                               |
|  CPU                    17%                                   ||  CPU capacity         14%                                                               |
|  GPU util               22%                                   ||  RSS used             3.1 GB (10%)                                                      |
|  GPU temperature        44C                                   ||  CUDA used            2.9 GB                                                            |
|                                                               ||  CUDA reserved        3.2 GB (20%)                                                      |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Full evidence: logs/bert_finetune/final_summary.json  (--html-report)                                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+""",
    "run_cpu_only_input_bound": """\
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|  TraceML Run Summary                                                                                                                                     |
|  quickstart · 1 rank · CPU only (no GPU detected) · 100 common steps · 2.0s                                                                              |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                                                          |
|  Verdict: INPUT-BOUND  (WARNING)                                                                                                                         |
|  Why: Input Wait took 14% of Step Time.                                                                                                                  |
|  Next: Increase workers or prefetch.                                                                                                                     |
|                                                                                                                                                          |
|  STEP TIMING (Window Average), CPU Clock                      ||  STEP MEMORY: NO GPU                                                                    |
|  Step Time             1.4 ms  100%                           ||  Evidence: Step memory uses torch-based GPU memory telemetry.                           |
|  ├─ Input Wait         0.2 ms   14%  ◀  cause                 ||                                                                                         |
|  ├─ Compute            1.0 ms   75%                           ||                                                                                         |
|  └─ Residual           0.1 ms   11%                           ||                                                                                         |
|  DataLoader fetch: 0.2 ms (CPU, supplemental)                 ||                                                                                         |
|                                                                                                                                                          |
|  SYSTEM METRICS: NORMAL                                       ||  PROCESS METRICS: NORMAL                                                                |
|                                                               ||                                                                                         |
|                                                               ||                                                                                         |
|                         avg                                   ||                       avg                                                               |
|  CPU                    25%                                   ||  CPU capacity         14%                                                               |
|                                                               ||  RSS used             3.1 GB (10%)                                                      |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Full evidence: logs/session_20260806_090618/final_summary.json  (--html-report)                                                                         |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+""",
    "run_not_enough_step_data": """\
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|  TraceML Run Summary                                                                                                                                     |
|  quickstart · 1 rank · 1 GPU observed · 12 common steps · 4.0s                                                                                           |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                                                          |
|  Verdict: INSUFFICIENT STEP-TIME DATA                                                                                                                    |
|  Why: Only 12 completed steps were available, so the diagnosis is not stable.                                                                            |
|  Next: Run for more steps or ensure step timing is recorded.                                                                                             |
|                                                                                                                                                          |
|                                                               ||  STEP MEMORY: BALANCED                                                                  |
|                                                               ||                                                                                         |
|                                                                                                                                                          |
|  SYSTEM METRICS: NORMAL                                       ||  PROCESS METRICS: NORMAL                                                                |
|                                                               ||                                                                                         |
|                                                               ||                                                                                         |
|                         avg                                   ||                       avg                                                               |
|  CPU                    17%                                   ||  CPU capacity         14%                                                               |
|  GPU util               31%                                   ||  RSS used             3.1 GB (10%)                                                      |
|                                                               ||  CUDA used            2.9 GB                                                            |
|                                                               ||  CUDA reserved        3.2 GB (20%)                                                      |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Full evidence: logs/quickstart/final_summary.json  (--html-report)                                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+""",
    "run_step_timing_incomplete": """\
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|  TraceML Run Summary                                                                                                                                     |
|  quickstart · 1 rank · 1 GPU observed · 4.0s                                                                                                             |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                                                          |
|  Verdict: INSUFFICIENT STEP-TIME DATA                                                                                                                    |
|  Why: Missing timing signals: backward, optimizer.                                                                                                       |
|  Next: Instrument the missing phases; the Step Time section lists the missing signal names.                                                              |
|                                                                                                                                                          |
|                                                               ||  STEP MEMORY: BALANCED                                                                  |
|                                                               ||                                                                                         |
|                                                                                                                                                          |
|  SYSTEM METRICS: NORMAL                                       ||  PROCESS METRICS: NORMAL                                                                |
|                                                               ||                                                                                         |
|                                                               ||                                                                                         |
|                                                               ||                       avg                                                               |
|                                                               ||  CPU capacity         14%                                                               |
|                                                               ||  RSS used             3.1 GB (10%)                                                      |
|                                                               ||  CUDA used            2.9 GB                                                            |
|                                                               ||  CUDA reserved        3.2 GB (20%)                                                      |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Full evidence: logs/quickstart/final_summary.json  (--html-report)                                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+""",
    "run_multi_input_straggler": """\
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|  TraceML Run Summary                                                                                                                                     |
|  ddp_pretrain · 4/4 ranks · 4 GPUs observed · 2/2 nodes · 250 common steps · 40.1s                                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                                                          |
|  Verdict: INPUT STRAGGLER  (CRITICAL)                                                                                                                    |
|  Why: R0/N0 waited 254.5 ms for input; R1/N0 waited 3.8 ms for input.                                                                                    |
|  Next: Inspect input wait on the slow rank.                                                                                                              |
|  Scope: N = node · R = global rank · G = GPU index                                                                                                       |
|                                                                                                                                                          |
|  STEP TIMING (Median R1/N0), GPU Clock                        ||  STEP MEMORY: BALANCED · 4/4 ranks                                                      |
|  Step Time           303.7 ms  100%                           ||                                                                                         |
|  ├─ Input Wait         3.8 ms    1%                           ||                                                                                         |
|  ├─ Compute          259.5 ms   85%                           ||  avg per-step peak           median rank avg     worst rank avg                         |
|  │  ├─ Forward        80.0 ms   26%                           ||  Allocated                   8.5 GB              9.4 GB, R2/N1                          |
|  │  ├─ Backward      169.5 ms   56%                           ||  Reserved                    8.9 GB              9.8 GB, R2/N1                          |
|  │  └─ Optimizer      10.0 ms    3%                           ||                                                                                         |
|  ├─ H2D                1.1 ms   <1%                           ||                                                                                         |
|  └─ Residual          39.3 ms   13%                           ||                                                                                         |
|  DataLoader fetch: 3.7 ms (CPU, supplemental)                 ||                                                                                         |
|                                                                                                                                                          |
|  SYSTEM METRICS: LOW GPU UTIL · 2/2 nodes                     ||  PROCESS METRICS: NORMAL · 4/4 ranks                                                    |
|  Evidence: GPU utilization averaged 14%.                      ||                                                                                         |
|                                                               ||                                                                                         |
|                         median node avg   worst node avg      ||                       median rank avg   worst rank avg                                  |
|  CPU                    18%               26%, N1             ||  CPU capacity         12%               81%, R2/N1                                      |
|  RAM used               16.0 GB (27%)     20.8 GB (35%), N1   ||  RSS used             3.1 GB (10%)      5.4 GB (17%), R1/N0                             |
|  GPU util               9%                9%, N1              ||  CUDA used            2.9 GB            4.6 GB, R3/N1                                   |
|  GPU memory/device      5.0 GB (31%)      7.0 GB (44%), N1    ||  CUDA reserved        3.2 GB (20%)      6.8 GB (43%), R3/N1                             |
|  GPU temperature        58C               70C, N1             ||                                                                                         |
|  GPU power              220W              280W, N1            ||                                                                                         |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Full evidence: logs/ddp_pretrain/final_summary.json  (--html-report)                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+""",
    "run_multi_healthy": """\
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|  TraceML Run Summary                                                                                                                                     |
|  ddp_pretrain · 4/4 ranks · 4 GPUs observed · 1/1 node · 250 common steps · 38.7s                                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                                                          |
|  Verdict: NO CLEAR PERFORMANCE BOTTLENECK                                                                                                                |
|  Why: The measured Step Time phases did not identify a material bottleneck.                                                                              |
|  Next: No data-pipeline or rank-skew bottleneck was detected; use model/kernel-level profiling if more speed is needed.                                  |
|  Scope: N = node · R = global rank · G = GPU index                                                                                                       |
|                                                                                                                                                          |
|  STEP TIMING (Median R1/N0), GPU Clock                        ||  STEP MEMORY: BALANCED · 4/4 ranks                                                      |
|  Step Time           152.1 ms  100%                           ||                                                                                         |
|  ├─ Input Wait         2.0 ms    1%                           ||                                                                                         |
|  ├─ Compute          145.2 ms   95%                           ||  avg per-step peak           median rank avg     worst rank avg                         |
|  │  ├─ Forward        50.0 ms   33%                           ||  Allocated                   8.7 GB              8.8 GB, R2/N0                          |
|  │  ├─ Backward       85.2 ms   56%                           ||  Reserved                    9.0 GB              9.1 GB, R2/N0                          |
|  │  └─ Optimizer      10.0 ms    7%                           ||                                                                                         |
|  ├─ H2D                1.6 ms    1%                           ||                                                                                         |
|  └─ Residual           3.3 ms    2%                           ||                                                                                         |
|  DataLoader fetch: 1.8 ms (CPU, supplemental)                 ||                                                                                         |
|                                                                                                                                                          |
|  SYSTEM METRICS: NORMAL                                       ||  PROCESS METRICS: NORMAL · 4/4 ranks                                                    |
|                                                               ||                                                                                         |
|                                                               ||                                                                                         |
|                         avg                                   ||                       median rank avg   worst rank avg                                  |
|  CPU                    38%                                   ||  CPU capacity         12%               81%, R2/N0                                      |
|  RAM used               18.4 GB (31%)                         ||  RSS used             3.1 GB (10%)      5.4 GB (17%), R1/N0                             |
|  GPU util               94%                                   ||  CUDA used            2.9 GB            4.6 GB, R3/N0                                   |
|  GPU memory/device      6.0 GB (38%)                          ||  CUDA reserved        3.2 GB (20%)      6.8 GB (43%), R3/N0                             |
|  GPU temperature        64C                                   ||                                                                                         |
|  GPU power              240W                                  ||                                                                                         |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Full evidence: logs/ddp_pretrain/final_summary.json  (--html-report)                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+""",
    "run_single_gpu_on_shared_host": """\
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|  TraceML Run Summary                                                                                                                                     |
|  session_20260806_101725_5580d3 · 1 rank · 4 GPUs observed · 128 common steps · 12.1s                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                                                          |
|  Verdict: LOW GPU UTILIZATION                                                                                                                            |
|  Why: GPU utilization averaged 0.2%; the measured Step Time phases did not identify why.                                                                 |
|  Next: Inspect untraced work, validation/checkpointing, kernel efficiency, or missing instrumentation.                                                   |
|                                                                                                                                                          |
|  STEP TIMING (Window Average), GPU Clock                      ||  STEP MEMORY: BALANCED                                                                  |
|  Step Time            34.2 ms  100%                           ||                                                                                         |
|  ├─ Input Wait         2.5 ms    7%                           ||                                                                                         |
|  ├─ Compute           29.4 ms   86%                           ||  avg per-step peak           avg                                                        |
|  └─ Residual           2.3 ms    7%                           ||  Allocated                   18.8 MB                                                    |
|  DataLoader fetch: 2.5 ms (CPU, supplemental)                 ||  Reserved                    23.1 MB                                                    |
|                                                                                                                                                          |
|  SYSTEM METRICS: LOW GPU UTIL                                 ||  PROCESS METRICS: NORMAL                                                                |
|  Evidence: GPU utilization averaged 0.2%.                     ||                                                                                         |
|                                                               ||                                                                                         |
|                         avg                                   ||                       avg                                                               |
|  CPU                    0%                                    ||  CPU capacity         14%                                                               |
|  RAM used               4.0 GB (2%)                           ||  RSS used             3.1 GB (10%)                                                      |
|  GPU util               0%                                    ||  CUDA used            2.9 GB                                                            |
|  GPU memory/device      0.5 GB (3%)                           ||  CUDA reserved        3.2 GB (20%)                                                      |
|  GPU temperature        34C                                   ||                                                                                         |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Full evidence: logs/session_20260806_101725_5580d3/final_summary.json  (--html-report)                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+""",
    "run_multi_residual_heavy": """\
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|  TraceML Run Summary                                                                                                                                     |
|  session_20260806_101837_1b81f7 · 4/4 ranks · 4 GPUs observed · 1/1 node · 128 common steps · 2.7s                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                                                          |
|  Verdict: RESIDUAL-HEAVY  (WARNING)                                                                                                                      |
|  Why: Residual time took 14% of Step Time.                                                                                                               |
|  Next: Investigate untraced work between steps.                                                                                                          |
|  Scope: N = node · R = global rank · G = GPU index                                                                                                       |
|                                                                                                                                                          |
|  STEP TIMING (Median R1/N0), GPU Clock                        ||  STEP MEMORY: BALANCED · 4/4 ranks                                                      |
|  Step Time             5.2 ms  100%                           ||                                                                                         |
|  ├─ Input Wait         0.6 ms   11%                           ||                                                                                         |
|  ├─ Compute            3.8 ms   74%                           ||  avg per-step peak           median rank avg     worst rank avg                         |
|  │  ├─ Forward         1.3 ms   25%                           ||  Allocated                   18.8 MB             21.0 MB, R0/N0                         |
|  │  ├─ Backward        2.2 ms   42%                           ||  Reserved                    23.1 MB             25.1 MB, R0/N0                         |
|  │  └─ Optimizer       0.3 ms    6%                           ||                                                                                         |
|  ├─ H2D                0.1 ms    2%                           ||                                                                                         |
|  └─ Residual           0.7 ms   14%  ◀  cause                 ||                                                                                         |
|  DataLoader fetch: 0.6 ms (CPU, supplemental)                 ||                                                                                         |
|                                                                                                                                                          |
|  SYSTEM METRICS: LOW GPU UTIL                                 ||  PROCESS METRICS: NORMAL · 4/4 ranks                                                    |
|  Evidence: GPU utilization averaged 6.8%.                     ||                                                                                         |
|                                                               ||                                                                                         |
|                         avg                                   ||                       median rank avg   worst rank avg                                  |
|  CPU                    2%                                    ||  CPU capacity         12%               81%, R2/N0                                      |
|  GPU util               7%                                    ||  RSS used             3.1 GB (10%)      5.4 GB (17%), R1/N0                             |
|                                                               ||  CUDA used            2.9 GB            4.6 GB, R3/N0                                   |
|                                                               ||  CUDA reserved        3.2 GB (20%)      6.8 GB (43%), R3/N0                             |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Full evidence: logs/session_20260806_101837_1b81f7/final_summary.json  (--html-report)                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+""",
    "run_partial_rank_payload": """\
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|  TraceML Run Summary                                                                                                                                     |
|  crossed-points · 4/4 ranks · 4 GPUs observed · 2/2 nodes · 40 common steps · 5.0s                                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                                                          |
|  Verdict: NO CLEAR PERFORMANCE BOTTLENECK                                                                                                                |
|  Why: The measured Step Time phases did not identify a material bottleneck.                                                                              |
|  Next: No data-pipeline or rank-skew bottleneck was detected; use model/kernel-level profiling if more speed is needed.                                  |
|  Scope: N = node · R = global rank · G = GPU index                                                                                                       |
|                                                                                                                                                          |
|  STEP TIMING unavailable: selected rank row missing.          ||  STEP MEMORY: BALANCED · 4/4 ranks                                                      |
|                                                               ||                                                                                         |
|                                                               ||                                                                                         |
|                                                               ||  avg per-step peak           median rank avg     worst rank avg                         |
|                                                               ||  Allocated                                       5.0 GB, R1/N0                          |
|                                                               ||  Reserved                                        6.0 GB, R1/N0                          |
|                                                                                                                                                          |
|  SYSTEM METRICS: NORMAL · 2/2 nodes                           ||  PROCESS METRICS: NORMAL · 4/4 ranks                                                    |
|                                                               ||                                                                                         |
|                                                               ||                                                                                         |
|                         median node avg   worst node avg      ||                       median rank avg   worst rank avg                                  |
|  CPU                    10%               30%, N0             ||  CPU capacity         12%               81%, R2/N1                                      |
|  RAM used               6.0 GB (20%)      10.0 GB (30%), N1   ||  RSS used             3.1 GB (10%)      5.4 GB (17%), R1/N0                             |
|  GPU util               85%               85%, N1             ||  CUDA used            2.9 GB            4.6 GB, R3/N1                                   |
|  GPU memory/device      2.0 GB (12%)      6.0 GB (38%), N0    ||  CUDA reserved        3.2 GB (20%)      6.8 GB (43%), R3/N1                             |
|  GPU temperature        50C               70C, N0             ||                                                                                         |
|  GPU power              220W              280W, N0            ||                                                                                         |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Full evidence: logs/crossed-points/final_summary.json  (--html-report)                                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+""",
    "run_multi_duplicate_memory_pair": """\
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|  TraceML Run Summary                                                                                                                                     |
|  crossed-points · 4/4 ranks · 4 GPUs observed · 2/2 nodes · 40 common steps · 5.0s                                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                                                          |
|  Verdict: NO CLEAR PERFORMANCE BOTTLENECK                                                                                                                |
|  Why: The measured Step Time phases did not identify a material bottleneck.                                                                              |
|  Next: No data-pipeline or rank-skew bottleneck was detected; use model/kernel-level profiling if more speed is needed.                                  |
|  Scope: N = node · R = global rank · G = GPU index                                                                                                       |
|                                                                                                                                                          |
|  STEP TIMING (Median R2/N1), GPU Clock                        ||  STEP MEMORY: BALANCED · 4/4 ranks                                                      |
|  Step Time           100.0 ms  100%                           ||                                                                                         |
|  ├─ Input Wait        10.0 ms   10%                           ||                                                                                         |
|  ├─ Compute           80.0 ms   80%                           ||  avg per-step peak           median rank avg     worst rank avg                         |
|  │  ├─ Forward        20.0 ms   20%                           ||  Allocated                   3.0 GB              3.0 GB, R3/N1                          |
|  │  ├─ Backward       50.0 ms   50%                           ||  Reserved                    4.0 GB              4.0 GB, R3/N1                          |
|  │  └─ Optimizer      10.0 ms   10%                           ||                                                                                         |
|  ├─ H2D                2.0 ms    2%                           ||                                                                                         |
|  └─ Residual           8.0 ms    8%                           ||                                                                                         |
|  DataLoader fetch: 7.0 ms (CPU, supplemental)                 ||                                                                                         |
|                                                                                                                                                          |
|  SYSTEM METRICS: NORMAL · 2/2 nodes                           ||  PROCESS METRICS: NORMAL · 4/4 ranks                                                    |
|                                                               ||                                                                                         |
|                                                               ||                                                                                         |
|                         median node avg   worst node avg      ||                       median rank avg   worst rank avg                                  |
|  CPU                    10%               30%, N0             ||  CPU capacity         12%               81%, R2/N1                                      |
|  RAM used               6.0 GB (20%)      10.0 GB (30%), N1   ||  RSS used             3.1 GB (10%)      5.4 GB (17%), R1/N0                             |
|  GPU util               85%               85%, N1             ||  CUDA used            2.9 GB            4.6 GB, R3/N1                                   |
|  GPU memory/device      2.0 GB (12%)      6.0 GB (38%), N0    ||  CUDA reserved        3.2 GB (20%)      6.8 GB (43%), R3/N1                             |
|  GPU temperature        50C               70C, N0             ||                                                                                         |
|  GPU power              220W              280W, N0            ||                                                                                         |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Full evidence: logs/crossed-points/final_summary.json  (--html-report)                                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+""",
    "run_multi_high_power_partial": """\
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|  TraceML Run Summary                                                                                                                                     |
|  multi-high-power · 4/4 ranks · 4 GPUs observed · 2/4 nodes · 40 common steps · 5.0s                                                                     |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                                                          |
|  Verdict: NO CLEAR PERFORMANCE BOTTLENECK                                                                                                                |
|  Why: The measured Step Time phases did not identify a material bottleneck.                                                                              |
|  Next: No data-pipeline or rank-skew bottleneck was detected; use model/kernel-level profiling if more speed is needed.                                  |
|  Scope: N = node · R = global rank · G = GPU index                                                                                                       |
|                                                                                                                                                          |
|  STEP TIMING (Median R2/N1), GPU Clock                        ||  STEP MEMORY: BALANCED · 4/4 ranks                                                      |
|  Step Time           100.0 ms  100%                           ||                                                                                         |
|  ├─ Input Wait        10.0 ms   10%                           ||                                                                                         |
|  ├─ Compute           80.0 ms   80%                           ||  avg per-step peak           median rank avg     worst rank avg                         |
|  │  ├─ Forward        20.0 ms   20%                           ||  Allocated                   3.0 GB              5.0 GB, R1/N0                          |
|  │  ├─ Backward       50.0 ms   50%                           ||  Reserved                    4.0 GB              6.0 GB, R1/N0                          |
|  │  └─ Optimizer      10.0 ms   10%                           ||                                                                                         |
|  ├─ H2D                2.0 ms    2%                           ||                                                                                         |
|  └─ Residual           8.0 ms    8%                           ||                                                                                         |
|  DataLoader fetch: 7.0 ms (CPU, supplemental)                 ||                                                                                         |
|                                                                                                                                                          |
|  SYSTEM METRICS: HIGH GPU PWR  (WARNING) · 2/4 nodes          ||  PROCESS METRICS: NORMAL · 4/4 ranks                                                    |
|  Evidence: GPU power 83.7% of limit · N0/G0                   ||                                                                                         |
|                                                               ||                                                                                         |
|                         median node avg   worst node avg      ||                       median rank avg   worst rank avg                                  |
|  CPU                    10%               30%, N0             ||  CPU capacity         12%               81%, R2/N1                                      |
|  RAM used               6.0 GB (20%)      10.0 GB (30%), N1   ||  RSS used             3.1 GB (10%)      5.4 GB (17%), R1/N0                             |
|  GPU util               85%               85%, N1             ||  CUDA used            2.9 GB            4.6 GB, R3/N1                                   |
|  GPU memory/device      2.0 GB (12%)      6.0 GB (38%), N0    ||  CUDA reserved        3.2 GB (20%)      6.8 GB (43%), R3/N1                             |
|  GPU temperature        50C               70C, N0             ||                                                                                         |
|  GPU power              220W              280W, N0            ||                                                                                         |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Full evidence: logs/multi-high-power/final_summary.json  (--html-report)                                                                                |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+""",
    "run_measured_zero_and_null": """\
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|  TraceML Run Summary                                                                                                                                     |
|  measured-zero · 1 rank · 1 GPU observed · 20 common steps · 1.0s                                                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                                                          |
|  Verdict: NO CLEAR PERFORMANCE BOTTLENECK                                                                                                                |
|  Why: The measured Step Time phases did not identify a material bottleneck.                                                                              |
|  Next: No data-pipeline or rank-skew bottleneck was detected; use model/kernel-level profiling if more speed is needed.                                  |
|                                                                                                                                                          |
|  STEP TIMING (Window Average), GPU Clock                      ||  STEP MEMORY: BALANCED                                                                  |
|  Step Time            10.0 ms  100%                           ||                                                                                         |
|  ├─ Input Wait         0.0 ms    0%                           ||                                                                                         |
|  ├─ Compute           10.0 ms  100%                           ||  avg per-step peak           avg                                                        |
|  │  ├─ Forward         0.0 ms    0%                           ||  Allocated                   0.0 MB                                                     |
|  │  ├─ Backward       10.0 ms  100%                           ||  Reserved                    0.0 MB                                                     |
|  │  └─ Optimizer       0.0 ms    0%                           ||                                                                                         |
|  ├─ H2D                0.0 ms    0%                           ||                                                                                         |
|  └─ Residual           0.0 ms    0%                           ||                                                                                         |
|  DataLoader fetch: 0.0 ms (CPU, supplemental)                 ||                                                                                         |
|                                                                                                                                                          |
|  SYSTEM METRICS: NORMAL                                       ||  PROCESS METRICS: NORMAL                                                                |
|                                                               ||                                                                                         |
|                                                               ||                                                                                         |
|                         avg                                   ||                       avg                                                               |
|  CPU                    0%                                    ||  CPU capacity         0%                                                                |
|  RAM used               0.0 MB (0%)                           ||  RSS used             0.0 MB (0%)                                                       |
|  GPU util               0%                                    ||                                                                                         |
|  GPU temperature        0C                                    ||                                                                                         |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Full evidence: logs/measured-zero/final_summary.json  (--html-report)                                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+""",
    "run_unmeasured_resource_sections": """\
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|  TraceML Run Summary                                                                                                                                     |
|  unmeasured · 1 rank · CPU only (no GPU detected) · 20 common steps · 1.0s                                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                                                          |
|  Verdict: NO CLEAR PERFORMANCE BOTTLENECK                                                                                                                |
|  Why: The measured Step Time phases did not identify a material bottleneck.                                                                              |
|  Next: No data-pipeline or rank-skew bottleneck was detected; use model/kernel-level profiling if more speed is needed.                                  |
|                                                                                                                                                          |
|  STEP TIMING (Window Average), CPU Clock                      ||  STEP MEMORY: NO GPU · 0/1 rank                                                         |
|  Step Time            10.0 ms  100%                           ||  Evidence: Step memory uses torch-based GPU memory telemetry.                           |
|  ├─ Input Wait         1.0 ms   10%                           ||                                                                                         |
|  ├─ Compute            8.0 ms   80%                           ||                                                                                         |
|  └─ Residual           1.0 ms   10%                           ||                                                                                         |
|                                                                                                                                                          |
|  SYSTEM METRICS: NO DATA                                      ||  PROCESS METRICS: NO DATA · 0/1 rank                                                    |
|  Evidence: System sampler produced no rows.                   ||  Evidence: Process telemetry was not measured.                                          |
|                                                               ||                                                                                         |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Full evidence: logs/unmeasured/final_summary.json  (--html-report)                                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+""",
    "run_h2d_bound": """\
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|  TraceML Run Summary                                                                                                                                     |
|  h2d-bound · 1 rank · 1 GPU observed · 40 common steps · 5.0s                                                                                            |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                                                          |
|  Verdict: H2D-BOUND  (WARNING)                                                                                                                           |
|  Why: H2D transfers took 20% of Step Time.                                                                                                               |
|  Next: Inspect pinned memory and batch transfers.                                                                                                        |
|                                                                                                                                                          |
|  STEP TIMING (Window Average), GPU Clock                      ||  STEP MEMORY: BALANCED                                                                  |
|  Step Time           100.0 ms  100%                           ||                                                                                         |
|  ├─ Input Wait         5.0 ms    5%                           ||                                                                                         |
|  ├─ Compute           70.0 ms   70%                           ||  avg per-step peak           avg                                                        |
|  │  ├─ Forward        20.0 ms   20%                           ||  Allocated                   2.0 GB                                                     |
|  │  ├─ Backward       45.0 ms   45%                           ||  Reserved                    2.4 GB                                                     |
|  │  └─ Optimizer       5.0 ms    5%                           ||                                                                                         |
|  ├─ H2D               20.0 ms   20%  ◀  cause                 ||                                                                                         |
|  └─ Residual           5.0 ms    5%                           ||                                                                                         |
|                                                                                                                                                          |
|  SYSTEM METRICS: NORMAL                                       ||  PROCESS METRICS: NORMAL                                                                |
|                                                               ||                                                                                         |
|                                                               ||                                                                                         |
|                         avg                                   ||                       avg                                                               |
|  GPU util               72%                                   ||  CPU capacity         14%                                                               |
|                                                               ||  RSS used             3.1 GB (10%)                                                      |
|                                                               ||  CUDA used            2.9 GB                                                            |
|                                                               ||  CUDA reserved        3.2 GB (20%)                                                      |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Full evidence: logs/h2d-bound/final_summary.json  (--html-report)                                                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+""",
    "run_compute_bound": """\
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|  TraceML Run Summary                                                                                                                                     |
|  compute-bound · 1 rank · 1 GPU observed · 40 common steps · 5.0s                                                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                                                          |
|  Verdict: COMPUTE-BOUND                                                                                                                                  |
|  Why: Compute took 95% of Step Time; Backward was the largest compute phase.                                                                             |
|  Next: Optimize model compute or reduce step cost.                                                                                                       |
|                                                                                                                                                          |
|  STEP TIMING (Window Average), GPU Clock                      ||  STEP MEMORY: BALANCED                                                                  |
|  Step Time           100.0 ms  100%                           ||                                                                                         |
|  ├─ Input Wait         1.0 ms    1%                           ||                                                                                         |
|  ├─ Compute           95.0 ms   95%  ◀  cause                 ||  avg per-step peak           avg                                                        |
|  │  ├─ Forward        20.0 ms   20%                           ||  Allocated                   2.0 GB                                                     |
|  │  ├─ Backward       65.0 ms   65%                           ||  Reserved                    2.4 GB                                                     |
|  │  └─ Optimizer      10.0 ms   10%                           ||                                                                                         |
|  ├─ H2D                1.0 ms    1%                           ||                                                                                         |
|  └─ Residual           3.0 ms    3%                           ||                                                                                         |
|                                                                                                                                                          |
|  SYSTEM METRICS: NORMAL                                       ||  PROCESS METRICS: NORMAL                                                                |
|                                                               ||                                                                                         |
|                                                               ||                                                                                         |
|                         avg                                   ||                       avg                                                               |
|  GPU util               95%                                   ||  CPU capacity         14%                                                               |
|                                                               ||  RSS used             3.1 GB (10%)                                                      |
|                                                               ||  CUDA used            2.9 GB                                                            |
|                                                               ||  CUDA reserved        3.2 GB (20%)                                                      |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Full evidence: logs/compute-bound/final_summary.json  (--html-report)                                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+""",
    "run_multi_h2d_straggler": """\
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|  TraceML Run Summary                                                                                                                                     |
|  h2d-straggler · 4/4 ranks · 4 GPUs observed · 2/2 nodes · 40 common steps · 5.0s                                                                        |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                                                          |
|  Verdict: H2D STRAGGLER  (WARNING)                                                                                                                       |
|  Why: R0/N0 spent 84.0 ms on H2D transfers; R1/N0 spent 4.0 ms on H2D transfers.                                                                         |
|  Next: Inspect the diagnosed ranks.                                                                                                                      |
|  Scope: N = node · R = global rank · G = GPU index                                                                                                       |
|                                                                                                                                                          |
|  STEP TIMING (Median R2/N1), GPU Clock                        ||  STEP MEMORY: BALANCED · 4/4 ranks                                                      |
|  Step Time           100.0 ms  100%                           ||                                                                                         |
|  ├─ Input Wait        10.0 ms   10%                           ||                                                                                         |
|  ├─ Compute           80.0 ms   80%                           ||  avg per-step peak           median rank avg     worst rank avg                         |
|  │  ├─ Forward        20.0 ms   20%                           ||  Allocated                   3.0 GB              5.0 GB, R1/N0                          |
|  │  ├─ Backward       50.0 ms   50%                           ||  Reserved                    4.0 GB              6.0 GB, R1/N0                          |
|  │  └─ Optimizer      10.0 ms   10%                           ||                                                                                         |
|  ├─ H2D                2.0 ms    2%                           ||                                                                                         |
|  └─ Residual           8.0 ms    8%                           ||                                                                                         |
|  DataLoader fetch: 7.0 ms (CPU, supplemental)                 ||                                                                                         |
|                                                                                                                                                          |
|  SYSTEM METRICS: NORMAL · 2/2 nodes                           ||  PROCESS METRICS: NORMAL · 4/4 ranks                                                    |
|                                                               ||                                                                                         |
|                                                               ||                                                                                         |
|                         median node avg   worst node avg      ||                       median rank avg   worst rank avg                                  |
|  CPU                    10%               30%, N0             ||  CPU capacity         12%               81%, R2/N1                                      |
|  RAM used               6.0 GB (20%)      10.0 GB (30%), N1   ||  RSS used             3.1 GB (10%)      5.4 GB (17%), R1/N0                             |
|  GPU util               85%               85%, N1             ||  CUDA used            2.9 GB            4.6 GB, R3/N1                                   |
|  GPU memory/device      2.0 GB (12%)      6.0 GB (38%), N0    ||  CUDA reserved        3.2 GB (20%)      6.8 GB (43%), R3/N1                             |
|  GPU temperature        50C               70C, N0             ||                                                                                         |
|  GPU power              220W              280W, N0            ||                                                                                         |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Full evidence: logs/h2d-straggler/final_summary.json  (--html-report)                                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+""",
    "run_multi_compute_straggler": """\
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|  TraceML Run Summary                                                                                                                                     |
|  compute-straggler · 4/4 ranks · 4 GPUs observed · 2/2 nodes · 40 common steps · 5.0s                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                                                          |
|  Verdict: COMPUTE STRAGGLER  (WARNING)                                                                                                                   |
|  Why: R0/N0 spent 100.0 ms in Forward; R1/N0 spent 20.0 ms in Forward.                                                                                   |
|  Next: Inspect the diagnosed ranks.                                                                                                                      |
|  Scope: N = node · R = global rank · G = GPU index                                                                                                       |
|                                                                                                                                                          |
|  STEP TIMING (Median R2/N1), GPU Clock                        ||  STEP MEMORY: BALANCED · 4/4 ranks                                                      |
|  Step Time           100.0 ms  100%                           ||                                                                                         |
|  ├─ Input Wait        10.0 ms   10%                           ||                                                                                         |
|  ├─ Compute           80.0 ms   80%                           ||  avg per-step peak           median rank avg     worst rank avg                         |
|  │  ├─ Forward        20.0 ms   20%                           ||  Allocated                   3.0 GB              5.0 GB, R1/N0                          |
|  │  ├─ Backward       50.0 ms   50%                           ||  Reserved                    4.0 GB              6.0 GB, R1/N0                          |
|  │  └─ Optimizer      10.0 ms   10%                           ||                                                                                         |
|  ├─ H2D                2.0 ms    2%                           ||                                                                                         |
|  └─ Residual           8.0 ms    8%                           ||                                                                                         |
|  DataLoader fetch: 7.0 ms (CPU, supplemental)                 ||                                                                                         |
|                                                                                                                                                          |
|  SYSTEM METRICS: NORMAL · 2/2 nodes                           ||  PROCESS METRICS: NORMAL · 4/4 ranks                                                    |
|                                                               ||                                                                                         |
|                                                               ||                                                                                         |
|                         median node avg   worst node avg      ||                       median rank avg   worst rank avg                                  |
|  CPU                    10%               30%, N0             ||  CPU capacity         12%               81%, R2/N1                                      |
|  RAM used               6.0 GB (20%)      10.0 GB (30%), N1   ||  RSS used             3.1 GB (10%)      5.4 GB (17%), R1/N0                             |
|  GPU util               85%               85%, N1             ||  CUDA used            2.9 GB            4.6 GB, R3/N1                                   |
|  GPU memory/device      2.0 GB (12%)      6.0 GB (38%), N0    ||  CUDA reserved        3.2 GB (20%)      6.8 GB (43%), R3/N1                             |
|  GPU temperature        50C               70C, N0             ||                                                                                         |
|  GPU power              220W              280W, N0            ||                                                                                         |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Full evidence: logs/compute-straggler/final_summary.json  (--html-report)                                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+""",
    "run_multi_generic_straggler": """\
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|  TraceML Run Summary                                                                                                                                     |
|  generic-straggler · 4/4 ranks · 4 GPUs observed · 2/2 nodes · 40 common steps · 5.0s                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                                                          |
|  Verdict: STRAGGLER  (WARNING)                                                                                                                           |
|  Why: R0/N0 was the straggler; R1/N0 spent 100.0 ms longer in Backward waiting at synchronization. No measured component clearly explained the gap.      |
|  Next: Inspect the diagnosed ranks.                                                                                                                      |
|  Scope: N = node · R = global rank · G = GPU index                                                                                                       |
|                                                                                                                                                          |
|  STEP TIMING (Median R2/N1), GPU Clock                        ||  STEP MEMORY: BALANCED · 4/4 ranks                                                      |
|  Step Time           100.0 ms  100%                           ||                                                                                         |
|  ├─ Input Wait        10.0 ms   10%                           ||                                                                                         |
|  ├─ Compute           80.0 ms   80%                           ||  avg per-step peak           median rank avg     worst rank avg                         |
|  │  ├─ Forward        20.0 ms   20%                           ||  Allocated                   3.0 GB              5.0 GB, R1/N0                          |
|  │  ├─ Backward       50.0 ms   50%                           ||  Reserved                    4.0 GB              6.0 GB, R1/N0                          |
|  │  └─ Optimizer      10.0 ms   10%                           ||                                                                                         |
|  ├─ H2D                2.0 ms    2%                           ||                                                                                         |
|  └─ Residual           8.0 ms    8%                           ||                                                                                         |
|  DataLoader fetch: 7.0 ms (CPU, supplemental)                 ||                                                                                         |
|                                                                                                                                                          |
|  SYSTEM METRICS: NORMAL · 2/2 nodes                           ||  PROCESS METRICS: NORMAL · 4/4 ranks                                                    |
|                                                               ||                                                                                         |
|                                                               ||                                                                                         |
|                         median node avg   worst node avg      ||                       median rank avg   worst rank avg                                  |
|  CPU                    10%               30%, N0             ||  CPU capacity         12%               81%, R2/N1                                      |
|  RAM used               6.0 GB (20%)      10.0 GB (30%), N1   ||  RSS used             3.1 GB (10%)      5.4 GB (17%), R1/N0                             |
|  GPU util               85%               85%, N1             ||  CUDA used            2.9 GB            4.6 GB, R3/N1                                   |
|  GPU memory/device      2.0 GB (12%)      6.0 GB (38%), N0    ||  CUDA reserved        3.2 GB (20%)      6.8 GB (43%), R3/N1                             |
|  GPU temperature        50C               70C, N0             ||                                                                                         |
|  GPU power              220W              280W, N0            ||                                                                                         |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Full evidence: logs/generic-straggler/final_summary.json  (--html-report)                                                                               |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+""",
    "run_multi_fsdp_straggler": """\
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|  TraceML Run Summary                                                                                                                                     |
|  fsdp-straggler · 4/4 ranks · 4 GPUs observed · 2/2 nodes · 40 common steps · 5.0s                                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                                                          |
|  Verdict: STRAGGLER  (WARNING)                                                                                                                           |
|  Why: R0/N0 was the straggler; R1/N0 spent 100.0 ms longer in Forward + Backward waiting at synchronization. No measured component clearly explained the |
|  gap.                                                                                                                                                    |
|  Next: Inspect the diagnosed ranks.                                                                                                                      |
|  Scope: N = node · R = global rank · G = GPU index                                                                                                       |
|                                                                                                                                                          |
|  STEP TIMING (Median R2/N1), GPU Clock                        ||  STEP MEMORY: BALANCED · 4/4 ranks                                                      |
|  Step Time           100.0 ms  100%                           ||                                                                                         |
|  ├─ Input Wait        10.0 ms   10%                           ||                                                                                         |
|  ├─ Compute           80.0 ms   80%                           ||  avg per-step peak           median rank avg     worst rank avg                         |
|  │  ├─ Forward        20.0 ms   20%                           ||  Allocated                   3.0 GB              5.0 GB, R1/N0                          |
|  │  ├─ Backward       50.0 ms   50%                           ||  Reserved                    4.0 GB              6.0 GB, R1/N0                          |
|  │  └─ Optimizer      10.0 ms   10%                           ||                                                                                         |
|  ├─ H2D                2.0 ms    2%                           ||                                                                                         |
|  └─ Residual           8.0 ms    8%                           ||                                                                                         |
|  DataLoader fetch: 7.0 ms (CPU, supplemental)                 ||                                                                                         |
|                                                                                                                                                          |
|  SYSTEM METRICS: NORMAL · 2/2 nodes                           ||  PROCESS METRICS: NORMAL · 4/4 ranks                                                    |
|                                                               ||                                                                                         |
|                                                               ||                                                                                         |
|                         median node avg   worst node avg      ||                       median rank avg   worst rank avg                                  |
|  CPU                    10%               30%, N0             ||  CPU capacity         12%               81%, R2/N1                                      |
|  RAM used               6.0 GB (20%)      10.0 GB (30%), N1   ||  RSS used             3.1 GB (10%)      5.4 GB (17%), R1/N0                             |
|  GPU util               85%               85%, N1             ||  CUDA used            2.9 GB            4.6 GB, R3/N1                                   |
|  GPU memory/device      2.0 GB (12%)      6.0 GB (38%), N0    ||  CUDA reserved        3.2 GB (20%)      6.8 GB (43%), R3/N1                             |
|  GPU temperature        50C               70C, N0             ||                                                                                         |
|  GPU power              220W              280W, N0            ||                                                                                         |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Full evidence: logs/fsdp-straggler/final_summary.json  (--html-report)                                                                                  |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+""",
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


def _logical_text(text: str) -> str:
    """Collapse wrapped card rows for wording-focused assertions."""
    rows = []
    for line in text.splitlines():
        if line.startswith("|") and line.endswith("|"):
            body = line[3:-1].strip()
            if body:
                rows.append(body)
    return " ".join(rows)


def _process_case_doc(
    process: Dict[str, Any], *, world_size: Optional[int]
) -> CardDoc:
    """Build one Process payload inside an otherwise balanced Run card."""
    return _card(
        profile="run",
        system=_system_single(gpu_util_percent=90.0),
        process=process,
        step_time=_step_time_single(
            diagnosis=_issue("BALANCED", "BALANCED"),
            steps_analyzed=20,
            step_time_ms=10.0,
            input_wait_ms=1.0,
            traced_step_time_ms=9.0,
            compute_ms=8.0,
            residual_ms=1.0,
        ),
        step_memory=_step_memory_single(None, None),
        meta=_meta(run_name="process-case", world_size=world_size),
        duration_s=1.0,
        artifact_hint="logs/process-case/final_summary.json",
    )


def _render_process_case(
    process: Dict[str, Any], *, world_size: Optional[int]
) -> str:
    """Render a Process-focused Run case as plain terminal text."""
    return card_to_plain(_process_case_doc(process, world_size=world_size))


def _process_with_diagnosis(
    diagnosis: Dict[str, Any], *, multi_rank: bool
) -> Dict[str, Any]:
    """Attach one Process diagnosis to a measured single/multi fixture."""
    process = copy.deepcopy(
        _ONE_NODE_MULTI_PROCESS if multi_rank else _NORMAL_PROCESS
    )
    process["diagnosis"] = diagnosis
    process["issues"] = [diagnosis]
    return process


def _process_block(text: str) -> str:
    """Extract non-empty Process-pane rows from the parallel resource block."""
    rows: List[str] = []
    in_process_block = False
    for line in text.splitlines():
        if "PROCESS METRICS:" in line:
            in_process_block = True
        if not in_process_block:
            continue
        if in_process_block and "||  " not in line:
            break
        if "||  " not in line:
            continue
        process_text = line.split("||  ", 1)[1][:-1].rstrip()
        if process_text:
            rows.append(process_text)
    return "\n".join(rows)


def _coherent_multi_inputs(
    *, reserved_points_measured: bool = True
) -> Dict[str, Any]:
    """Build crossed aggregate points backed by coherent grouped rows."""
    node_ranks = {0: 0, 1: 0, 2: 1, 3: 1}
    system = _system_multi_node(
        diagnosis=_issue("NORMAL", "NORMAL"),
        average={
            "cpu_percent": 20.0,
            "ram_bytes": 8.0e9,
            "ram_percent": 25.0,
            "gpu_util_percent": 90.0,
            "gpu_mem_bytes": 4.0e9,
            "gpu_mem_percent": 25.0,
            "gpu_temp_c": 60.0,
            "gpu_power_w": 250.0,
        },
        median={
            "cpu_percent": _point(10.0, "1"),
            # Crossed byte indexes prove percent anchors select coherent rows.
            "ram_bytes": _point(999.0e9, "1"),
            "ram_percent": _point(20.0, "0"),
            "gpu_util_percent": _point(85.0, "1"),
            "gpu_mem_bytes": _point(777.0e9, "0"),
            "gpu_mem_percent": _point(12.0, "1"),
            "gpu_temp_c": _point(50.0, "1"),
            "gpu_power_w": _point(220.0, "1"),
        },
        worst={
            "cpu_percent": _point(30.0, "0"),
            "ram_bytes": _point(888.0e9, "0"),
            "ram_percent": _point(30.0, "1"),
            "gpu_util_percent": _point(85.0, "1"),
            "gpu_mem_bytes": _point(666.0e9, "1"),
            "gpu_mem_percent": _point(38.0, "0"),
            "gpu_temp_c": _point(70.0, "0"),
            "gpu_power_w": _point(280.0, "0"),
        },
        metrics_by_node={
            0: {
                "cpu_percent": 30.0,
                "ram_bytes": 6.0e9,
                "ram_percent": 20.0,
                "gpu_util_percent": 95.0,
                "gpu_mem_bytes": 6.0e9,
                "gpu_mem_percent": 38.0,
                "gpu_temp_c": 70.0,
                "gpu_power_w": 280.0,
            },
            1: {
                "cpu_percent": 10.0,
                "ram_bytes": 10.0e9,
                "ram_percent": 30.0,
                "gpu_util_percent": 85.0,
                "gpu_mem_bytes": 2.0e9,
                "gpu_mem_percent": 12.0,
                "gpu_temp_c": 50.0,
                "gpu_power_w": 220.0,
            },
        },
    )
    step_time = _step_time_multi_rank(
        diagnosis=_issue("BALANCED", "BALANCED"),
        steps_analyzed=40,
        # The aggregate points intentionally select different ranks. Only the
        # step_time_ms index may select the representative grouped row.
        median={
            "step_time_ms": _point(100.0, "2"),
            "input_wait_ms": _point(777.0, "0"),
            "traced_step_time_ms": _point(666.0, "1"),
            "compute_ms": _point(555.0, "3"),
            "h2d_ms": _point(444.0, "0"),
            "residual_ms": _point(333.0, "1"),
            "forward_ms": _point(222.0, "3"),
            "backward_ms": _point(111.0, "0"),
            "optimizer_ms": _point(99.0, "1"),
            "dataloader_fetch_cpu_ms": _point(88.0, "3"),
        },
        worst={"step_time_ms": _point(101.0, "1")},
        node_ranks=node_ranks,
        metrics_by_rank={
            1: {"step_time_ms": 101.0},
            2: {
                "step_time_ms": 100.0,
                "input_wait_ms": 10.0,
                "traced_step_time_ms": 90.0,
                "compute_ms": 80.0,
                "h2d_ms": 2.0,
                "residual_ms": 8.0,
                "forward_ms": 20.0,
                "backward_ms": 50.0,
                "optimizer_ms": 10.0,
                "dataloader_fetch_cpu_ms": 7.0,
            },
        },
    )
    step_memory = _step_memory_multi_rank(
        diagnosis=_issue("BALANCED", "BALANCED"),
        median={
            "peak_allocated_bytes": _point(
                999.0e9,
                "0" if reserved_points_measured else "3",
            ),
            "peak_reserved_bytes": _point(
                4.0e9 if reserved_points_measured else None,
                "3" if reserved_points_measured else "missing",
            ),
        },
        worst={
            "peak_allocated_bytes": _point(
                888.0e9,
                "2" if reserved_points_measured else "1",
            ),
            "peak_reserved_bytes": _point(
                6.0e9 if reserved_points_measured else None,
                "1" if reserved_points_measured else "missing",
            ),
        },
        node_ranks=node_ranks,
        metrics_by_rank={
            1: {
                "peak_allocated_bytes": 5.0e9,
                "peak_reserved_bytes": 6.0e9,
            },
            3: {
                "peak_allocated_bytes": 3.0e9,
                "peak_reserved_bytes": 4.0e9,
            },
        },
    )
    return {
        "profile": "run",
        "system": system,
        "process": _TWO_NODE_MULTI_PROCESS,
        "step_time": step_time,
        "step_memory": step_memory,
        "meta": _meta(
            run_name="crossed-points",
            mode="multi_node",
            world_size=4,
            nodes_observed=2,
            gpus_observed=4,
        ),
        "duration_s": 5.0,
        "artifact_hint": "logs/crossed-points/final_summary.json",
    }


@pytest.mark.parametrize("name", sorted(CASES))
def test_card_matches_golden(name: str) -> None:
    assert plain(name) == GOLDENS[name]


def test_run_action_sits_between_why_and_panes() -> None:
    """Keep the promoted action beside the verdict that selected it."""
    text = plain("run_input_bound_critical")

    assert text.index("Why:") < text.index("Next:") < text.index("STEP TIMING")


@pytest.mark.parametrize("name", sorted(CASES))
def test_card_lines_are_exactly_card_width(name: str) -> None:
    expected_width = 78 if name.startswith("watch_") else 156
    for line in plain(name).splitlines():
        assert len(line) == expected_width, (name, len(line), line)


@pytest.mark.parametrize(
    "name",
    (
        "run_input_bound_critical",
        "run_multi_healthy",
        "run_multi_input_straggler",
    ),
)
def test_resource_panes_keep_one_fixed_separator_for_all_topologies(
    name: str,
) -> None:
    lines = [line for line in plain(name).splitlines() if "||  " in line]

    assert lines
    assert all(line.count("||") == 1 for line in lines)
    assert {line.index("||") for line in lines} == {64}
    assert "STEP TIMING" in lines[0]
    assert "STEP MEMORY:" in lines[0]
    assert any("SYSTEM METRICS:" in line for line in lines)
    assert any("PROCESS METRICS:" in line for line in lines)


def test_step_memory_pane_keeps_a_blank_row_before_its_table() -> None:
    """Keep the upper-pane heading/table separation visible in plain text."""
    lines = plain("run_multi_input_straggler").splitlines()
    heading_index = next(
        index for index, line in enumerate(lines) if "STEP MEMORY:" in line
    )
    table_index = next(
        index
        for index, line in enumerate(lines)
        if "avg per-step peak" in line
    )
    right_pane_rows = [
        line.split("||  ", 1)[1][:-1].strip()
        for line in lines[heading_index + 1 : table_index]
        if "||  " in line
    ]

    assert right_pane_rows
    assert set(right_pane_rows) == {""}


def test_run_timing_tree_omits_bars_and_keeps_cause_by_share() -> None:
    text = plain("run_input_bound_critical")

    assert "█" not in text
    assert "░" not in text
    assert "├─ Input Wait       128.0 ms   64%  ◀  cause" in text


def test_run_header_reports_system_gpu_observations_without_capping() -> None:
    # System telemetry observed four devices even though this is a one-rank
    # run. The header must describe that provenance, not infer device usage.
    header = plain("run_single_gpu_on_shared_host").splitlines()[2]
    assert "1 rank" in header
    assert "4 GPUs observed" in header


def test_run_header_does_not_infer_gpu_count_without_system_telemetry() -> (
    None
):
    system = _section(
        diagnosis=_issue(
            "NO_DATA",
            "NO DATA",
            summary="System telemetry was not measured.",
        ),
        metadata={"nodes_observed": 1},
        by="node_rank",
    )
    doc = _card(
        profile="run",
        system=system,
        process=_NORMAL_PROCESS,
        step_time=_step_time_single(
            diagnosis=_issue("BALANCED", "BALANCED"),
            steps_analyzed=32,
            step_time_ms=10.0,
            input_wait_ms=1.0,
            traced_step_time_ms=9.0,
            compute_ms=8.0,
            residual_ms=1.0,
        ),
        step_memory=_step_memory_single(1.0e9, 1.2e9),
        meta=_meta(world_size=4, gpus_observed=4),
        duration_s=2.0,
        artifact_hint="logs/missing-system/final_summary.json",
    )
    header = card_to_plain(doc).splitlines()[2]
    assert "1/4 ranks" in header
    assert "GPU" not in header


def test_one_node_ddp_uses_distributed_card_routing() -> None:
    text = plain("run_multi_healthy")
    header = text.splitlines()[2]
    assert "4/4 ranks" in header
    assert "1/1 node" in header
    assert "STEP TIMING (Median R" in text
    assert "Step time breakdown — window average" not in text


def test_distributed_routing_falls_back_to_grouped_rows_without_world_size() -> (
    None
):
    inputs = _coherent_multi_inputs()
    inputs["meta"]["world_size"] = None
    inputs["step_time"]["metadata"]["global_ranks_used"] = None

    text = card_to_plain(_card(**inputs))

    assert "STEP TIMING (Median R2/N1)" in text


def test_header_says_common_steps_only_for_stored_common_alignment() -> None:
    aligned = _coherent_multi_inputs()
    aligned_text = card_to_plain(_card(**aligned))
    assert "40 common steps" in aligned_text

    unaligned = _coherent_multi_inputs()
    unaligned["step_time"]["global"]["window"]["alignment"] = "per_rank"
    unaligned_text = card_to_plain(_card(**unaligned))
    assert "40 steps analyzed" in _logical_text(unaligned_text)
    assert "40 common steps" not in unaligned_text


def test_multi_process_timing_uses_one_median_rank_row() -> None:
    text = card_to_plain(_card(**_coherent_multi_inputs()))

    assert "STEP TIMING (Median R2/N1), GPU Clock" in text
    assert "Step Time           100.0 ms" in text
    assert re.search(r"Input Wait\s+10\.0 ms", text)
    assert "Traced Step Time" not in text
    assert re.search(r"Compute\s+80\.0 ms", text)
    assert re.search(r"├─ Forward\s+20\.0 ms\s+20%", text)
    assert re.search(r"├─ Backward\s+50\.0 ms\s+50%", text)
    assert re.search(r"└─ Optimizer\s+10\.0 ms\s+10%", text)
    assert "DataLoader fetch: 7.0 ms (CPU, supplemental)" in text
    for mixed_median in (
        "777.0 ms",
        "666.0 ms",
        "555.0 ms",
        "444.0 ms",
        "333.0 ms",
        "222.0",
        "111.0",
        "99.0 ms",
        "88.0 ms",
    ):
        assert mixed_median not in text


def test_compute_children_keep_tree_connectors_when_other_phases_are_absent() -> (
    None
):
    step_time = _step_time_single(
        diagnosis=_issue("BALANCED", "BALANCED"),
        steps_analyzed=20,
        step_time_ms=10.0,
        input_wait_ms=1.0,
        traced_step_time_ms=9.0,
        compute_ms=9.0,
        h2d_ms=None,
        residual_ms=None,
        forward_ms=0.0,
        backward_ms=None,
        optimizer_ms=2.0,
    )
    text = _render_single_run(run_name="child-connectors", step_time=step_time)

    assert re.search(r"└─ Compute\s+9\.0 ms\s+90%", text)
    assert re.search(r"   ├─ Forward\s+0\.0 ms\s+0%", text)
    assert re.search(r"   └─ Optimizer\s+2\.0 ms\s+20%", text)
    assert "Backward" not in text


@pytest.mark.parametrize(
    "measured",
    [
        (),
        ("Forward",),
        ("Backward",),
        ("Optimizer",),
        ("Forward", "Backward"),
        ("Forward", "Optimizer"),
        ("Backward", "Optimizer"),
        ("Forward", "Backward", "Optimizer"),
    ],
)
def test_compute_child_combinations_omit_nulls_and_keep_connectors(
    measured: tuple[str, ...],
) -> None:
    values = {"Forward": 1.0, "Backward": 2.0, "Optimizer": 4.0}
    step_time = _step_time_single(
        diagnosis=_issue("BALANCED", "BALANCED"),
        steps_analyzed=20,
        step_time_ms=10.0,
        input_wait_ms=1.0,
        traced_step_time_ms=9.0,
        compute_ms=7.0,
        h2d_ms=1.0,
        residual_ms=1.0,
        forward_ms=values["Forward"] if "Forward" in measured else None,
        backward_ms=(values["Backward"] if "Backward" in measured else None),
        optimizer_ms=(
            values["Optimizer"] if "Optimizer" in measured else None
        ),
    )
    text = _render_single_run(
        run_name="child-combinations", step_time=step_time
    )

    for label, value in values.items():
        if label not in measured:
            assert label not in text
            continue
        index = measured.index(label)
        branch = "└─" if index == len(measured) - 1 else "├─"
        share = int(value * 10)
        assert re.search(
            rf"│  {branch} {label}\s+{value:.1f} ms\s+{share}%",
            text,
        )


def test_compute_bound_why_uses_stored_phase_without_selecting_a_largest() -> (
    None
):
    step_time = _step_time_single(
        diagnosis=_issue(
            "COMPUTE_BOUND",
            "COMPUTE-BOUND",
            summary="Stored summary.",
            action="Stored action.",
            phase="optimizer",
            share_pct=0.9,
        ),
        steps_analyzed=20,
        step_time_ms=100.0,
        input_wait_ms=1.0,
        traced_step_time_ms=99.0,
        compute_ms=90.0,
        h2d_ms=1.0,
        residual_ms=8.0,
        forward_ms=80.0,
        backward_ms=9.0,
        optimizer_ms=1.0,
    )
    text = _render_single_run(
        run_name="stored-compute-phase", step_time=step_time
    )

    assert (
        "Why: Compute took 90% of Step Time; Optimizer was the largest" in text
    )


@pytest.mark.parametrize(
    ("name", "why"),
    [
        (
            "run_multi_input_straggler",
            "R0/N0 waited 254.5 ms for input; R1/N0 waited 3.8 ms for "
            "input.",
        ),
        (
            "run_multi_h2d_straggler",
            "R0/N0 spent 84.0 ms on H2D transfers; R1/N0 spent 4.0 ms "
            "on H2D transfers.",
        ),
        (
            "run_multi_compute_straggler",
            "R0/N0 spent 100.0 ms in Forward; R1/N0 spent 20.0 ms in "
            "Forward.",
        ),
    ],
)
def test_attributed_straggler_why_uses_diagnosed_rank_rows(
    name: str, why: str
) -> None:
    assert why in _logical_text(plain(name))


def test_run_why_falls_back_to_stored_summary_for_legacy_bound_payload() -> (
    None
):
    step_time = _step_time_single(
        diagnosis=_issue(
            "INPUT_BOUND",
            "INPUT-BOUND",
            severity="warn",
            summary="Legacy input summary stays unchanged.",
            action="Stored action.",
        ),
        steps_analyzed=20,
        step_time_ms=10.0,
        input_wait_ms=5.0,
        traced_step_time_ms=5.0,
        compute_ms=4.0,
        residual_ms=1.0,
    )
    text = _render_single_run(run_name="legacy-bound", step_time=step_time)

    assert "Why: Legacy input summary stays unchanged." in text
    assert "Next: Stored action." in text


def test_run_why_falls_back_when_diagnosed_straggler_row_is_missing() -> None:
    inputs = _coherent_multi_inputs()
    diagnosis = _issue(
        "INPUT_STRAGGLER",
        "INPUT STRAGGLER",
        severity="warn",
        summary="Legacy straggler summary stays unchanged.",
        action="Stored action.",
        phase="input",
        evidence={"culprit_rank": 9, "victim_rank": 1},
    )
    inputs["step_time"]["diagnosis"] = diagnosis
    inputs["step_time"]["issues"] = [diagnosis]

    text = card_to_plain(_card(**inputs))

    assert "Why: Legacy straggler summary stays unchanged." in text


def test_multi_process_step_memory_uses_reserved_selected_rows() -> None:
    text = card_to_plain(_card(**_coherent_multi_inputs()))

    assert (
        "Allocated                   3.0 GB              5.0 GB, R1/N0" in text
    )
    assert (
        "Reserved                    4.0 GB              6.0 GB, R1/N0" in text
    )
    assert "999.0 GB" not in text
    assert "888.0 GB" not in text


def test_multi_process_step_memory_falls_back_to_allocated_selectors() -> None:
    text = card_to_plain(
        _card(**_coherent_multi_inputs(reserved_points_measured=False))
    )

    assert (
        "Allocated                   3.0 GB              5.0 GB, R1/N0" in text
    )
    assert (
        "Reserved                    4.0 GB              6.0 GB, R1/N0" in text
    )
    assert "999.0 GB" not in text
    assert "888.0 GB" not in text


def test_multi_process_step_memory_keeps_metric_rows_when_points_share_a_rank() -> (
    None
):
    text = plain("run_multi_duplicate_memory_pair")

    assert (
        "Allocated                   3.0 GB              3.0 GB, R3/N1" in text
    )
    assert (
        "Reserved                    4.0 GB              4.0 GB, R3/N1" in text
    )


def test_multi_process_step_memory_preserves_selected_zero_values() -> None:
    inputs = _coherent_multi_inputs()
    step_memory = inputs["step_memory"]
    for block, rank in (("median", "3"), ("worst", "1")):
        for metric in ("peak_allocated_bytes", "peak_reserved_bytes"):
            step_memory["global"][block][metric] = _point(0.0, rank)
    for rank in ("1", "3"):
        step_memory["groups"]["rows"][rank]["metrics"] = {
            "peak_allocated_bytes": 0.0,
            "peak_reserved_bytes": 0.0,
        }

    text = card_to_plain(_card(**inputs))

    assert (
        "Allocated                   0.0 MB              0.0 MB, R1/N0" in text
    )
    assert (
        "Reserved                    0.0 MB              0.0 MB, R1/N0" in text
    )


def test_multi_process_step_memory_omits_missing_selected_rows() -> None:
    inputs = _coherent_multi_inputs()
    step_memory = inputs["step_memory"]
    for block in ("median", "worst"):
        for metric in ("peak_allocated_bytes", "peak_reserved_bytes"):
            step_memory["global"][block][metric]["idx"] = "missing"

    text = card_to_plain(_card(**inputs))

    assert "STEP MEMORY: BALANCED" in text
    assert "avg per-step peak" not in text
    assert "999.0 GB" not in text
    assert "888.0 GB" not in text


def test_rendering_does_not_mutate_summary_inputs() -> None:
    inputs = _coherent_multi_inputs()
    before = copy.deepcopy(inputs)

    _card(**inputs)

    assert inputs == before


def test_partial_rank_payload_reports_missing_coherent_rows() -> None:
    text = plain("run_partial_rank_payload")

    assert "STEP TIMING unavailable: selected rank row missing." in text
    assert (
        "Allocated                                       5.0 GB, R1/N0" in text
    )
    assert (
        "Reserved                                        6.0 GB, R1/N0" in text
    )
    assert "999.0 GB" not in text
    assert "888.0 GB" not in text


def test_resource_blocks_preserve_measured_zero_and_omit_null() -> None:
    text = plain("run_measured_zero_and_null")

    assert "CPU                    0%" in text
    assert "RAM used               0.0 MB (0%)" in text
    assert "GPU util               0%" in text
    assert "GPU temperature        0C" in text
    assert "CPU capacity         0%" in text
    assert "RSS used             0.0 MB (0%)" in text
    assert "Allocated                   0.0 MB" in text
    assert "Reserved                    0.0 MB" in text
    assert "GPU memory" not in text
    assert "CUDA" not in text
    assert "n/a" not in text


def test_one_node_ddp_uses_the_system_average_table() -> None:
    text = plain("run_multi_healthy")

    assert "SYSTEM METRICS: NORMAL" in text
    assert (
        "|                         avg                                   ||"
        in text
    )
    assert "median node avg" not in text
    assert "GPU power              240W" in text


def test_system_table_falls_back_to_grouped_node_rows() -> None:
    inputs = _coherent_multi_inputs()
    inputs["system"]["metadata"].pop("nodes_observed")
    inputs["system"]["metadata"].pop("nodes_expected")

    text = card_to_plain(_card(**inputs))

    assert "SYSTEM METRICS: NORMAL · 2 nodes" in text
    assert "median node avg   worst node avg" in text


def test_multi_node_system_table_uses_stored_points_and_coherent_pairs() -> (
    None
):
    text = plain("run_multi_high_power_partial")

    assert "SYSTEM METRICS: HIGH GPU PWR  (WARNING) · 2/4 nodes" in text
    assert "median node avg   worst node avg" in text
    assert "CPU                    10%               30%, N0" in text
    assert "RAM used               6.0 GB (20%)      10.0 GB (30%), N1" in text
    assert "GPU memory/device      2.0 GB (12%)      6.0 GB (38%), N0" in text
    assert "GPU power              220W              280W, N0" in text
    system_block = text[
        text.index("SYSTEM METRICS:") : text.index("STEP MEMORY:")
    ]
    assert "999.0 GB" not in system_block
    assert "888.0 GB" not in system_block
    assert "777.0 GB" not in text
    assert "666.0 GB" not in text


def test_high_power_table_keeps_stored_diagnosis_evidence() -> None:
    text = plain("run_multi_high_power_partial")

    assert "GPU power              220W              280W, N0" in text
    assert "Evidence: GPU power 83.7% of limit · N0/G0" in text


def test_single_rank_process_warning_block_matches_exact_layout() -> None:
    diagnosis = _issue(
        "HIGH_PROCESS_RSS",
        "HIGH PROCESS RSS",
        severity="warn",
        summary="Stored fallback.",
        evidence={"ram_peak_percent": 91.2, "highest_rss_rank": 0},
    )

    text = _render_process_case(
        _process_with_diagnosis(diagnosis, multi_rank=False),
        world_size=1,
    )

    assert (
        _process_block(text)
        == """\
PROCESS METRICS: HIGH PROCESS RSS  (WARNING)
Evidence: RSS peak 91.2% · R0/N0
                     avg
CPU capacity         14%
RSS used             3.1 GB (10%)
CUDA used            2.9 GB
CUDA reserved        3.2 GB (20%)"""
    )


def test_multi_rank_process_warning_block_matches_exact_layout() -> None:
    diagnosis = _issue(
        "RANK_GPU_MEMORY_IMBALANCE",
        "RANK GPU MEMORY IMBALANCE",
        severity="warn",
        summary="Stored fallback.",
        metric="rank_gpu_reserved_imbalance_percent",
        evidence={
            "rank_gpu_reserved_imbalance_percent": 54.4,
            "rank_gpu_memory_pressure_percent": 92.3,
            "rank_gpu_memory_pressure_rank": 3,
        },
    )

    text = _render_process_case(
        _process_with_diagnosis(diagnosis, multi_rank=True),
        world_size=4,
    )

    assert (
        _process_block(text)
        == """\
PROCESS METRICS: RANK GPU MEMORY IMBALANCE  (WARNING) · 4/4 ranks
Evidence: CUDA reserved imbalance 54.4% · R3/N0
                     median rank avg   worst rank avg
CPU capacity         12%               81%, R2/N0
RSS used             3.1 GB (10%)      5.4 GB (17%), R1/N0
CUDA used            2.9 GB            4.6 GB, R3/N0
CUDA reserved        3.2 GB (20%)      6.8 GB (43%), R3/N0"""
    )


@pytest.mark.parametrize(
    ("multi_rank", "diagnosis", "expected"),
    [
        (
            False,
            _issue(
                "HIGH_PROCESS_CPU",
                "HIGH PROCESS CPU",
                severity="warn",
                summary="Stored fallback.",
                evidence={"cpu_capacity_percent": 88.5},
            ),
            "CPU capacity 88.5%",
        ),
        (
            False,
            _issue(
                "HIGH_PROCESS_RSS",
                "HIGH PROCESS RSS",
                severity="warn",
                summary="Stored fallback.",
                evidence={"ram_peak_percent": 91.2},
            ),
            "RSS peak 91.2%",
        ),
        (
            True,
            _issue(
                "HIGH_PROCESS_RSS",
                "HIGH PROCESS RSS",
                severity="warn",
                summary="Stored fallback.",
                evidence={
                    "ram_peak_percent": 91.2,
                    "highest_rss_rank": 1,
                },
            ),
            "RSS peak 91.2% · R1/N0",
        ),
        (
            False,
            _issue(
                "HIGH_PROCESS_GPU_MEMORY",
                "HIGH PROCESS GPU MEMORY",
                severity="warn",
                summary="Stored fallback.",
                metric="gpu_mem_used_peak_percent",
                evidence={"gpu_mem_used_peak_percent": 82.5},
            ),
            "CUDA used peak 82.5%",
        ),
        (
            False,
            _issue(
                "VERY_HIGH_PROCESS_GPU_MEMORY",
                "VERY HIGH PROCESS GPU MEMORY",
                severity="crit",
                summary="Stored fallback.",
                metric="gpu_mem_reserved_peak_percent",
                evidence={"gpu_mem_reserved_peak_percent": 96.1},
            ),
            "CUDA reserved peak 96.1%",
        ),
        (
            True,
            _issue(
                "HIGH_PROCESS_GPU_MEMORY",
                "HIGH PROCESS GPU MEMORY",
                severity="warn",
                summary="Stored fallback.",
                metric="gpu_mem_reserved_peak_percent",
                evidence={
                    "gpu_mem_reserved_peak_percent": 89.4,
                    "rank": 3,
                },
            ),
            "CUDA reserved peak 89.4% · R3/N0",
        ),
        (
            True,
            _issue(
                "VERY_HIGH_PROCESS_GPU_MEMORY",
                "VERY HIGH PROCESS GPU MEMORY",
                severity="crit",
                summary="Stored fallback.",
                metric="gpu_mem_used_peak_percent",
                evidence={
                    "gpu_mem_used_peak_percent": 97.2,
                    "rank": 2,
                },
            ),
            "CUDA used peak 97.2% · R2/N0",
        ),
        (
            True,
            _issue(
                "GPU_MEMORY_RESERVED_OVERHANG",
                "HIGH CUDA ALLOCATOR RESERVED/ALLOCATED RATIO",
                severity="warn",
                summary="Stored fallback.",
                evidence={
                    "gpu_mem_reserved_overhang_ratio": 3.25,
                    "gpu_mem_reserved_peak_percent": 88.0,
                    "highest_overhang_rank": 2,
                },
            ),
            "CUDA reserved/allocated 3.25x · R2/N0",
        ),
        (
            True,
            _issue(
                "RANK_GPU_MEMORY_IMBALANCE",
                "RANK GPU MEMORY IMBALANCE",
                severity="warn",
                summary="Stored fallback.",
                metric="rank_gpu_reserved_imbalance_percent",
                evidence={
                    "rank_gpu_reserved_imbalance_percent": 54.4,
                    "rank_gpu_memory_pressure_percent": 92.3,
                    "rank_gpu_memory_pressure_rank": 3,
                },
            ),
            "CUDA reserved imbalance 54.4% · R3/N0",
        ),
        (
            True,
            _issue(
                "RANK_GPU_MEMORY_IMBALANCE",
                "RANK GPU MEMORY IMBALANCE",
                severity="warn",
                summary="Stored fallback.",
                metric="rank_gpu_used_imbalance_percent",
                evidence={
                    "rank_gpu_used_imbalance_percent": 44.4,
                    "rank_gpu_memory_pressure_percent": 82.3,
                    "rank_gpu_memory_pressure_rank": 1,
                },
            ),
            "CUDA used imbalance 44.4% · R1/N0",
        ),
    ],
)
def test_process_evidence_uses_compact_stored_fields(
    multi_rank: bool,
    diagnosis: Dict[str, Any],
    expected: str,
) -> None:
    text = _render_process_case(
        _process_with_diagnosis(diagnosis, multi_rank=multi_rank),
        world_size=4 if multi_rank else 1,
    )

    assert f"Evidence: {expected}" in _process_block(text)


def test_process_evidence_falls_back_to_stored_summary() -> None:
    diagnosis = _issue(
        "HIGH_PROCESS_RSS",
        "HIGH PROCESS RSS",
        severity="warn",
        summary="Legacy Process summary stays unchanged.",
        evidence={},
    )

    text = _render_process_case(
        _process_with_diagnosis(diagnosis, multi_rank=True),
        world_size=4,
    )

    assert "Evidence: Legacy Process summary stays unchanged." in text


def test_process_normal_omits_evidence_and_no_data_omits_empty_table() -> None:
    normal = _render_process_case(
        copy.deepcopy(_NORMAL_PROCESS),
        world_size=1,
    )
    no_data_diagnosis = _issue(
        "NO_DATA",
        "NO DATA",
        summary="Process telemetry was not measured.",
    )
    no_data = _render_process_case(
        _section(diagnosis=no_data_diagnosis),
        world_size=1,
    )

    assert "Evidence:" not in _process_block(normal)
    assert (
        _process_block(no_data)
        == """\
PROCESS METRICS: NO DATA · 0/1 rank
Evidence: Process telemetry was not measured."""
    )


def test_process_table_uses_observed_rank_count_and_reports_coverage() -> None:
    partial = _render_process_case(
        copy.deepcopy(_NORMAL_PROCESS),
        world_size=4,
    )
    no_expected = copy.deepcopy(_ONE_NODE_MULTI_PROCESS)
    no_expected["metadata"].clear()
    observed = _render_process_case(no_expected, world_size=None)

    assert "PROCESS METRICS: NORMAL · 1/4 ranks" in partial
    assert "                     avg" in _process_block(partial)
    assert "median rank avg" not in _process_block(partial)
    assert "PROCESS METRICS: NORMAL · 4 ranks observed" in observed
    assert "median rank avg" in _process_block(observed)


def test_multi_rank_process_table_uses_coherent_rows_and_metric_scopes() -> (
    None
):
    text = plain("run_multi_healthy")

    assert "CPU capacity         12%               81%, R2/N0" in text
    assert "RSS used             3.1 GB (10%)      5.4 GB (17%), R1/N0" in text
    assert "CUDA used            2.9 GB            4.6 GB, R3/N0" in text
    assert "CUDA reserved        3.2 GB (20%)      6.8 GB (43%), R3/N0" in text
    for crossed_value in ("999.0 GB", "888.0 GB", "777.0 GB", "666.0 GB"):
        assert crossed_value not in text


def test_process_primary_is_not_repeated_but_later_issue_is_retained() -> None:
    primary = _issue(
        "HIGH_PROCESS_CPU",
        "HIGH PROCESS CPU",
        severity="warn",
        summary="Primary Process summary.",
        evidence={"cpu_capacity_percent": 88.5},
    )
    later = _issue(
        "HIGH_PROCESS_RSS",
        "HIGH PROCESS RSS",
        severity="warn",
        summary="Later Process RSS finding.",
        evidence={"ram_peak_percent": 91.2, "highest_rss_rank": 1},
    )
    process = copy.deepcopy(_ONE_NODE_MULTI_PROCESS)
    process["diagnosis"] = primary
    process["issues"] = [primary, later]

    text = _render_process_case(process, world_size=4)

    assert text.count("Evidence: CPU capacity 88.5%") == 1
    assert text.count("Later Process RSS finding.") == 1


def test_non_normal_resource_blocks_show_stored_trigger_and_scope() -> None:
    inputs = _coherent_multi_inputs()
    system_diagnosis = _issue(
        "POWER_HIGH",
        "HIGH POWER",
        severity="warn",
        summary="GPU power averaged 83.7% of limit.",
        evidence={"scope": {"level": "gpu", "node_rank": 0, "gpu_idx": 2}},
    )
    inputs["system"]["diagnosis"] = system_diagnosis
    inputs["system"]["issues"] = [system_diagnosis]

    process_diagnosis = _issue(
        "HIGH_PROCESS_CPU",
        "HIGH PROCESS CPU",
        severity="warn",
        summary="Process CPU capacity averaged 390%.",
        evidence={"cpu_capacity_percent": 390.0},
    )
    process = copy.deepcopy(_TWO_NODE_MULTI_PROCESS)
    process["diagnosis"] = process_diagnosis
    process["issues"] = [process_diagnosis]
    inputs["process"] = process

    memory_diagnosis = _issue(
        "CREEP_CONFIRMED",
        "MEMORY CREEP",
        severity="warn",
        summary="Peak reserved memory is rising across the window.",
        metric="peak_reserved",
        ranks=[1],
        evidence={
            "overall_abs_delta_bytes": 1.2e9,
            "overall_worst_growth_pct": 0.18,
        },
    )
    inputs["step_memory"]["diagnosis"] = memory_diagnosis
    inputs["step_memory"]["issues"] = [memory_diagnosis]

    text = card_to_plain(_card(**inputs))

    assert "GPU power averaged 83.7% of limit. · N0/G2" in text
    assert "Evidence: CPU capacity 390.0%" in text
    assert "Evidence: Memory creep +1.2 GB (18.0%) · R1/N0" in text


@pytest.mark.parametrize(
    "diagnosis, expected",
    [
        (
            _issue(
                "HIGH_PRESSURE",
                "HIGH PRESSURE",
                severity="crit",
                metric="peak_reserved",
                ranks=[2],
                evidence={"pressure_frac": 0.923},
            ),
            "Evidence: CUDA reserved pressure 92.3% · R2/N1",
        ),
        (
            _issue(
                "IMBALANCE",
                "IMBALANCE",
                severity="warn",
                metric="peak_reserved",
                ranks=[2],
                evidence={"skew_pct": 0.412, "pressure_frac": 0.923},
            ),
            "Evidence: CUDA reserved skew 41.2% · pressure 92.3% · R2/N1",
        ),
        (
            _issue(
                "CREEP_EARLY",
                "MEMORY RISING",
                metric="peak_allocated",
                ranks=[1],
                evidence={
                    "overall_abs_delta_bytes": 50.0e6,
                    "overall_worst_growth_pct": 0.075,
                },
            ),
            "Evidence: Memory rising +50.0 MB (7.5%) · R1/N0",
        ),
        (
            _issue(
                "NO_GPU",
                "NO GPU",
                summary=(
                    "No GPU detected. Step memory uses torch-based GPU memory "
                    "telemetry."
                ),
            ),
            "Evidence: Step memory uses torch-based GPU memory telemetry.",
        ),
    ],
)
def test_step_memory_evidence_uses_compact_stored_fields(
    diagnosis: Dict[str, Any], expected: str
) -> None:
    inputs = _coherent_multi_inputs()
    inputs["step_memory"]["diagnosis"] = diagnosis
    inputs["step_memory"]["issues"] = [diagnosis]

    text = card_to_plain(_card(**inputs))

    assert expected in text


@pytest.mark.parametrize(
    "diagnosis",
    [
        _issue(
            "HIGH_PRESSURE",
            "HIGH PRESSURE",
            severity="warn",
            summary="Legacy pressure summary stays unchanged.",
            metric="peak_reserved",
        ),
        _issue(
            "IMBALANCE",
            "IMBALANCE",
            severity="warn",
            summary="Legacy imbalance summary stays unchanged.",
            metric="peak_reserved",
            evidence={"pressure_frac": 0.923},
        ),
        _issue(
            "CREEP_CONFIRMED",
            "MEMORY CREEP",
            severity="warn",
            summary="Legacy creep summary stays unchanged.",
            metric="peak_reserved",
            ranks=[2],
        ),
    ],
)
def test_step_memory_evidence_falls_back_to_stored_summary(
    diagnosis: Dict[str, Any],
) -> None:
    inputs = _coherent_multi_inputs()
    inputs["step_memory"]["diagnosis"] = diagnosis
    inputs["step_memory"]["issues"] = [diagnosis]

    text = card_to_plain(_card(**inputs))

    assert f"Evidence: {diagnosis['summary']}" in text


def test_resource_panes_wrap_without_moving_or_clipping_divider() -> None:
    inputs = _coherent_multi_inputs()
    system_diagnosis = _issue(
        "HIGH_GPU_POWER",
        "HIGH GPU PWR",
        severity="warn",
        summary=(
            "GPU power evidence contains a deliberately long explanation "
            "that must remain visible inside the System pane."
        ),
        evidence={"scope": {"level": "gpu", "node_rank": 1, "gpu_idx": 7}},
    )
    inputs["system"]["diagnosis"] = system_diagnosis
    inputs["system"]["issues"] = [system_diagnosis]

    process_diagnosis = _issue(
        "HIGH_PROCESS_CPU",
        "HIGH PROCESS CPU",
        severity="warn",
        summary=(
            "Process evidence contains a deliberately long explanation "
            "that must remain visible inside the Process pane."
        ),
        ranks=[3],
    )
    inputs["process"]["diagnosis"] = process_diagnosis
    inputs["process"]["issues"] = [process_diagnosis]

    text = card_to_plain(_card(**inputs))
    lines = text.splitlines()
    start = next(
        i for i, line in enumerate(lines) if "SYSTEM METRICS:" in line
    )
    end = next(
        i for i in range(start, len(lines)) if "Full evidence:" in lines[i]
    )
    pane_lines = [line for line in lines[start:end] if "||" in line]

    assert all("||  " in line for line in pane_lines)
    assert {line.index("||") for line in pane_lines} == {64}
    assert all(len(line) == 156 for line in pane_lines)
    logical = " ".join(pane_lines)
    right_logical = " ".join(
        line.split("||  ", 1)[1][:-1].strip() for line in pane_lines
    )
    assert "must remain visible inside the System pane." in logical
    assert "must remain visible inside the Process pane." in right_logical
    assert "N1/G7" in text
    assert "R3/N1" in text
    assert text.index("Evidence: ") < text.index("median node avg")


def test_duplicate_diagnosis_ranks_render_once() -> None:
    inputs = _coherent_multi_inputs()
    diagnosis = _issue(
        "MEMORY_CREEP",
        "MEMORY CREEP",
        severity="warn",
        summary="Duplicate-rank trigger.",
        ranks=[3, 3, "3"],
    )
    inputs["step_memory"]["diagnosis"] = diagnosis
    inputs["step_memory"]["issues"] = [diagnosis]

    logical = _logical_text(card_to_plain(_card(**inputs)))

    assert "Duplicate-rank trigger. · R3/N1" in logical
    assert "R3/N1, R3/N1" not in logical


@pytest.mark.parametrize(
    "name",
    [
        "run_multi_input_straggler",
        "run_multi_h2d_straggler",
        "run_multi_compute_straggler",
        "run_multi_generic_straggler",
        "run_multi_fsdp_straggler",
    ],
)
def test_straggler_card_has_no_aggregate_comparison_or_cause_marker(
    name: str,
) -> None:
    text = plain(name)

    assert "Input comparison:" not in text
    assert "H2D comparison:" not in text
    assert "Forward comparison:" not in text
    assert "x median" not in text
    assert "◀" not in text


def test_unmeasured_resource_sections_show_their_stored_reason() -> None:
    text = plain("run_unmeasured_resource_sections")

    assert "SYSTEM METRICS: NO DATA" in text
    assert "System sampler produced no rows." in text
    assert "PROCESS METRICS: NO DATA" in text
    assert "Process telemetry was not measured." in text
    assert "STEP MEMORY: NO GPU" in text
    assert "Step memory uses torch-based GPU memory telemetry." in text


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


def test_other_findings_skip_section_primaries_and_keep_later_issues() -> None:
    # Step Time already owns the primary verdict and timing evidence, so the
    # bounded secondary block keeps the later resource finding instead.
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
    fragmentation = _issue(
        "ALLOCATOR_FRAGMENTATION",
        "ALLOCATOR FRAGMENTATION",
        severity="warn",
        summary="Allocator reserved memory is 2.4x live tensor memory.",
        action="Inspect retained blocks and allocation sizes.",
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
    step_memory = _step_memory_single(3.9e9, 4.6e9, diagnosis=creep)
    step_memory["issues"] = [creep, fragmentation]

    text = _render_single_run(
        run_name="ddp_balanced",
        step_time=step_time,
        system=_system_single(gpu_util_percent=88.0),
        step_memory=step_memory,
        duration_s=2.7,
    )

    assert text.count("Step memory grew 1.2 GB over the run") == 1
    assert text.count("Allocator reserved memory is 2.4x") == 1
    assert "Input wait is 11.0%" not in text


def test_step_memory_falls_back_to_megabytes() -> None:
    # A sub-0.1 GB footprint must not render as "0.0 GB".
    text = plain("run_single_gpu_on_shared_host")
    assert "Allocated                   18.8 MB" in text
    assert "Reserved                    23.1 MB" in text
    assert "0.0 GB" not in text
    multi = plain("run_multi_residual_heavy")
    assert (
        "Allocated                   18.8 MB             21.0 MB, R0/N0"
        in multi
    )
    assert (
        "Reserved                    23.1 MB             25.1 MB, R0/N0"
        in multi
    )


def test_step_memory_keeps_gigabytes_above_the_threshold() -> None:
    text = plain("run_healthy")
    assert "Allocated                   2.9 GB" in text
    assert "Reserved                    3.2 GB" in text


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


def _spans_for_line(doc: CardDoc, text: str):
    """Return the styled spans in the first card row containing ``text``."""
    for line in doc.lines:
        if text in "".join(span.text for span in line.spans):
            return line.spans
    raise AssertionError(f"No card row contains {text!r}")


def _resource_pane_spans(
    spans: Sequence[Span], *, right: bool
) -> Tuple[Span, ...]:
    """Return meaningful spans from one side of the resource divider."""
    separator = next(
        index for index, span in enumerate(spans) if span.text == "||  "
    )
    selected = list(spans[separator + 1 :] if right else spans[:separator])
    while selected and not selected[-1].text.strip():
        selected.pop()
    return tuple(selected)


def test_process_table_style_contract() -> None:
    diagnosis = _issue(
        "HIGH_PROCESS_CPU",
        "HIGH PROCESS CPU",
        severity="warn",
        summary="Stored fallback.",
        evidence={"cpu_capacity_percent": 88.5},
    )
    doc = _process_case_doc(
        _process_with_diagnosis(diagnosis, multi_rank=True),
        world_size=4,
    )

    status = _resource_pane_spans(
        _spans_for_line(doc, "PROCESS METRICS:"), right=True
    )
    assert [(span.text, span.style) for span in status] == [
        ("PROCESS METRICS: ", STYLE_BOLD),
        ("HIGH PROCESS CPU  (WARNING)", STYLE_WARN),
        (" · 4/4 ranks", STYLE_DIM),
    ]

    evidence = _resource_pane_spans(
        _spans_for_line(doc, "Evidence: CPU capacity 88.5%"), right=True
    )
    assert [(span.text, span.style) for span in evidence] == [
        ("Evidence: ", STYLE_BOLD),
        ("CPU capacity 88.5%", STYLE_PLAIN),
    ]

    header = _resource_pane_spans(
        _spans_for_line(doc, "median rank avg"), right=True
    )
    assert all(span.style == STYLE_DIM for span in header)

    measurement = _resource_pane_spans(
        _spans_for_line(doc, "CPU capacity         12%"), right=True
    )
    assert all(span.style == STYLE_PLAIN for span in measurement)


def test_terminal_style_contract_keeps_evidence_neutral() -> None:
    doc = CASES["run_input_bound_critical"]()

    verdict = _spans_for_line(doc, "Verdict:")
    assert [(span.text, span.style) for span in verdict] == [
        ("Verdict: ", STYLE_BOLD),
        ("INPUT-BOUND  (CRITICAL)", STYLE_CRIT),
    ]

    culprit = _spans_for_line(doc, "├─ Input Wait")
    assert culprit[0].style == STYLE_PLAIN
    assert culprit[1] == Span("◀  cause", STYLE_CRIT)

    next_line = _spans_for_line(doc, "Next:")
    assert next_line[0] == Span("Next: ", STYLE_BOLD)
    assert next_line[1].style == STYLE_NEXT


def test_terminal_style_contract_colours_primary_status_values() -> None:
    system = _resource_pane_spans(
        _spans_for_line(CASES["run_healthy"](), "SYSTEM METRICS:"),
        right=False,
    )
    assert [(span.text, span.style) for span in system[:2]] == [
        ("SYSTEM METRICS: ", STYLE_BOLD),
        ("NORMAL", STYLE_OK),
    ]
    assert all(span.style == STYLE_PLAIN for span in system[2:])

    process = _resource_pane_spans(
        _spans_for_line(CASES["run_healthy"](), "PROCESS METRICS:"),
        right=True,
    )
    assert [(span.text, span.style) for span in process[:2]] == [
        ("PROCESS METRICS: ", STYLE_BOLD),
        ("NORMAL", STYLE_OK),
    ]
    assert all(span.style == STYLE_PLAIN for span in process[2:])

    step_memory = _resource_pane_spans(
        _spans_for_line(
            CASES["run_input_bound_with_also"](),
            "STEP MEMORY:",
        ),
        right=True,
    )
    assert [(span.text, span.style) for span in step_memory[:2]] == [
        ("STEP MEMORY: ", STYLE_BOLD),
        ("MEMORY CREEP  (WARNING)", STYLE_WARN),
    ]
    assert all(span.style == STYLE_PLAIN for span in step_memory[2:])

    secondary = _spans_for_line(
        CASES["run_input_bound_with_also"](),
        "Allocator reserved memory",
    )
    assert [(span.text, span.style) for span in secondary] == [
        (
            "! Allocator reserved memory is 2.4x live tensor memory.  (",
            STYLE_PLAIN,
        ),
        ("WARNING", STYLE_WARN),
        (")", STYLE_PLAIN),
    ]

    normal = _spans_for_line(CASES["watch_healthy"](), "Host health:")
    assert [(span.text, span.style) for span in normal] == [
        ("Host health: ", STYLE_BOLD),
        ("NORMAL", STYLE_OK),
    ]

    warning = _spans_for_line(
        CASES["watch_memory_pressure"](),
        "Host health:",
    )
    assert [(span.text, span.style) for span in warning] == [
        ("Host health: ", STYLE_BOLD),
        ("MEMORY PRESSURE  (WARNING)", STYLE_WARN),
    ]

    multi_normal = _spans_for_line(
        CASES["watch_multi_node"](),
        "Host health:",
    )
    assert [(span.text, span.style) for span in multi_normal] == [
        ("Host health: ", STYLE_BOLD),
        ("NORMAL on all 3 nodes", STYLE_OK),
    ]


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
def test_single_machine_cards_avoid_multi_rank_vocabulary(
    name: str,
) -> None:
    text = plain(name)
    if name.startswith("run_"):
        assert "1 rank" in text.splitlines()[2]
        # Why/Next now preserve the stored primary-diagnosis wording, which
        # may explicitly say that rank skew was not identified.
        banned_words = ("node",)
    else:
        banned_words = ("rank", "node", "skew")
    for banned in banned_words:
        assert not re.search(rf"\b{banned}", text, re.IGNORECASE), banned
