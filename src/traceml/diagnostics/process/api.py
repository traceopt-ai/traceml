"""
Summary-oriented process diagnosis API.

Process diagnosis is intentionally conservative and primarily aimed at final
summary interpretation and JSON output. It is not used to change live runtime
rendering behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..common import BaseDiagnosis, DiagnosticResult, Severity
from .context import ProcessSummarySignals, build_process_summary_signals
from .rules import run_process_rules


@dataclass(frozen=True)
class ProcessDiagnosis(BaseDiagnosis):
    """
    Primary process diagnosis used in final-summary cards and JSON.
    """

    kind: str
    samples_used: int


def _mk_diag(
    *,
    kind: str,
    severity: Severity,
    status: str,
    reason: str,
    action: str,
    samples_used: int,
) -> ProcessDiagnosis:
    return ProcessDiagnosis(
        kind=str(kind),
        severity=severity,
        status=str(status),
        reason=str(reason),
        action=str(action),
        samples_used=int(samples_used),
    )


def _default_primary(signals: ProcessSummarySignals) -> ProcessDiagnosis:
    """Build the default non-issue process diagnosis."""
    if signals.samples <= 0:
        return _mk_diag(
            kind="NO_DATA",
            severity="info",
            status="NO DATA",
            reason="No traced process telemetry was recorded.",
            action="Collect process telemetry for workload-local context.",
            samples_used=signals.samples,
        )

    has_gpu_memory = any(
        value is not None
        for value in (
            signals.gpu_mem_used_peak_percent,
            signals.gpu_mem_reserved_peak_percent,
        )
    )
    reason = (
        "Process CPU, RSS, and GPU memory showed no pressure."
        if has_gpu_memory
        else "Process CPU and RSS showed no pressure."
    )
    return _mk_diag(
        kind="NORMAL",
        severity="info",
        status="NORMAL",
        reason=reason,
        action="Use training diagnostics for model-level bottlenecks.",
        samples_used=signals.samples,
    )


def build_process_diagnosis_result(
    *,
    duration_s: Optional[float],
    process_samples: int,
    distinct_ranks: int,
    distinct_pids: int,
    cpu_avg_percent: Optional[float],
    cpu_peak_percent: Optional[float],
    cpu_logical_core_count: Optional[int],
    ram_avg_bytes: Optional[float],
    ram_peak_bytes: Optional[float],
    ram_total_bytes: Optional[float],
    gpu_available: Optional[bool],
    gpu_count: Optional[int],
    gpu_device_index: Optional[int],
    gpu_mem_used_avg_bytes: Optional[float],
    gpu_mem_used_peak_bytes: Optional[float],
    gpu_mem_reserved_avg_bytes: Optional[float],
    gpu_mem_reserved_peak_bytes: Optional[float],
    gpu_mem_total_bytes: Optional[float],
    per_rank: Dict[int, Dict[str, Optional[float]]],
) -> DiagnosticResult[ProcessDiagnosis]:
    """
    Build one rich process diagnosis result from summary-level process signals.
    """
    signals = build_process_summary_signals(
        duration_s=duration_s,
        samples=process_samples,
        distinct_ranks=distinct_ranks,
        distinct_pids=distinct_pids,
        cpu_avg_percent=cpu_avg_percent,
        cpu_peak_percent=cpu_peak_percent,
        cpu_logical_core_count=cpu_logical_core_count,
        ram_avg_bytes=ram_avg_bytes,
        ram_peak_bytes=ram_peak_bytes,
        ram_total_bytes=ram_total_bytes,
        gpu_available=gpu_available,
        gpu_count=gpu_count,
        gpu_device_index=gpu_device_index,
        gpu_mem_used_avg_bytes=gpu_mem_used_avg_bytes,
        gpu_mem_used_peak_bytes=gpu_mem_used_peak_bytes,
        gpu_mem_reserved_avg_bytes=gpu_mem_reserved_avg_bytes,
        gpu_mem_reserved_peak_bytes=gpu_mem_reserved_peak_bytes,
        gpu_mem_total_bytes=gpu_mem_total_bytes,
        per_rank=per_rank,
    )

    issues = run_process_rules(signals) if signals.samples > 0 else ()
    if issues:
        primary_issue = issues[0]
        primary = _mk_diag(
            kind=primary_issue.kind,
            severity=primary_issue.severity,
            status=primary_issue.status,
            reason=primary_issue.summary,
            action=primary_issue.action,
            samples_used=signals.samples,
        )
    else:
        primary = _default_primary(signals)

    metric_attribution: Dict[str, Any] = {
        "scope": {
            "ranks": signals.distinct_ranks,
            "pids": signals.distinct_pids,
        },
        "cpu": {
            "avg_percent": signals.cpu_avg_percent,
            "peak_percent": signals.cpu_peak_percent,
            "logical_core_count": signals.cpu_logical_core_count,
            "pressure_frac": signals.cpu_pressure_frac,
            "capacity_percent": signals.cpu_capacity_percent,
        },
        "ram": {
            "avg_bytes": signals.ram_avg_bytes,
            "peak_bytes": signals.ram_peak_bytes,
            "total_bytes": signals.ram_total_bytes,
            "pressure_frac": signals.ram_pressure_frac,
            "peak_percent": signals.ram_peak_percent,
            "highest_rss_rank": signals.highest_rss_rank,
            "rank_imbalance_pct": signals.rank_rss_imbalance_pct,
        },
        "gpu_rollup": {
            "available": signals.gpu_available,
            "count": signals.gpu_count,
            "device_index": signals.gpu_device_index,
            "used_avg_bytes": signals.gpu_mem_used_avg_bytes,
            "used_peak_bytes": signals.gpu_mem_used_peak_bytes,
            "reserved_avg_bytes": signals.gpu_mem_reserved_avg_bytes,
            "reserved_peak_bytes": signals.gpu_mem_reserved_peak_bytes,
            "total_bytes": signals.gpu_mem_total_bytes,
            "used_peak_frac": signals.gpu_mem_used_peak_frac,
            "reserved_peak_frac": signals.gpu_mem_reserved_peak_frac,
            "used_peak_percent": signals.gpu_mem_used_peak_percent,
            "reserved_peak_percent": signals.gpu_mem_reserved_peak_percent,
            "reserved_overhang_ratio": signals.gpu_mem_reserved_overhang_ratio,
            "highest_overhang_rank": signals.highest_overhang_rank,
            "highest_used_rank": signals.highest_used_rank,
            "highest_reserved_rank": signals.highest_reserved_rank,
            "least_headroom_rank": signals.least_headroom_rank,
            "least_headroom_bytes": signals.least_headroom_bytes,
            "rank_gpu_used_imbalance_pct": signals.rank_gpu_used_imbalance_pct,
            "rank_gpu_reserved_imbalance_pct": (
                signals.rank_gpu_reserved_imbalance_pct
            ),
            "rank_gpu_used_imbalance_percent": (
                signals.rank_gpu_used_imbalance_percent
            ),
            "rank_gpu_reserved_imbalance_percent": (
                signals.rank_gpu_reserved_imbalance_percent
            ),
        },
    }

    per_rank_out = {
        str(rank_id): {
            "rank": int(rank_id),
            "metrics": item,
        }
        for rank_id, item in sorted(per_rank.items())
    }

    return DiagnosticResult(
        primary=primary,
        issues=issues,
        metric_attribution=metric_attribution,
        per_rank=per_rank_out,
    )


__all__ = [
    "ProcessDiagnosis",
    "build_process_diagnosis_result",
]
