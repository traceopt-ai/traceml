"""
Summary-oriented system diagnosis API.

System diagnosis is intentionally conservative and primarily aimed at final
summary interpretation and JSON output. It is not yet used to change live
runtime rendering behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..common import BaseDiagnosis, DiagnosticResult, Severity, sort_issues
from .context import SystemSummarySignals, build_system_summary_signals
from .rules import run_system_rules


@dataclass(frozen=True)
class SystemDiagnosis(BaseDiagnosis):
    """
    Primary system diagnosis used in final-summary cards and JSON.
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
) -> SystemDiagnosis:
    return SystemDiagnosis(
        kind=str(kind),
        severity=severity,
        status=str(status),
        reason=str(reason),
        action=str(action),
        samples_used=int(samples_used),
    )


def _default_primary(signals: SystemSummarySignals) -> SystemDiagnosis:
    """
    Build the default non-issue system diagnosis.
    """
    if signals.gpu_available is False:
        return _mk_diag(
            kind="NO_GPU",
            severity="info",
            status="NO GPU",
            reason="No GPU telemetry was recorded for this summary window.",
            action="Treat CPU and RAM context as the main system signals.",
            samples_used=signals.samples,
        )

    return _mk_diag(
        kind="HEALTHY",
        severity="info",
        status="HEALTHY",
        reason="No dominant host or GPU system pressure signal is visible.",
        action="Treat the system context as normal unless training metrics disagree.",
        samples_used=signals.samples,
    )


def build_system_diagnosis_result(
    *,
    duration_s: Optional[float],
    system_samples: int,
    cpu_avg_percent: Optional[float],
    cpu_peak_percent: Optional[float],
    ram_avg_bytes: Optional[float],
    ram_peak_bytes: Optional[float],
    ram_total_bytes: Optional[float],
    gpu_available: Optional[bool],
    gpu_count: Optional[int],
    gpu_util_avg_percent: Optional[float],
    gpu_util_peak_percent: Optional[float],
    gpu_mem_avg_bytes: Optional[float],
    gpu_mem_peak_bytes: Optional[float],
    gpu_temp_avg_c: Optional[float],
    gpu_temp_peak_c: Optional[float],
    gpu_power_avg_w: Optional[float],
    gpu_power_peak_w: Optional[float],
    per_gpu: Dict[int, Dict[str, Optional[float]]],
) -> DiagnosticResult[SystemDiagnosis]:
    """
    Build one rich system diagnosis result from summary-level system signals.
    """
    signals = build_system_summary_signals(
        duration_s=duration_s,
        samples=system_samples,
        cpu_avg_percent=cpu_avg_percent,
        cpu_peak_percent=cpu_peak_percent,
        ram_avg_bytes=ram_avg_bytes,
        ram_peak_bytes=ram_peak_bytes,
        ram_total_bytes=ram_total_bytes,
        gpu_available=gpu_available,
        gpu_count=gpu_count,
        gpu_util_avg_percent=gpu_util_avg_percent,
        gpu_util_peak_percent=gpu_util_peak_percent,
        gpu_mem_avg_bytes=gpu_mem_avg_bytes,
        gpu_mem_peak_bytes=gpu_mem_peak_bytes,
        gpu_temp_avg_c=gpu_temp_avg_c,
        gpu_temp_peak_c=gpu_temp_peak_c,
        gpu_power_avg_w=gpu_power_avg_w,
        gpu_power_peak_w=gpu_power_peak_w,
        per_gpu=per_gpu,
    )

    issues = sort_issues(run_system_rules(signals))
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
        "cpu": {
            "avg_percent": signals.cpu_avg_percent,
            "peak_percent": signals.cpu_peak_percent,
        },
        "ram": {
            "avg_bytes": signals.ram_avg_bytes,
            "peak_bytes": signals.ram_peak_bytes,
            "total_bytes": signals.ram_total_bytes,
            "pressure_frac": signals.ram_pressure_frac,
        },
        "gpu_rollup": {
            "available": signals.gpu_available,
            "count": signals.gpu_count,
            "util_avg_percent": signals.gpu_util_avg_percent,
            "util_peak_percent": signals.gpu_util_peak_percent,
            "mem_avg_bytes": signals.gpu_mem_avg_bytes,
            "mem_peak_bytes": signals.gpu_mem_peak_bytes,
            "temp_avg_c": signals.gpu_temp_avg_c,
            "temp_peak_c": signals.gpu_temp_peak_c,
            "power_avg_w": signals.gpu_power_avg_w,
            "power_peak_w": signals.gpu_power_peak_w,
            "util_imbalance_pct": signals.gpu_util_imbalance_pct,
            "mem_imbalance_pct": signals.gpu_mem_imbalance_pct,
            "lowest_util_gpu_idx": signals.lowest_util_gpu_idx,
            "highest_util_gpu_idx": signals.highest_util_gpu_idx,
            "highest_mem_gpu_idx": signals.highest_mem_gpu_idx,
        },
    }

    per_rank = {
        str(gpu_idx): {
            "gpu_idx": int(gpu_idx),
            "metrics": item,
        }
        for gpu_idx, item in sorted(per_gpu.items())
    }

    return DiagnosticResult(
        primary=primary,
        issues=issues,
        metric_attribution=metric_attribution,
        per_rank=per_rank,
    )


__all__ = [
    "SystemDiagnosis",
    "build_system_diagnosis_result",
]
