"""
Summary-oriented step-memory diagnosis API.

This module preserves the existing live primary diagnosis while adding a richer
multi-issue result for final summary JSON and downstream analysis.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from traceml.diagnostics.framework import DiagnosticResult, sort_issues
from traceml.diagnostics.step_memory import (
    DEFAULT_STEP_MEMORY_THRESHOLDS,
    StepMemoryDiagnosisThresholds,
    build_step_memory_diagnosis,
)
from traceml.renderers.step_memory.schema import StepMemoryCombinedMetric

from .context import build_step_memory_summary_signals
from .rules import run_step_memory_summary_rules


def build_step_memory_summary_diagnosis_result(
    metrics: Sequence[StepMemoryCombinedMetric],
    *,
    gpu_total_bytes: Optional[float] = None,
    per_rank: Optional[Dict[str, Any]] = None,
    thresholds: StepMemoryDiagnosisThresholds = DEFAULT_STEP_MEMORY_THRESHOLDS,
) -> DiagnosticResult:
    """
    Build a rich summary-oriented step-memory diagnosis result.

    Notes
    -----
    - `primary` intentionally comes from the existing live diagnosis engine so
      runtime, dashboard, and final summary surfaces stay coherent.
    - `issues` preserves all materially triggered summary signals across all
      available memory metrics.
    """
    primary = build_step_memory_diagnosis(
        metrics,
        gpu_total_bytes=gpu_total_bytes,
        thresholds=thresholds,
    )

    signals = build_step_memory_summary_signals(
        metrics,
        gpu_total_bytes=gpu_total_bytes,
        thresholds=thresholds,
    )

    issues = []
    metric_attribution: Dict[str, Any] = {}
    for metric in metrics:
        signal = signals.get(metric.metric)
        if signal is None:
            continue
        issues.extend(run_step_memory_summary_rules(signal))
        metric_attribution[metric.metric] = {
            "metric": signal.metric,
            "device": signal.device,
            "steps_used": signal.steps_used,
            "window_size": signal.window_size,
            "completed_step": signal.completed_step,
            "ranks_seen": signal.ranks_seen,
            "worst_rank": signal.worst_rank,
            "worst_peak_bytes": signal.worst_peak_bytes,
            "median_peak_bytes": signal.median_peak_bytes,
            "skew_ratio": signal.skew_ratio,
            "skew_pct": signal.skew_pct,
            "pressure_frac": signal.pressure_frac,
            "trend": {
                "eligible": signal.trend.eligible,
                "baseline_avg_bytes": signal.trend.baseline_avg_bytes,
                "mid_avg_bytes": signal.trend.mid_avg_bytes,
                "recent_avg_bytes": signal.trend.recent_avg_bytes,
                "overall_abs_delta_bytes": (
                    signal.trend.overall_abs_delta_bytes
                ),
                "overall_worst_growth_pct": (
                    signal.trend.overall_worst_growth_pct
                ),
                "overall_median_growth_pct": (
                    signal.trend.overall_median_growth_pct
                ),
                "early": signal.trend.early,
                "confirmed": signal.trend.confirmed,
                "score": signal.trend.score,
            },
        }

    return DiagnosticResult(
        primary=primary,
        issues=sort_issues(issues),
        metric_attribution=metric_attribution,
        per_rank=dict(per_rank or {}),
    )


__all__ = [
    "build_step_memory_summary_diagnosis_result",
]
