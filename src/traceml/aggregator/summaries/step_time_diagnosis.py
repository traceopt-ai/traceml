"""
Summary-mode diagnosis adapter for STEP TIME post-run reports.

This module converts per-rank summary averages into the same metric schema used
by live step-time diagnostics, so we can reuse one diagnosis engine across
renderers while keeping summary-specific thresholds and confidence gating.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from traceml.renderers.step_time.diagnostics import (
    DiagnosisThresholds,
    StepDiagnosis,
    build_step_diagnosis,
)
from traceml.renderers.step_time.schema import (
    StepCombinedTimeCoverage,
    StepCombinedTimeMetric,
    StepCombinedTimeSummary,
)


@dataclass(frozen=True)
class RankStepSignals:
    """
    Per-rank summary signals required for diagnosis.

    All timing values are expected in milliseconds and represent per-step
    averages over each rank's analyzed summary window.
    """

    steps_analyzed: int
    dataloader_ms: float
    forward_ms: float
    backward_ms: float
    optimizer_ms: float
    step_cpu_ms: float


@dataclass(frozen=True)
class SummaryDiagnosisConfig:
    """
    Configuration for summary-mode diagnosis behavior.

    `thresholds` are intentionally stricter than live diagnostics because
    post-run summaries should prefer precision over sensitivity.
    """

    thresholds: DiagnosisThresholds = field(
        default_factory=lambda: DiagnosisThresholds(
            straggler_skew_warn=0.10,
            straggler_skew_crit=0.18,
            input_share_warn=0.30,
            input_share_crit=0.40,
            wait_share_warn=0.18,
            wait_share_crit=0.28,
            compute_skew_warn=0.12,
            compute_skew_crit=0.22,
            compute_share_min=0.12,
            low_step_skew=0.04,
        )
    )
    min_steps_for_diag: int = 20


DEFAULT_SUMMARY_DIAG_CONFIG = SummaryDiagnosisConfig()


def _finite_float(x: Any) -> float:
    """Convert to float; return 0.0 for non-finite or invalid values."""
    try:
        v = float(x)
    except Exception:
        return 0.0
    return v if np.isfinite(v) else 0.0


def _metric_from_rank_values(
    *,
    metric_key: str,
    rank_values: Dict[int, float],
    coverage: StepCombinedTimeCoverage,
    worst_rank_override: Optional[int] = None,
) -> Optional[StepCombinedTimeMetric]:
    """
    Build one StepCombinedTimeMetric from rank->aggregate values.
    """
    if not rank_values:
        return None

    ranks = sorted(int(r) for r in rank_values.keys())
    arr = np.asarray(
        [_finite_float(rank_values[r]) for r in ranks], dtype=np.float64
    )
    if arr.size == 0:
        return None

    median_total = float(np.median(arr))
    worst_idx = int(np.argmax(arr))
    worst_total = float(arr[worst_idx])
    worst_rank = int(ranks[worst_idx])

    if coverage.ranks_present <= 1:
        median_total = worst_total
        skew_ratio = 0.0
        skew_pct = 0.0
    elif median_total > 0.0:
        skew_ratio = worst_total / median_total
        skew_pct = (worst_total - median_total) / median_total
    else:
        skew_ratio = 0.0
        skew_pct = 0.0

    if worst_rank_override is not None:
        worst_rank = int(worst_rank_override)

    return StepCombinedTimeMetric(
        metric=str(metric_key),
        clock="mixed",
        series=None,
        summary=StepCombinedTimeSummary(
            window_size=int(coverage.expected_steps),
            steps_used=int(coverage.steps_used),
            median_total=float(median_total),
            worst_total=float(worst_total),
            worst_rank=int(worst_rank),
            skew_ratio=float(skew_ratio),
            skew_pct=float(skew_pct),
        ),
        coverage=coverage,
    )


def build_summary_step_diagnosis(
    rank_signals: Dict[int, RankStepSignals],
    *,
    max_rows: int,
    config: SummaryDiagnosisConfig = DEFAULT_SUMMARY_DIAG_CONFIG,
) -> Optional[StepDiagnosis]:
    """
    Build a summary-mode diagnosis from per-rank averaged timing signals.

    Returns None when there is not enough data to produce a stable diagnosis.
    """
    if not rank_signals:
        return None

    ranks = sorted(rank_signals.keys())
    min_steps = min(s.steps_analyzed for s in rank_signals.values())
    if min_steps < int(config.min_steps_for_diag):
        return None

    coverage = StepCombinedTimeCoverage(
        expected_steps=int(max_rows),
        steps_used=int(min_steps),
        completed_step=0,  # synthetic summary context
        world_size=len(ranks),
        ranks_present=len(ranks),
        incomplete=False,
    )

    dl = {r: _finite_float(s.dataloader_ms) for r, s in rank_signals.items()}
    fwd = {r: _finite_float(s.forward_ms) for r, s in rank_signals.items()}
    bwd = {r: _finite_float(s.backward_ms) for r, s in rank_signals.items()}
    opt = {r: _finite_float(s.optimizer_ms) for r, s in rank_signals.items()}
    step_cpu = {
        r: _finite_float(s.step_cpu_ms) for r, s in rank_signals.items()
    }

    # Keep the same semantics as live diagnostics.
    wait = {
        r: max(0.0, step_cpu[r] - (fwd[r] + bwd[r] + opt[r])) for r in ranks
    }

    overall_rank_scores = {
        r: dl[r] + max(step_cpu[r], fwd[r] + bwd[r] + opt[r]) for r in ranks
    }
    overall_worst_rank = int(
        max(ranks, key=lambda r: (overall_rank_scores[r], -r))
    )

    metric_values = {
        "dataloader_fetch": dl,
        "forward": fwd,
        "backward": bwd,
        "optimizer_step": opt,
        "step_time": step_cpu,
        "wait_proxy": wait,
    }

    metrics: list[StepCombinedTimeMetric] = []
    for key in (
        "dataloader_fetch",
        "forward",
        "backward",
        "optimizer_step",
        "step_time",
        "wait_proxy",
    ):
        metric = _metric_from_rank_values(
            metric_key=key,
            rank_values=metric_values[key],
            coverage=coverage,
            worst_rank_override=(
                overall_worst_rank if key == "step_time" else None
            ),
        )
        if metric is not None:
            metrics.append(metric)

    if not metrics:
        return None

    return build_step_diagnosis(metrics, thresholds=config.thresholds)


def diagnosis_to_json(
    diagnosis: Optional[StepDiagnosis],
) -> Optional[Dict[str, Any]]:
    """
    Serialize StepDiagnosis into a JSON-friendly dict.
    """
    if diagnosis is None:
        return None

    return {
        "kind": diagnosis.kind,
        "severity": diagnosis.severity,
        "status": diagnosis.status,
        "reason": diagnosis.reason,
        "action": diagnosis.action,
        "steps_used": diagnosis.steps_used,
        "worst_rank": diagnosis.worst_rank,
        "note": diagnosis.note,
    }
