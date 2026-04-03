"""
Summary-mode diagnosis adapter for STEP TIME post-run reports.

This module converts per-rank summary aggregates into the same metric schema used
by live step-time diagnostics, so one diagnosis engine can be reused across
renderers and summaries.

It supports optional step-aligned series generation for trend-aware diagnosis
notes. If step-level data is missing or cannot be aligned, it gracefully falls
back to summary-only diagnosis (no trend notes).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from traceml.diagnostics.step_time import (
    DiagnosisThresholds,
    StepDiagnosis,
    build_step_diagnosis,
)
from traceml.renderers.step_time.schema import (
    StepCombinedTimeCoverage,
    StepCombinedTimeMetric,
    StepCombinedTimeSeries,
    StepCombinedTimeSummary,
)


@dataclass(frozen=True)
class RankStepSignals:
    """
    Per-rank summary signals required for diagnosis.

    All timing values are milliseconds and represent per-step averages over each
    rank's analyzed summary window.
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

    Thresholds are slightly stricter than live diagnostics because post-run
    summaries should favor precision over sensitivity.
    """

    thresholds: DiagnosisThresholds = field(
        default_factory=lambda: DiagnosisThresholds(
            distributed_effect_warn=0.10,
            distributed_effect_crit=0.18,
            input_share_warn=0.30,
            input_share_crit=0.40,
            wait_share_warn=0.18,
            wait_share_crit=0.28,
            input_skew_warn=0.12,
            input_skew_crit=0.22,
            compute_skew_warn=0.12,
            compute_skew_crit=0.22,
            compute_share_min=0.12,
            input_bound_max_skew=0.05,
            compute_bound_max_skew=0.05,
            compute_bound_share_warn=0.88,
            compute_bound_share_crit=0.94,
            low_step_skew=0.04,
            min_steps_for_confident_diag=20,
        )
    )
    min_steps_for_diag: int = 20


DEFAULT_SUMMARY_DIAG_CONFIG = SummaryDiagnosisConfig()

# Per-rank per-step canonical metrics map:
# rank -> step -> metric_key -> value_ms
RankStepMetricSeries = Dict[int, Dict[int, Dict[str, float]]]


def _finite_float(x: Any) -> float:
    """Convert to float; return 0.0 for non-finite or invalid values."""
    try:
        v = float(x)
    except Exception:
        return 0.0
    return v if np.isfinite(v) else 0.0


def _common_suffix_steps(
    per_rank_step_metrics: RankStepMetricSeries,
    max_rows: int,
) -> List[int]:
    """
    Compute a robust common suffix of step ids across all ranks.

    Returns steps sorted ascending. Returns empty list on any issue.
    """
    try:
        if not per_rank_step_metrics:
            return []

        step_sets = []
        for step_map in per_rank_step_metrics.values():
            if not step_map:
                return []
            step_sets.append(set(step_map.keys()))

        common = set.intersection(*step_sets) if step_sets else set()
        if not common:
            return []

        steps = sorted(int(s) for s in common)
        if max_rows > 0:
            steps = steps[-int(max_rows) :]
        return steps
    except Exception:
        return []


def _build_metric_series(
    *,
    metric_key: str,
    steps: List[int],
    per_rank_step_metrics: RankStepMetricSeries,
) -> Optional[StepCombinedTimeSeries]:
    """
    Build median/worst per-step series for one canonical metric key.

    Returns None if series cannot be built safely.
    """
    try:
        if not steps or not per_rank_step_metrics:
            return None

        ranks = sorted(per_rank_step_metrics.keys())
        median_vals: List[float] = []
        worst_vals: List[float] = []

        for st in steps:
            vals = []
            for r in ranks:
                v = (
                    per_rank_step_metrics.get(r, {})
                    .get(st, {})
                    .get(metric_key, 0.0)
                )
                vals.append(_finite_float(v))

            arr = np.asarray(vals, dtype=np.float64)
            if arr.size == 0:
                median_vals.append(0.0)
                worst_vals.append(0.0)
            else:
                median_vals.append(float(np.median(arr)))
                worst_vals.append(float(np.max(arr)))

        # For compatibility with schema (sum list exists but isn't used by diagnosis),
        # fill `sum` with zeros of matching length.
        return StepCombinedTimeSeries(
            steps=list(steps),
            median=median_vals,
            worst=worst_vals,
            sum=[0.0] * len(steps),
        )
    except Exception:
        return None


def _metric_from_rank_values(
    *,
    metric_key: str,
    rank_values: Dict[int, float],
    coverage: StepCombinedTimeCoverage,
    series: Optional[StepCombinedTimeSeries] = None,
    worst_rank_override: Optional[int] = None,
) -> Optional[StepCombinedTimeMetric]:
    """
    Build one StepCombinedTimeMetric from rank->aggregate values.
    """
    if not rank_values:
        return None

    ranks = sorted(int(r) for r in rank_values.keys())
    arr = np.asarray(
        [_finite_float(rank_values[r]) for r in ranks],
        dtype=np.float64,
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
        series=series,
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
    per_rank_step_metrics: Optional[RankStepMetricSeries] = None,
    config: SummaryDiagnosisConfig = DEFAULT_SUMMARY_DIAG_CONFIG,
) -> Optional[StepDiagnosis]:
    """
    Build summary-mode diagnosis from per-rank averaged timing signals.

    Notes
    -----
    - Primary diagnosis uses robust rank-level aggregates.
    - Optional per-step series are used only to enable trend notes.
    - If series preparation fails, diagnosis still succeeds without trend notes.
    """
    if not rank_signals:
        return None

    ranks = sorted(rank_signals.keys())
    min_steps = min(s.steps_analyzed for s in rank_signals.values())
    if min_steps < int(config.min_steps_for_diag):
        return None

    # Optional step-aligned series support (for trend notes).
    common_steps: List[int] = []
    metric_series: Dict[str, Optional[StepCombinedTimeSeries]] = {}
    if per_rank_step_metrics:
        common_steps = _common_suffix_steps(
            per_rank_step_metrics, max_rows=max_rows
        )
        for mk in (
            "dataloader_fetch",
            "forward",
            "backward",
            "optimizer_step",
            "step_time",
            "wait_proxy",
        ):
            metric_series[mk] = _build_metric_series(
                metric_key=mk,
                steps=common_steps,
                per_rank_step_metrics=per_rank_step_metrics,
            )

    coverage_steps_used = (
        int(len(common_steps)) if common_steps else int(min_steps)
    )

    coverage = StepCombinedTimeCoverage(
        expected_steps=int(max_rows),
        steps_used=coverage_steps_used,
        completed_step=(int(common_steps[-1]) if common_steps else 0),
        world_size=len(ranks),
        ranks_present=len(ranks),
        incomplete=False,
    )

    dl = {r: _finite_float(s.dataloader_ms) for r, s in rank_signals.items()}
    fwd = {r: _finite_float(s.forward_ms) for r, s in rank_signals.items()}
    bwd = {r: _finite_float(s.backward_ms) for r, s in rank_signals.items()}
    opt = {r: _finite_float(s.optimizer_ms) for r, s in rank_signals.items()}
    step_raw = {
        r: _finite_float(s.step_cpu_ms) for r, s in rank_signals.items()
    }

    compute = {r: fwd[r] + bwd[r] + opt[r] for r in ranks}
    step_effective = {r: max(step_raw[r], compute[r]) for r in ranks}
    wait = {r: max(0.0, step_effective[r] - compute[r]) for r in ranks}

    # Keep overall rank identity aligned with summary-card semantics.
    overall_rank_scores = {r: dl[r] + step_effective[r] for r in ranks}
    overall_worst_rank = int(
        max(ranks, key=lambda r: (overall_rank_scores[r], -r))
    )

    metric_values = {
        "dataloader_fetch": dl,
        "forward": fwd,
        "backward": bwd,
        "optimizer_step": opt,
        "step_time": step_effective,
        "wait_proxy": wait,
    }

    metrics: List[StepCombinedTimeMetric] = []
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
            series=metric_series.get(key),
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
