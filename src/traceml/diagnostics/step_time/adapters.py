"""Adapters that feed summary step-time data into shared diagnosis rules."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from traceml.diagnostics.common import DiagnosticResult, diagnosis_to_dict
from traceml.diagnostics.step_time.api import (
    StepDiagnosis,
    build_step_diagnosis_result,
)
from traceml.diagnostics.step_time.policy import (
    SUMMARY_STEP_TIME_POLICY,
    StepTimeDiagnosisPolicy,
)
from traceml.loggers.error_log import get_error_logger
from traceml.renderers.step_time.schema import (
    StepCombinedTimeCoverage,
    StepCombinedTimeMetric,
    StepCombinedTimeSeries,
    StepCombinedTimeSummary,
)

_LOGGER = get_error_logger("StepTimeDiagnostics")


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


DEFAULT_SUMMARY_DIAG_CONFIG = SUMMARY_STEP_TIME_POLICY

# Per-rank per-step canonical metrics map:
# rank -> step -> metric_key -> value_ms
RankStepMetricSeries = Dict[int, Dict[int, Dict[str, float]]]


def _log_adapter_error(message: str, exc: Exception) -> None:
    """Log adapter failures without blocking final summary generation."""
    try:
        _LOGGER.exception("[TraceML] %s", message)
    except Exception:
        pass


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
    except Exception as exc:
        _log_adapter_error("Step-time summary step alignment failed.", exc)
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

        # The schema carries a `sum` series for renderers; diagnosis does not
        # need it here, so keep it shape-compatible with zeros.
        return StepCombinedTimeSeries(
            steps=list(steps),
            median=median_vals,
            worst=worst_vals,
            sum=[0.0] * len(steps),
        )
    except Exception as exc:
        _log_adapter_error("Step-time summary series preparation failed.", exc)
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


def _build_summary_per_rank_timing(
    rank_signals: Dict[int, RankStepSignals],
) -> Dict[int, Dict[str, float]]:
    """
    Build canonical per-rank timing maps for the shared diagnosis result path.

    These per-rank values mirror the summary semantics used elsewhere:
    - `step_time` is the effective local step time
    - `wait_proxy = max(0, effective_step - compute)`
    """
    out: Dict[int, Dict[str, float]] = {}
    for rank, item in rank_signals.items():
        dataloader = _finite_float(item.dataloader_ms)
        forward = _finite_float(item.forward_ms)
        backward = _finite_float(item.backward_ms)
        optimizer = _finite_float(item.optimizer_ms)
        step_cpu = _finite_float(item.step_cpu_ms)
        compute = forward + backward + optimizer
        step_effective = max(step_cpu, compute)
        wait = max(0.0, step_effective - compute)

        out[int(rank)] = {
            "dataloader_fetch": dataloader,
            "forward": forward,
            "backward": backward,
            "optimizer_step": optimizer,
            "step_time": step_effective,
            "wait_proxy": wait,
        }
    return out


def build_summary_step_diagnosis_result(
    rank_signals: Dict[int, RankStepSignals],
    *,
    max_rows: int,
    per_rank_step_metrics: Optional[RankStepMetricSeries] = None,
    policy: StepTimeDiagnosisPolicy = DEFAULT_SUMMARY_DIAG_CONFIG,
) -> Optional[DiagnosticResult]:
    """
    Build rich summary-mode diagnosis from per-rank averaged timing signals.

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
    if min_steps < int(policy.min_steps_for_diag):
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

    return build_step_diagnosis_result(
        metrics,
        thresholds=policy.thresholds,
        per_rank_timing=_build_summary_per_rank_timing(rank_signals),
    )


def build_summary_step_diagnosis(
    rank_signals: Dict[int, RankStepSignals],
    *,
    max_rows: int,
    per_rank_step_metrics: Optional[RankStepMetricSeries] = None,
    policy: StepTimeDiagnosisPolicy = DEFAULT_SUMMARY_DIAG_CONFIG,
) -> Optional[StepDiagnosis]:
    """
    Backward-compatible summary-mode primary diagnosis builder.
    """
    result = build_summary_step_diagnosis_result(
        rank_signals,
        max_rows=max_rows,
        per_rank_step_metrics=per_rank_step_metrics,
        policy=policy,
    )
    return result.primary if result is not None else None


def diagnosis_to_json(
    diagnosis: Optional[StepDiagnosis],
) -> Optional[Dict[str, Any]]:
    """
    Serialize StepDiagnosis into a JSON-friendly dict.
    """
    return diagnosis_to_dict(diagnosis, drop_none=True)


def diagnosis_result_to_json(
    result: Optional[DiagnosticResult],
) -> Optional[Dict[str, Any]]:
    """
    Serialize a rich step-time diagnosis result into a JSON-friendly dict.
    """
    return diagnosis_to_dict(result, drop_none=True)
