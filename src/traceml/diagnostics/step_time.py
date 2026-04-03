"""
Step-time diagnosis logic shared by live renderers and post-run summaries.

Semantics
---------
- `step_time` excludes dataloader fetch time.
- `wait_proxy = step_time - (forward + backward + optimizer_step)`.
- `step_time.summary.worst_rank` is treated as the UI-visible overall worst rank.
"""

import math
from dataclasses import dataclass, replace
from typing import Literal, Optional, Sequence

from traceml.renderers.step_time.schema import StepCombinedTimeMetric

from .common import BaseDiagnosis, Severity, validate_confidence
from .step_time_trend import (
    DEFAULT_STEP_TREND_HEURISTICS,
    build_step_trend_note,
)

DiagnosisKind = Literal[
    "NO_DATA",
    "BALANCED",
    "STRAGGLER",
    "INPUT_BOUND",
    "WAIT_HEAVY",
    "COMPUTE_IMBALANCE",
]

_STATUS_BY_KIND: dict[DiagnosisKind, str] = {
    "NO_DATA": "NO DATA",
    "BALANCED": "BALANCED",
    "STRAGGLER": "STRAGGLER",
    "INPUT_BOUND": "INPUT-BOUND",
    "WAIT_HEAVY": "WAIT-HEAVY",
    "COMPUTE_IMBALANCE": "COMPUTE-IMBALANCE",
}


@dataclass(frozen=True)
class DiagnosisThresholds:
    """Thresholds controlling diagnosis selection."""

    straggler_skew_warn: float = 0.10
    straggler_skew_crit: float = 0.20

    input_share_warn: float = 0.25
    input_share_crit: float = 0.35

    wait_share_warn: float = 0.15
    wait_share_crit: float = 0.25

    compute_skew_warn: float = 0.10
    compute_skew_crit: float = 0.20
    compute_share_min: float = 0.10

    low_step_skew: float = 0.05
    straggler_dl_share_min: float = 0.15

    # Count of steps required for a stable diagnosis.
    min_steps_for_confident_diag: int = 8


DEFAULT_THRESHOLDS = DiagnosisThresholds()


@dataclass(frozen=True)
class StepDiagnosis(BaseDiagnosis):
    """Diagnosis payload used by step-time renderers and summaries."""

    kind: DiagnosisKind
    steps_used: int
    worst_rank: Optional[int] = None
    note: Optional[str] = None
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        validate_confidence(self.confidence)


@dataclass(frozen=True)
class ComputeSignal:
    """Dominant compute-side signal used for diagnosis."""

    label: str
    share: float
    skew: float
    worst_rank: Optional[int]


def _mk_diag(
    *,
    kind: DiagnosisKind,
    severity: Severity,
    reason: str,
    action: str,
    steps_used: int,
    worst_rank: Optional[int] = None,
    note: Optional[str] = None,
) -> StepDiagnosis:
    return StepDiagnosis(
        kind=kind,
        severity=severity,
        status=_STATUS_BY_KIND[kind],
        reason=reason,
        action=action,
        steps_used=int(steps_used),
        worst_rank=worst_rank,
        note=note,
    )


def _merge_note(base: Optional[str], extra: Optional[str]) -> Optional[str]:
    if not extra:
        return base
    if not base:
        return extra
    return f"{base} {extra}"


def _apply_trend_note(
    diagnosis: StepDiagnosis,
    *,
    single_rank: bool,
    step_metric: Optional[StepCombinedTimeMetric],
    wait_metric: Optional[StepCombinedTimeMetric],
    dataloader_metric: Optional[StepCombinedTimeMetric],
    wait_share: float,
    dataloader_share: float,
    thresholds: DiagnosisThresholds,
) -> StepDiagnosis:
    """
    Best-effort trend annotation.

    This function never raises; on any failure it returns the original diagnosis.
    """
    try:
        trend_note = build_step_trend_note(
            diagnosis_kind=diagnosis.kind,
            steps_used=diagnosis.steps_used,
            single_rank=single_rank,
            step_metric=step_metric,
            wait_metric=wait_metric,
            dataloader_metric=dataloader_metric,
            wait_share=wait_share,
            dataloader_share=dataloader_share,
            wait_warn_threshold=thresholds.wait_share_warn,
            input_warn_threshold=thresholds.input_share_warn,
            cfg=DEFAULT_STEP_TREND_HEURISTICS,
        )
        if not trend_note:
            return diagnosis
        return replace(diagnosis, note=_merge_note(diagnosis.note, trend_note))
    except Exception:
        return diagnosis


def build_step_diagnosis(
    metrics: Sequence[StepCombinedTimeMetric],
    thresholds: DiagnosisThresholds = DEFAULT_THRESHOLDS,
) -> StepDiagnosis:
    """
    Build one primary diagnosis from step-combined metrics.

    Priority
    --------
    1. STRAGGLER
    2. INPUT_BOUND
    3. WAIT_HEAVY
    4. COMPUTE_IMBALANCE
    5. BALANCED
    """
    metric_names = [m.metric for m in metrics]
    if len(metric_names) != len(set(metric_names)):
        return _mk_diag(
            kind="NO_DATA",
            severity="info",
            reason="Duplicate metric keys found in diagnosis input.",
            action="Check upstream metric aggregation for duplicate entries.",
            steps_used=0,
        )

    by_key = {metric.metric: metric for metric in metrics}

    step = by_key.get("step_time")
    if step is None:
        return _mk_diag(
            kind="NO_DATA",
            severity="info",
            reason="step_time metric is missing.",
            action="Wait for the first complete window.",
            steps_used=0,
        )

    coverage = step.coverage
    single_rank = (coverage.world_size <= 1) or (coverage.ranks_present <= 1)
    steps_used = int(step.summary.steps_used)
    overall_worst_rank = step.summary.worst_rank

    step_total = _metric_total(step, single_rank)

    if step_total <= 0.0:
        return _mk_diag(
            kind="NO_DATA",
            severity="info",
            reason="No usable step-time data yet.",
            action="Wait for the first complete window.",
            steps_used=steps_used,
            worst_rank=overall_worst_rank,
        )

    if steps_used < thresholds.min_steps_for_confident_diag:
        return _mk_diag(
            kind="NO_DATA",
            severity="info",
            reason=(
                f"Only {steps_used} steps available; "
                "need more samples for a confident diagnosis."
            ),
            action="Wait for more completed steps in the window.",
            steps_used=steps_used,
            worst_rank=overall_worst_rank,
        )

    dl = by_key.get("dataloader_fetch")
    wait = by_key.get("wait_proxy")
    fwd = by_key.get("forward")
    bwd = by_key.get("backward")
    opt = by_key.get("optimizer_step")

    step_skew = _metric_skew(step, single_rank)
    burden_skew = _overall_burden_skew(
        dataloader=dl,
        step=step,
        forward=fwd,
        backward=bwd,
        optimizer=opt,
        single_rank=single_rank,
    )

    dl_share = _share(_metric_total(dl, single_rank), step_total)
    dl_skew = _metric_skew(dl, single_rank)
    dl_worst_rank = _metric_worst_rank(dl)

    wait_share = _share(_metric_total(wait, single_rank), step_total)

    dominant_compute = _dominant_compute_signal(
        forward=fwd,
        backward=bwd,
        optimizer=opt,
        step_total=step_total,
        single_rank=single_rank,
    )

    def _finalize(diag: StepDiagnosis) -> StepDiagnosis:
        return _apply_trend_note(
            diag,
            single_rank=single_rank,
            step_metric=step,
            wait_metric=wait,
            dataloader_metric=dl,
            wait_share=wait_share,
            dataloader_share=dl_share,
            thresholds=thresholds,
        )

    def _emit(
        *,
        kind: DiagnosisKind,
        severity: Severity,
        reason: str,
        action: str,
        worst_rank: Optional[int] = None,
        note: Optional[str] = None,
        apply_trend: bool = True,
    ) -> StepDiagnosis:
        diag = _mk_diag(
            kind=kind,
            severity=severity,
            reason=reason,
            action=action,
            steps_used=steps_used,
            worst_rank=worst_rank,
            note=note,
        )
        return _finalize(diag) if apply_trend else diag

    # 1) STRAGGLER
    if not single_rank and burden_skew >= thresholds.straggler_skew_warn:
        rank_str = _rank_str(overall_worst_rank)
        severity = _severity(burden_skew, thresholds.straggler_skew_crit)

        if (
            dl_share >= thresholds.straggler_dl_share_min
            and dl_skew >= thresholds.compute_skew_warn
            and dl_worst_rank == overall_worst_rank
        ):
            return _emit(
                kind="STRAGGLER",
                severity=severity,
                reason=(
                    f"Overall burden skew is +{_pct(burden_skew)}; "
                    f"{rank_str} also leads dataloader imbalance."
                ),
                action=f"Check input loading on {rank_str}.",
                worst_rank=overall_worst_rank,
            )

        if (
            dominant_compute is not None
            and dominant_compute.skew >= thresholds.compute_skew_warn
            and dominant_compute.share >= thresholds.compute_share_min
            and dominant_compute.worst_rank == overall_worst_rank
        ):
            return _emit(
                kind="STRAGGLER",
                severity=severity,
                reason=(
                    f"Overall burden skew is +{_pct(burden_skew)}; "
                    f"{dominant_compute.label} is most imbalanced on {rank_str}."
                ),
                action=f"Inspect {dominant_compute.label.lower()} on {rank_str}.",
                worst_rank=overall_worst_rank,
            )

        if wait_share >= thresholds.wait_share_warn:
            return _emit(
                kind="STRAGGLER",
                severity=severity,
                reason=(
                    f"Overall burden skew is +{_pct(burden_skew)}; "
                    f"WAIT* is elevated on {rank_str}."
                ),
                action=f"Inspect sync / CPU stalls on {rank_str}.",
                worst_rank=overall_worst_rank,
            )

        return _emit(
            kind="STRAGGLER",
            severity=severity,
            reason=(
                f"Overall burden skew is +{_pct(burden_skew)}; "
                f"worst rank is {rank_str}."
            ),
            action=f"Inspect slower work on {rank_str}.",
            worst_rank=overall_worst_rank,
        )

    # 2) INPUT-BOUND
    if dl_share >= thresholds.input_share_warn:
        reason = f"Dataloader is {_pct(dl_share)} of step time."
        if not single_rank and dl_worst_rank is not None:
            reason = (
                f"Dataloader is {_pct(dl_share)} of step time; "
                f"worst rank is {_rank_str(dl_worst_rank)}."
            )

        return _emit(
            kind="INPUT_BOUND",
            severity=_severity(dl_share, thresholds.input_share_crit),
            reason=reason,
            action="Increase workers, prefetch, or storage throughput.",
            worst_rank=None if single_rank else dl_worst_rank,
        )

    # 3) WAIT-HEAVY
    if wait_share >= thresholds.wait_share_warn:
        return _emit(
            kind="WAIT_HEAVY",
            severity=_severity(wait_share, thresholds.wait_share_crit),
            reason=f"WAIT* is {_pct(wait_share)} of step time.",
            action="Inspect sync points, CPU stalls, and H2D copies.",
            worst_rank=None if single_rank else overall_worst_rank,
            note="WAIT* = step_time - (forward + backward + optimizer_step).",
        )

    # 4) COMPUTE-IMBALANCE
    if (
        not single_rank
        and dominant_compute is not None
        and dominant_compute.skew >= thresholds.compute_skew_warn
        and dominant_compute.share >= thresholds.compute_share_min
    ):
        note = None
        if step_skew < thresholds.low_step_skew:
            note = "Step time stays fairly balanced, so this may be partly hidden."

        return _emit(
            kind="COMPUTE_IMBALANCE",
            severity=_severity(
                dominant_compute.skew, thresholds.compute_skew_crit
            ),
            reason=(
                f"{dominant_compute.label} skew is +{_pct(dominant_compute.skew)} "
                f"on {_rank_str(dominant_compute.worst_rank)}."
            ),
            action=(
                f"Inspect {dominant_compute.label.lower()} "
                f"on {_rank_str(dominant_compute.worst_rank)}."
            ),
            worst_rank=dominant_compute.worst_rank,
            note=note,
        )

    # 5) BALANCED
    return _emit(
        kind="BALANCED",
        severity="info",
        reason="No dominant bottleneck is visible in this window.",
        action="Focus on throughput only if overall speed is still low.",
        worst_rank=None if single_rank else overall_worst_rank,
    )


def _non_negative_finite(value: float) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return max(0.0, out)


def _metric_total(
    metric: Optional[StepCombinedTimeMetric],
    single_rank: bool,
) -> float:
    """Return the visible total used for diagnosis."""
    if metric is None:
        return 0.0
    raw = (
        metric.summary.worst_total
        if single_rank
        else metric.summary.median_total
    )
    return _non_negative_finite(raw)


def _overall_burden_summary(
    *,
    dataloader: Optional[StepCombinedTimeMetric],
    step: Optional[StepCombinedTimeMetric],
    forward: Optional[StepCombinedTimeMetric],
    backward: Optional[StepCombinedTimeMetric],
    optimizer: Optional[StepCombinedTimeMetric],
    single_rank: bool,
) -> tuple[float, float]:
    """
    Return median-visible and worst-visible overall step burden.

    The burden definition matches the renderer's dashboard ranking logic:
        dataloader_fetch + max(step_time, forward + backward + optimizer_step)

    Notes
    -----
    - For multi-rank runs we compute burden from median and worst summaries.
    - This is an intentionally stable approximation for diagnosis.
    """

    def _pick(metric: Optional[StepCombinedTimeMetric], which: str) -> float:
        if metric is None:
            return 0.0
        raw = (
            metric.summary.worst_total
            if which == "worst"
            else metric.summary.median_total
        )
        return _non_negative_finite(raw)

    dl_med = _pick(dataloader, "median")
    dl_worst = _pick(dataloader, "worst")

    step_med = _pick(step, "median")
    step_worst = _pick(step, "worst")

    compute_med = (
        _pick(forward, "median")
        + _pick(backward, "median")
        + _pick(optimizer, "median")
    )
    compute_worst = (
        _pick(forward, "worst")
        + _pick(backward, "worst")
        + _pick(optimizer, "worst")
    )

    visible_median = dl_med + max(step_med, compute_med)
    visible_worst = dl_worst + max(step_worst, compute_worst)

    if single_rank:
        visible_median = visible_worst

    return visible_median, visible_worst


def _metric_skew(
    metric: Optional[StepCombinedTimeMetric],
    single_rank: bool,
) -> float:
    """Return skew for multi-rank runs, else 0."""
    if metric is None or single_rank:
        return 0.0
    return _non_negative_finite(metric.summary.skew_pct)


def _overall_burden_skew(
    *,
    dataloader: Optional[StepCombinedTimeMetric],
    step: Optional[StepCombinedTimeMetric],
    forward: Optional[StepCombinedTimeMetric],
    backward: Optional[StepCombinedTimeMetric],
    optimizer: Optional[StepCombinedTimeMetric],
    single_rank: bool,
) -> float:
    """
    Return skew of the combined overall step burden.

    This is used for straggler detection instead of step_time skew alone.
    """
    if single_rank:
        return 0.0

    median_total, worst_total = _overall_burden_summary(
        dataloader=dataloader,
        step=step,
        forward=forward,
        backward=backward,
        optimizer=optimizer,
        single_rank=single_rank,
    )

    if median_total <= 0.0:
        return 0.0

    return max(0.0, (worst_total - median_total) / median_total)


def _metric_worst_rank(
    metric: Optional[StepCombinedTimeMetric],
) -> Optional[int]:
    """Return the metric's worst rank, if available."""
    if metric is None or metric.summary.worst_rank is None:
        return None
    try:
        return int(metric.summary.worst_rank)
    except Exception:
        return None


def _dominant_compute_signal(
    *,
    forward: Optional[StepCombinedTimeMetric],
    backward: Optional[StepCombinedTimeMetric],
    optimizer: Optional[StepCombinedTimeMetric],
    step_total: float,
    single_rank: bool,
) -> Optional[ComputeSignal]:
    """Pick the compute component with strongest skew, then share."""
    candidates = []

    for label, metric in (
        ("Forward", forward),
        ("Backward", backward),
        ("Optimizer", optimizer),
    ):
        if metric is None:
            continue

        total = _metric_total(metric, single_rank)
        if total <= 0.0:
            continue

        candidates.append(
            ComputeSignal(
                label=label,
                share=_share(total, step_total),
                skew=_metric_skew(metric, single_rank),
                worst_rank=_metric_worst_rank(metric),
            )
        )

    if not candidates:
        return None

    return max(candidates, key=lambda x: (x.skew, x.share))


def _share(value: float, total: float) -> float:
    """Return a safe non-negative share."""
    total_safe = _non_negative_finite(total)
    if total_safe <= 0.0:
        return 0.0
    return max(0.0, _non_negative_finite(value) / total_safe)


def _pct(value: float) -> str:
    """Format a ratio as a percentage string."""
    return f"{_non_negative_finite(value) * 100:.1f}%"


def _rank_str(rank: Optional[int]) -> str:
    """Format a rank identifier for UI text."""
    return f"r{rank}" if rank is not None else "—"


def _severity(value: float, crit_threshold: float) -> Severity:
    """Map a scalar signal to warn or crit severity."""
    return "crit" if _non_negative_finite(value) >= crit_threshold else "warn"


__all__ = [
    "Severity",
    "DiagnosisKind",
    "DiagnosisThresholds",
    "DEFAULT_THRESHOLDS",
    "StepDiagnosis",
    "ComputeSignal",
    "build_step_diagnosis",
]
