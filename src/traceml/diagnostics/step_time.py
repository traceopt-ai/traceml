"""
Step-time diagnosis logic shared by live renderers and post-run summaries.

Semantics
---------
- `step_time` excludes dataloader fetch time.
- `wait_proxy = step_time - (forward + backward + optimizer_step)`.
- DDP stragglers are diagnosed from two layers of evidence:
  1. distributed effect:
     - wall-clock skew and/or elevated wait
  2. likely culprit:
     - dataloader imbalance
     - compute imbalance

Policy
------
- If only a subset of ranks is materially slower, emit a straggler diagnosis.
- If most/all ranks look similar, classify the run by dominant phase:
  - INPUT_BOUND
  - COMPUTE_BOUND
- If waiting dominates without a clear culprit, emit WAIT_HEAVY.
"""

from __future__ import annotations

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
    "INPUT_STRAGGLER",
    "COMPUTE_STRAGGLER",
    "INPUT_BOUND",
    "COMPUTE_BOUND",
    "WAIT_HEAVY",
]

_STATUS_BY_KIND: dict[DiagnosisKind, str] = {
    "NO_DATA": "NO DATA",
    "BALANCED": "BALANCED",
    "STRAGGLER": "STRAGGLER",
    "INPUT_STRAGGLER": "INPUT STRAGGLER",
    "COMPUTE_STRAGGLER": "COMPUTE STRAGGLER",
    "INPUT_BOUND": "INPUT-BOUND",
    "COMPUTE_BOUND": "COMPUTE-BOUND",
    "WAIT_HEAVY": "WAIT-HEAVY",
}


@dataclass(frozen=True)
class DiagnosisThresholds:
    """
    Thresholds controlling diagnosis selection.

    Design notes
    ------------
    - `distributed_effect_*` gates decide whether the window shows a multi-rank
      slowdown effect at all.
    - If effect exists, we attribute likely cause from dataloader or compute.
    - If effect does not exist, we classify the run by dominant phase.
    """

    distributed_effect_warn: float = 0.10
    distributed_effect_crit: float = 0.20

    input_share_warn: float = 0.25
    input_share_crit: float = 0.35

    wait_share_warn: float = 0.15
    wait_share_crit: float = 0.25

    input_skew_warn: float = 0.10
    input_skew_crit: float = 0.20

    compute_skew_warn: float = 0.10
    compute_skew_crit: float = 0.20
    compute_share_min: float = 0.10

    input_bound_max_skew: float = 0.06
    compute_bound_max_skew: float = 0.06

    compute_bound_share_warn: float = 0.85
    compute_bound_share_crit: float = 0.92

    low_step_skew: float = 0.05
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
    """
    Dominant local-work signal used for attribution.

    `share` is measured against typical step time.
    `skew` is cross-rank skew for the signal itself.
    """

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
    1. INPUT_STRAGGLER / COMPUTE_STRAGGLER / STRAGGLER
    2. INPUT_BOUND
    3. WAIT_HEAVY
    4. COMPUTE_BOUND
    5. BALANCED
    """
    metric_names = [m.metric for m in metrics]
    if len(metric_names) != len(set(metric_names)):
        return _mk_diag(
            kind="NO_DATA",
            severity="info",
            reason="Duplicate metric keys in diagnosis input.",
            action="Check upstream aggregation.",
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
    overall_worst_rank = _metric_worst_rank(step)

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
            reason=f"Only {steps_used} steps available.",
            action="Wait for a fuller window.",
            steps_used=steps_used,
            worst_rank=overall_worst_rank,
        )

    dl = by_key.get("dataloader_fetch")
    wait = by_key.get("wait_proxy")
    fwd = by_key.get("forward")
    bwd = by_key.get("backward")
    opt = by_key.get("optimizer_step")

    dl_total = _metric_total(dl, single_rank)
    wait_total = _metric_total(wait, single_rank)

    dl_share = _share(dl_total, step_total)
    wait_share = _share(wait_total, step_total)

    step_skew = _metric_skew(step, single_rank)
    wait_skew = _metric_skew(wait, single_rank)
    dl_skew = _metric_skew(dl, single_rank)
    dl_worst_rank = _metric_worst_rank(dl)

    dominant_compute = _dominant_compute_signal(
        forward=fwd,
        backward=bwd,
        optimizer=opt,
        step_total=step_total,
        single_rank=single_rank,
    )

    burden_skew = _overall_burden_skew(
        dataloader=dl,
        step=step,
        forward=fwd,
        backward=bwd,
        optimizer=opt,
        single_rank=single_rank,
    )

    distributed_effect = max(step_skew, wait_skew, burden_skew)

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

    # 1) STRAGGLER FAMILY
    if (
        not single_rank
        and distributed_effect >= thresholds.distributed_effect_warn
    ):
        severity = _severity(
            distributed_effect, thresholds.distributed_effect_crit
        )

        if (
            dl_worst_rank is not None
            and dl_skew >= thresholds.input_skew_warn
            and dl_share >= thresholds.compute_share_min
        ):
            return _emit(
                kind="INPUT_STRAGGLER",
                severity=severity,
                reason=(
                    f"Dataloader is imbalanced (+{_pct(dl_skew)}) on "
                    f"{_rank_str(dl_worst_rank)}."
                ),
                action=f"Inspect input loading on {_rank_str(dl_worst_rank)}.",
                worst_rank=dl_worst_rank,
                note=(
                    f"Distributed effect is +{_pct(distributed_effect)} across ranks."
                ),
            )

        if (
            dominant_compute is not None
            and dominant_compute.skew >= thresholds.compute_skew_warn
            and dominant_compute.share >= thresholds.compute_share_min
        ):
            return _emit(
                kind="COMPUTE_STRAGGLER",
                severity=severity,
                reason=(
                    f"{dominant_compute.label} is imbalanced "
                    f"(+{_pct(dominant_compute.skew)}) on "
                    f"{_rank_str(dominant_compute.worst_rank)}."
                ),
                action=(
                    f"Inspect {dominant_compute.label.lower()} on "
                    f"{_rank_str(dominant_compute.worst_rank)}."
                ),
                worst_rank=dominant_compute.worst_rank,
                note=(
                    f"Distributed effect is +{_pct(distributed_effect)} across ranks."
                ),
            )

        if wait_share >= thresholds.wait_share_warn:
            return _emit(
                kind="STRAGGLER",
                severity=severity,
                reason=f"Ranks are diverging and WAIT* is elevated ({_pct(wait_share)}).",
                action="Inspect sync points and uneven rank progress.",
                worst_rank=overall_worst_rank,
                note=f"Distributed effect is +{_pct(distributed_effect)} across ranks.",
            )

        return _emit(
            kind="STRAGGLER",
            severity=severity,
            reason=f"Ranks are diverging (+{_pct(distributed_effect)} effect).",
            action="Inspect the slowest rank and dominant phase.",
            worst_rank=overall_worst_rank,
        )

    # 2) INPUT-BOUND
    if dl_share >= thresholds.input_share_warn and (
        single_rank or dl_skew <= thresholds.input_bound_max_skew
    ):
        return _emit(
            kind="INPUT_BOUND",
            severity=_severity(dl_share, thresholds.input_share_crit),
            reason=f"Dataloader is {_pct(dl_share)} of step time.",
            action="Increase workers, prefetch, or storage throughput.",
            worst_rank=None if single_rank else dl_worst_rank,
        )

    # 3) WAIT-HEAVY
    if wait_share >= thresholds.wait_share_warn:
        return _emit(
            kind="WAIT_HEAVY",
            severity=_severity(wait_share, thresholds.wait_share_crit),
            reason=f"WAIT* is {_pct(wait_share)} of step time.",
            action="Inspect sync points, CPU stalls, or H2D copies.",
            worst_rank=None if single_rank else overall_worst_rank,
            note="WAIT* = step_time - (forward + backward + optimizer_step).",
        )

    # 4) COMPUTE-BOUND
    compute_total = _compute_total(
        forward=fwd,
        backward=bwd,
        optimizer=opt,
        single_rank=single_rank,
    )
    compute_share = _share(compute_total, step_total)
    compute_skew = (
        dominant_compute.skew if dominant_compute is not None else 0.0
    )

    if (
        compute_share >= thresholds.compute_bound_share_warn
        and dl_share < thresholds.input_share_warn
        and wait_share < thresholds.wait_share_warn
        and (single_rank or compute_skew <= thresholds.compute_bound_max_skew)
    ):
        label = (
            dominant_compute.label
            if dominant_compute is not None
            else "Compute"
        )
        return _emit(
            kind="COMPUTE_BOUND",
            severity=_severity(
                compute_share, thresholds.compute_bound_share_crit
            ),
            reason=f"{label} dominates the step ({_pct(compute_share)}).",
            action="Optimize model compute or reduce step cost.",
            worst_rank=None if single_rank else overall_worst_rank,
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
    """
    Return the visible total used for diagnosis.

    - single-rank: use worst_total (same as sum for one rank)
    - multi-rank: use median_total (typical rank)
    """
    if metric is None:
        return 0.0
    raw = (
        metric.summary.worst_total
        if single_rank
        else metric.summary.median_total
    )
    return _non_negative_finite(raw)


def _compute_total(
    *,
    forward: Optional[StepCombinedTimeMetric],
    backward: Optional[StepCombinedTimeMetric],
    optimizer: Optional[StepCombinedTimeMetric],
    single_rank: bool,
) -> float:
    """Return typical compute total for the step."""
    return (
        _metric_total(forward, single_rank)
        + _metric_total(backward, single_rank)
        + _metric_total(optimizer, single_rank)
    )


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

    Burden definition:
        dataloader_fetch + max(step_time, forward + backward + optimizer_step)

    This is used only to detect whether a distributed effect exists. It is not
    used to attribute culprit rank by itself.
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

    Used to detect a distributed slowdown effect, not to assign blame.
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
    """
    Pick the compute component with strongest skew, then strongest share.
    """
    candidates: list[ComputeSignal] = []

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

    return max(candidates, key=lambda item: (item.skew, item.share))


def _share(value: float, total: float) -> float:
    """Return a safe non-negative share."""
    total_safe = _non_negative_finite(total)
    if total_safe <= 0.0:
        return 0.0
    return max(0.0, _non_negative_finite(value) / total_safe)


def _pct(value: float) -> str:
    """Format a ratio as a percentage string."""
    return f"{_non_negative_finite(value) * 100.0:.1f}%"


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
