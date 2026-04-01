"""
Step-time diagnosis logic shared by live renderers and post-run summaries.

Semantics
---------
- `step_time` excludes dataloader fetch time.
- `wait_proxy = step_time - (forward + backward + optimizer_step)`.
- `step_time.summary.worst_rank` is treated as the UI-visible overall worst rank.
"""

import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from traceml.renderers.step_time.schema import StepCombinedTimeMetric

from .common import BaseDiagnosis, Severity, validate_confidence

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

    # 1) STRAGGLER
    if not single_rank and step_skew >= thresholds.straggler_skew_warn:
        rank_str = _rank_str(overall_worst_rank)
        severity = _severity(step_skew, thresholds.straggler_skew_crit)

        if (
            dl_share >= thresholds.straggler_dl_share_min
            and dl_skew >= thresholds.compute_skew_warn
            and dl_worst_rank == overall_worst_rank
        ):
            return _mk_diag(
                kind="STRAGGLER",
                severity=severity,
                reason=(
                    f"Step skew is +{_pct(step_skew)}; "
                    f"{rank_str} also leads dataloader imbalance."
                ),
                action=f"Check input loading on {rank_str}.",
                steps_used=steps_used,
                worst_rank=overall_worst_rank,
            )

        if (
            dominant_compute is not None
            and dominant_compute.skew >= thresholds.compute_skew_warn
            and dominant_compute.share >= thresholds.compute_share_min
            and dominant_compute.worst_rank == overall_worst_rank
        ):
            return _mk_diag(
                kind="STRAGGLER",
                severity=severity,
                reason=(
                    f"Step skew is +{_pct(step_skew)}; "
                    f"{dominant_compute.label} is most imbalanced on {rank_str}."
                ),
                action=f"Inspect {dominant_compute.label.lower()} on {rank_str}.",
                steps_used=steps_used,
                worst_rank=overall_worst_rank,
            )

        if wait_share >= thresholds.wait_share_warn:
            return _mk_diag(
                kind="STRAGGLER",
                severity=severity,
                reason=(
                    f"Step skew is +{_pct(step_skew)}; "
                    f"WAIT* is elevated on {rank_str}."
                ),
                action=f"Inspect sync / CPU stalls on {rank_str}.",
                steps_used=steps_used,
                worst_rank=overall_worst_rank,
            )

        return _mk_diag(
            kind="STRAGGLER",
            severity=severity,
            reason=(
                f"Step skew is +{_pct(step_skew)}; "
                f"worst rank is {rank_str}."
            ),
            action=f"Inspect slower work on {rank_str}.",
            steps_used=steps_used,
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

        return _mk_diag(
            kind="INPUT_BOUND",
            severity=_severity(dl_share, thresholds.input_share_crit),
            reason=reason,
            action="Increase workers, prefetch, or storage throughput.",
            steps_used=steps_used,
            worst_rank=None if single_rank else dl_worst_rank,
        )

    # 3) WAIT-HEAVY
    if wait_share >= thresholds.wait_share_warn:
        return _mk_diag(
            kind="WAIT_HEAVY",
            severity=_severity(wait_share, thresholds.wait_share_crit),
            reason=f"WAIT* is {_pct(wait_share)} of step time.",
            action="Inspect sync points, CPU stalls, and H2D copies.",
            steps_used=steps_used,
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

        return _mk_diag(
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
            steps_used=steps_used,
            worst_rank=dominant_compute.worst_rank,
            note=note,
        )

    # 5) BALANCED
    return _mk_diag(
        kind="BALANCED",
        severity="info",
        reason="No dominant bottleneck is visible in this window.",
        action="Focus on throughput only if overall speed is still low.",
        steps_used=steps_used,
        worst_rank=None if single_rank else overall_worst_rank,
    )


def format_cli_diagnosis(diagnosis: StepDiagnosis) -> str:
    """Render a short Rich-friendly diagnosis block for terminal output."""
    status = _styled_status(diagnosis.status, diagnosis.severity)
    return "\n".join(
        [
            f"[bold]Issue:[/bold] {status}",
            f"[bold]Why:[/bold] {diagnosis.reason}",
            f"[bold]Hint:[/bold] {diagnosis.action}",
        ]
    )


def format_dashboard_diagnosis(diagnosis: StepDiagnosis) -> str:
    """Render a short diagnosis block for dashboard use."""
    meta = f"*Window: {diagnosis.steps_used} steps"
    if diagnosis.worst_rank is not None and diagnosis.kind != "BALANCED":
        meta += f" · Worst rank: {_rank_str(diagnosis.worst_rank)}"
    meta += "*"

    text = (
        f"**Status:** {diagnosis.status}  \n"
        f"**Why:** {diagnosis.reason}  \n"
        f"**Next:** {diagnosis.action}  \n"
        f"{meta}"
    )
    if diagnosis.note:
        text += f"  \n*Note:* {diagnosis.note}"
    return text


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


def _metric_skew(
    metric: Optional[StepCombinedTimeMetric],
    single_rank: bool,
) -> float:
    """Return skew for multi-rank runs, else 0."""
    if metric is None or single_rank:
        return 0.0
    return _non_negative_finite(metric.summary.skew_pct)


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


def _styled_status(status: str, severity: Severity) -> str:
    """Render a colored status label for Rich CLI output."""
    style = {
        "crit": "bold red",
        "warn": "bold yellow",
        "info": "bold green",
    }.get(severity, "bold")
    return f"[{style}]{status}[/{style}]"


__all__ = [
    "Severity",
    "DiagnosisKind",
    "DiagnosisThresholds",
    "DEFAULT_THRESHOLDS",
    "StepDiagnosis",
    "ComputeSignal",
    "build_step_diagnosis",
    "format_cli_diagnosis",
    "format_dashboard_diagnosis",
]
