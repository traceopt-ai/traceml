"""
Shared diagnosis logic for step-combined timing summaries.

Purpose
-------
This module converts already-computed step timing metrics into one short,
action-oriented diagnosis that can be reused by both the CLI and dashboard
renderers.

Current semantics
-----------------
- `step_time` excludes dataloader fetch time.
- `wait_proxy` is defined as:

      wait_proxy = step_time - (forward + backward + optimizer_step)

  so WAIT* does NOT include dataloader time.
- In the current compute layer, `step_time.summary.worst_rank` is overwritten
  with the "overall worst rank" used by the UI. That overall rank identity is
  based on:

      dataloader_fetch + max(step_time, forward + backward + optimizer_step)

  This module intentionally follows that UI-visible meaning. Therefore, whenever
  this file refers to "worst rank", it means the rank exposed by the compute
  layer for the step-time row, i.e. the overall worst rank.

Design goals
------------
- Keep compute/storage layers free of renderer wording
- Emit exactly one primary diagnosis per window
- Reuse the same decision logic across CLI and dashboard
- Keep messages short, stable, and actionable
"""

from dataclasses import dataclass
from typing import Literal, Mapping, Optional, Sequence

from .schema import StepCombinedTimeMetric

Severity = Literal["info", "warn", "crit"]
DiagnosisKind = Literal[
    "NO_DATA",
    "BALANCED",
    "STRAGGLER",
    "INPUT_BOUND",
    "WAIT_HEAVY",
    "COMPUTE_IMBALANCE",
]


# ---------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class DiagnosisThresholds:
    """
    Thresholds controlling diagnosis selection.

    Notes
    -----
    Shares are ratios relative to `step_time`, since the current UI uses
    step time as the main denominator for quick interpretation.
    """

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


DEFAULT_THRESHOLDS = DiagnosisThresholds()


# ---------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class StepDiagnosis:
    """
    Shared diagnosis object consumed by CLI and dashboard renderers.

    Fields
    ------
    kind:
        Stable machine-friendly category.
    severity:
        One of: "info", "warn", "crit".
    status:
        Short human-facing label.
    reason:
        One short evidence sentence.
    action:
        One short next-step sentence.
    steps_used:
        Number of steps in the diagnosis window.
    worst_rank:
        Rank most associated with the issue, when applicable. In current
        semantics this follows the compute layer's overall-worst-rank meaning.
    note:
        Optional extra caveat for richer surfaces such as the dashboard.
    """

    kind: DiagnosisKind
    severity: Severity
    status: str
    reason: str
    action: str
    steps_used: int
    worst_rank: Optional[int] = None
    note: Optional[str] = None


# ---------------------------------------------------------------------
# Internal normalized views
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSnapshot:
    """
    Small extracted view of a metric used by diagnosis rules.
    """

    name: str
    total: float
    skew: float
    worst_rank: Optional[int]


@dataclass(frozen=True)
class ComputeSignal:
    """
    Candidate compute-side imbalance signal.
    """

    label: str
    total: float
    share: float
    skew: float
    worst_rank: Optional[int]


@dataclass(frozen=True)
class DiagnosisContext:
    """
    Normalized diagnosis inputs derived from metric summaries.
    """

    steps_used: int
    single_rank: bool
    overall_worst_rank: Optional[int]

    step: MetricSnapshot
    dataloader: MetricSnapshot
    wait: MetricSnapshot
    forward: MetricSnapshot
    backward: MetricSnapshot
    optimizer: MetricSnapshot

    dataloader_share: float
    wait_share: float
    dominant_compute: Optional[ComputeSignal]


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def build_step_diagnosis(
    metrics: Sequence[StepCombinedTimeMetric],
    thresholds: DiagnosisThresholds = DEFAULT_THRESHOLDS,
) -> StepDiagnosis:
    """
    Build one primary diagnosis from step-combined metrics.

    Diagnosis priority
    ------------------
    1. STRAGGLER
    2. INPUT_BOUND
    3. WAIT_HEAVY
    4. COMPUTE_IMBALANCE
    5. BALANCED

    Notes
    -----
    - For multi-rank runs, diagnosis shares use median totals.
    - For single-rank runs, diagnosis shares use worst totals
      (effectively the visible total for that run).
    - WAIT diagnosis uses WAIT* directly, because step_time excludes
      dataloader fetch time in the current instrumentation.
    """
    by_key = {metric.metric: metric for metric in metrics}

    step_metric = by_key.get("step_time")
    if step_metric is None:
        return StepDiagnosis(
            kind="NO_DATA",
            severity="info",
            status="NO DATA",
            reason="step_time metric is missing.",
            action="Wait for the first complete window.",
            steps_used=0,
        )

    context = _build_context(by_key, step_metric)
    if context is None:
        return StepDiagnosis(
            kind="NO_DATA",
            severity="info",
            status="NO DATA",
            reason="No usable step-time data yet.",
            action="Wait for the first complete window.",
            steps_used=int(step_metric.summary.steps_used),
            worst_rank=step_metric.summary.worst_rank,
        )

    return (
        _diagnose_straggler(context, thresholds)
        or _diagnose_input_bound(context, thresholds)
        or _diagnose_wait_heavy(context, thresholds)
        or _diagnose_compute_imbalance(context, thresholds)
        or _diagnose_balanced(context)
    )


def format_cli_diagnosis(diagnosis: StepDiagnosis) -> str:
    """
    Render a short Rich-friendly diagnosis block for terminal output.
    """
    status = _styled_status(diagnosis.status, diagnosis.severity)
    return "\n".join(
        [
            f"[bold]Issue:[/bold] {status}",
            f"[bold]Why:[/bold] {diagnosis.reason}",
            f"[bold]Hint:[/bold] {diagnosis.action}",
        ]
    )


def format_dashboard_diagnosis(diagnosis: StepDiagnosis) -> str:
    """
    Render a slightly richer but still compact diagnosis block for dashboard use.
    """
    meta = f"*Window: {diagnosis.steps_used} steps"
    if diagnosis.worst_rank is not None and diagnosis.kind != "BALANCED":
        meta += f" · Worst rank: {_rank_str(diagnosis.worst_rank)}"
    meta += "*"

    out = (
        f"**Status:** {diagnosis.status}  \n"
        f"**Why:** {diagnosis.reason}  \n"
        f"**Next:** {diagnosis.action}  \n"
        f"{meta}"
    )

    if diagnosis.note:
        out += f"  \n*Note:* {diagnosis.note}"

    return out


# ---------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------


def _build_context(
    metrics: Mapping[str, StepCombinedTimeMetric],
    step_metric: StepCombinedTimeMetric,
) -> Optional[DiagnosisContext]:
    """
    Build a normalized diagnosis context from renderer metric summaries.
    """
    coverage = step_metric.coverage
    single_rank = (coverage.world_size <= 1) or (coverage.ranks_present <= 1)
    steps_used = int(step_metric.summary.steps_used)

    # In the current compute layer this is already the UI's overall worst rank.
    overall_worst_rank = step_metric.summary.worst_rank

    step = _snapshot("step_time", metrics.get("step_time"), single_rank)
    if step.total <= 0.0:
        return None

    dataloader = _snapshot(
        "dataloader_fetch",
        metrics.get("dataloader_fetch"),
        single_rank,
    )
    wait = _snapshot("wait_proxy", metrics.get("wait_proxy"), single_rank)
    forward = _snapshot("forward", metrics.get("forward"), single_rank)
    backward = _snapshot("backward", metrics.get("backward"), single_rank)
    optimizer = _snapshot(
        "optimizer_step",
        metrics.get("optimizer_step"),
        single_rank,
    )

    dataloader_share = _share(dataloader.total, step.total)
    wait_share = _share(wait.total, step.total)

    dominant_compute = _pick_dominant_compute(
        step_total=step.total,
        forward=forward,
        backward=backward,
        optimizer=optimizer,
    )

    return DiagnosisContext(
        steps_used=steps_used,
        single_rank=single_rank,
        overall_worst_rank=overall_worst_rank,
        step=step,
        dataloader=dataloader,
        wait=wait,
        forward=forward,
        backward=backward,
        optimizer=optimizer,
        dataloader_share=dataloader_share,
        wait_share=wait_share,
        dominant_compute=dominant_compute,
    )


def _snapshot(
    name: str,
    metric: Optional[StepCombinedTimeMetric],
    single_rank: bool,
) -> MetricSnapshot:
    """
    Convert a metric object into a small stable diagnosis snapshot.
    """
    if metric is None:
        return MetricSnapshot(name=name, total=0.0, skew=0.0, worst_rank=None)

    total = (
        float(metric.summary.worst_total)
        if single_rank
        else float(metric.summary.median_total)
    )
    skew = 0.0 if single_rank else float(metric.summary.skew_pct)

    return MetricSnapshot(
        name=name,
        total=total,
        skew=skew,
        worst_rank=metric.summary.worst_rank,
    )


def _pick_dominant_compute(
    *,
    step_total: float,
    forward: MetricSnapshot,
    backward: MetricSnapshot,
    optimizer: MetricSnapshot,
) -> Optional[ComputeSignal]:
    """
    Pick the strongest compute-side imbalance candidate.

    Selection priority:
    - larger skew first
    - larger share second
    """
    candidates = [
        ComputeSignal(
            label="Forward",
            total=forward.total,
            share=_share(forward.total, step_total),
            skew=forward.skew,
            worst_rank=forward.worst_rank,
        ),
        ComputeSignal(
            label="Backward",
            total=backward.total,
            share=_share(backward.total, step_total),
            skew=backward.skew,
            worst_rank=backward.worst_rank,
        ),
        ComputeSignal(
            label="Optimizer",
            total=optimizer.total,
            share=_share(optimizer.total, step_total),
            skew=optimizer.skew,
            worst_rank=optimizer.worst_rank,
        ),
    ]

    candidates = [
        candidate for candidate in candidates if candidate.total > 0.0
    ]
    if not candidates:
        return None

    return max(
        candidates, key=lambda candidate: (candidate.skew, candidate.share)
    )


# ---------------------------------------------------------------------
# Diagnosis rules
# ---------------------------------------------------------------------


def _diagnose_straggler(
    context: DiagnosisContext,
    thresholds: DiagnosisThresholds,
) -> Optional[StepDiagnosis]:
    """
    Detect full-step imbalance across ranks.

    This uses step-time skew as the primary straggler signal and reports the
    compute-layer overall worst rank for consistency with the CLI/dashboard UI.
    """
    if context.single_rank:
        return None

    if context.step.skew < thresholds.straggler_skew_warn:
        return None

    overall_worst_rank = context.overall_worst_rank
    rank_str = _rank_str(overall_worst_rank)
    severity = _severity(context.step.skew, thresholds.straggler_skew_crit)

    if (
        context.dataloader_share >= 0.15
        and context.dataloader.skew >= thresholds.compute_skew_warn
        and context.dataloader.worst_rank == overall_worst_rank
    ):
        return StepDiagnosis(
            kind="STRAGGLER",
            severity=severity,
            status="STRAGGLER",
            reason=(
                f"Step skew is +{_pct(context.step.skew)}; "
                f"{rank_str} also leads dataloader imbalance."
            ),
            action=f"Check input loading on {rank_str}.",
            steps_used=context.steps_used,
            worst_rank=overall_worst_rank,
        )

    compute = context.dominant_compute
    if (
        compute is not None
        and compute.skew >= thresholds.compute_skew_warn
        and compute.share >= thresholds.compute_share_min
        and compute.worst_rank == overall_worst_rank
    ):
        return StepDiagnosis(
            kind="STRAGGLER",
            severity=severity,
            status="STRAGGLER",
            reason=(
                f"Step skew is +{_pct(context.step.skew)}; "
                f"{compute.label} is most imbalanced on {rank_str}."
            ),
            action=f"Inspect {compute.label.lower()} on {rank_str}.",
            steps_used=context.steps_used,
            worst_rank=overall_worst_rank,
        )

    if context.wait_share >= thresholds.wait_share_warn:
        return StepDiagnosis(
            kind="STRAGGLER",
            severity=severity,
            status="STRAGGLER",
            reason=(
                f"Step skew is +{_pct(context.step.skew)}; "
                f"WAIT* is elevated on {rank_str}."
            ),
            action=f"Inspect sync / CPU stalls on {rank_str}.",
            steps_used=context.steps_used,
            worst_rank=overall_worst_rank,
        )

    return StepDiagnosis(
        kind="STRAGGLER",
        severity=severity,
        status="STRAGGLER",
        reason=(
            f"Step skew is +{_pct(context.step.skew)}; "
            f"worst rank is {rank_str}."
        ),
        action=f"Inspect slower work on {rank_str}.",
        steps_used=context.steps_used,
        worst_rank=overall_worst_rank,
    )


def _diagnose_input_bound(
    context: DiagnosisContext,
    thresholds: DiagnosisThresholds,
) -> Optional[StepDiagnosis]:
    """
    Detect input-bound windows.

    Since step_time excludes dataloader fetch time, the ratio here is interpreted
    as dataloader time relative to measured step time.
    """
    if context.dataloader_share < thresholds.input_share_warn:
        return None

    overall_worst_rank = (
        None if context.single_rank else context.overall_worst_rank
    )
    reason = f"Dataloader is {_pct(context.dataloader_share)} of step time."

    if not context.single_rank and context.dataloader.worst_rank is not None:
        reason = (
            f"Dataloader is {_pct(context.dataloader_share)} of step time; "
            f"worst rank is {_rank_str(context.dataloader.worst_rank)}."
        )

    return StepDiagnosis(
        kind="INPUT_BOUND",
        severity=_severity(
            context.dataloader_share,
            thresholds.input_share_crit,
        ),
        status="INPUT-BOUND",
        reason=reason,
        action="Increase workers, prefetch, or storage throughput.",
        steps_used=context.steps_used,
        worst_rank=overall_worst_rank,
    )


def _diagnose_wait_heavy(
    context: DiagnosisContext,
    thresholds: DiagnosisThresholds,
) -> Optional[StepDiagnosis]:
    """
    Detect elevated non-model residual inside the measured step.

    WAIT* is used directly because step_time excludes dataloader fetch time.
    """
    if context.wait_share < thresholds.wait_share_warn:
        return None

    return StepDiagnosis(
        kind="WAIT_HEAVY",
        severity=_severity(context.wait_share, thresholds.wait_share_crit),
        status="WAIT-HEAVY",
        reason=f"WAIT* is {_pct(context.wait_share)} of step time.",
        action="Inspect sync points, CPU stalls, and H2D copies.",
        steps_used=context.steps_used,
        worst_rank=None if context.single_rank else context.overall_worst_rank,
        note="WAIT* = step_time - (forward + backward + optimizer_step).",
    )


def _diagnose_compute_imbalance(
    context: DiagnosisContext,
    thresholds: DiagnosisThresholds,
) -> Optional[StepDiagnosis]:
    """
    Detect compute-component imbalance that does not dominate full step time.
    """
    if context.single_rank:
        return None

    compute = context.dominant_compute
    if compute is None:
        return None

    if compute.skew < thresholds.compute_skew_warn:
        return None

    if compute.share < thresholds.compute_share_min:
        return None

    note = None
    if context.step.skew < thresholds.low_step_skew:
        note = "Step time stays fairly balanced, so this may be partly hidden."

    return StepDiagnosis(
        kind="COMPUTE_IMBALANCE",
        severity=_severity(compute.skew, thresholds.compute_skew_crit),
        status="COMPUTE-IMBALANCE",
        reason=(
            f"{compute.label} skew is +{_pct(compute.skew)} "
            f"on {_rank_str(compute.worst_rank)}."
        ),
        action=f"Inspect {compute.label.lower()} on {_rank_str(compute.worst_rank)}.",
        steps_used=context.steps_used,
        worst_rank=compute.worst_rank,
        note=note,
    )


def _diagnose_balanced(context: DiagnosisContext) -> StepDiagnosis:
    """
    Fallback diagnosis when no stronger signal is present.
    """
    return StepDiagnosis(
        kind="BALANCED",
        severity="info",
        status="BALANCED",
        reason="No dominant bottleneck is visible in this window.",
        action="Focus on throughput only if overall speed is still low.",
        steps_used=context.steps_used,
        worst_rank=None if context.single_rank else context.overall_worst_rank,
    )


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------


def _share(value: float, total: float) -> float:
    """
    Return a safe non-negative share.
    """
    if total <= 0.0:
        return 0.0
    return max(0.0, float(value) / float(total))


def _pct(value: float) -> str:
    """
    Format a ratio as a percentage string.
    """
    return f"{value * 100:.1f}%"


def _rank_str(rank: Optional[int]) -> str:
    """
    Format a rank identifier for UI text.
    """
    return f"r{rank}" if rank is not None else "—"


def _severity(value: float, crit_threshold: float) -> Severity:
    """
    Map a scalar signal to warning or critical severity.
    """
    return "crit" if value >= crit_threshold else "warn"


def _styled_status(status: str, severity: Severity) -> str:
    """
    Render a colored status label for Rich CLI output.
    """
    style = {
        "crit": "bold red",
        "warn": "bold yellow",
        "info": "bold green",
    }.get(severity, "bold")
    return f"[{style}]{status}[/{style}]"
