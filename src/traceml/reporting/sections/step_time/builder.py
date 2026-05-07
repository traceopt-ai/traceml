"""Payload builder for the final-report step-time section."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from traceml.diagnostics.step_time.adapters import (
    build_summary_step_diagnosis_result,
)
from traceml.reporting.sections.step_time.loader import StepTimeSectionData
from traceml.reporting.sections.step_time.model import (
    RankStepSummary,
    _build_global_rollup,
    _build_overview,
    _closest_rank_to_median,
    _compute_wait_avg_ms,
    _finite_float,
    _rank_entry_to_json,
    _timing_rollup_from_summary,
    _to_rank_signals,
)
from traceml.reporting.summaries.diagnosis_presentation import (
    diagnosis_presentation_to_json,
    present_step_time_summary_diagnosis,
)
from traceml.reporting.summaries.issue_summary import (
    issues_by_metric_json,
    issues_by_rank_json,
    issues_to_json,
)
from traceml.reporting.summaries.summary_formatting import format_ms


@dataclass(frozen=True)
class StepTimeMetricPair:
    """Median and worst-rank values for one timing bucket."""

    median_ms: Optional[float]
    worst_ms: Optional[float]
    median_rank: Optional[int]
    worst_rank: Optional[int]


@dataclass(frozen=True)
class StepTimeCardStats:
    """Compact values used by the human final-summary card."""

    rank_count: int
    step: StepTimeMetricPair
    compute: StepTimeMetricPair
    wait: StepTimeMetricPair
    input: StepTimeMetricPair

    @property
    def is_multi_rank(self) -> bool:
        return self.rank_count > 1


def _rank_label(rank: Optional[int]) -> str:
    """Format an optional rank id for summary-card text."""
    return f"r{int(rank)}" if rank is not None else "n/a"


def _format_ms_pair(
    left: Optional[float],
    right: Optional[float],
) -> str:
    """Format a compact `median/worst` millisecond pair."""
    if left is None or right is None:
        return "n/a"
    return f"{float(left):.1f}/{float(right):.1f}ms"


def _format_rank_pair(
    left: Optional[int],
    right: Optional[int],
) -> str:
    """Format a compact `median/worst` rank pair."""
    return f"{_rank_label(left)}/{_rank_label(right)}"


def _metric_pair_from_rank_values(
    rank_to_value: Dict[int, float],
) -> StepTimeMetricPair:
    """Return median/worst values and the ranks that best represent them."""
    if not rank_to_value:
        return StepTimeMetricPair(None, None, None, None)

    values = np.asarray(
        [_finite_float(value) for value in rank_to_value.values()],
        dtype=np.float64,
    )
    if values.size == 0:
        return StepTimeMetricPair(None, None, None, None)

    median_value = float(np.median(values))
    median_rank = _closest_rank_to_median(rank_to_value)
    worst_rank = max(
        rank_to_value,
        key=lambda rank: (_finite_float(rank_to_value[rank]), -int(rank)),
    )
    return StepTimeMetricPair(
        median_ms=median_value,
        worst_ms=_finite_float(rank_to_value[worst_rank]),
        median_rank=median_rank,
        worst_rank=int(worst_rank),
    )


def _build_card_stats(
    per_rank_summary: Dict[int, RankStepSummary],
) -> Optional[StepTimeCardStats]:
    """Build the timing values rendered in the final summary card."""
    if not per_rank_summary:
        return None

    return StepTimeCardStats(
        rank_count=len(per_rank_summary),
        step=_metric_pair_from_rank_values(
            {
                int(rank): _finite_float(summary.avg_total_step_ms)
                for rank, summary in per_rank_summary.items()
            }
        ),
        compute=_metric_pair_from_rank_values(
            {
                int(rank): _finite_float(summary.avg_gpu_compute_ms)
                for rank, summary in per_rank_summary.items()
            }
        ),
        wait=_metric_pair_from_rank_values(
            {
                int(rank): _compute_wait_avg_ms(summary)
                for rank, summary in per_rank_summary.items()
            }
        ),
        input=_metric_pair_from_rank_values(
            {
                int(rank): _finite_float(summary.avg_dataloader_ms)
                for rank, summary in per_rank_summary.items()
            }
        ),
    )


def _format_card_stats(stats: StepTimeCardStats) -> str:
    """Render the compact Step Time `Stats` line."""
    if stats.is_multi_rank:
        step = _format_ms_pair(stats.step.median_ms, stats.step.worst_ms)
        compute = _format_ms_pair(
            stats.compute.median_ms,
            stats.compute.worst_ms,
        )
        wait = _format_ms_pair(stats.wait.median_ms, stats.wait.worst_ms)
        input_ms = _format_ms_pair(
            stats.input.median_ms,
            stats.input.worst_ms,
        )
        return (
            "- Stats: median/worst | "
            f"step {step} | compute {compute} | "
            f"wait {wait} | input {input_ms}"
        )

    return (
        "- Stats: "
        f"step {format_ms(stats.step.worst_ms)} | "
        f"compute {format_ms(stats.compute.worst_ms)} | "
        f"wait {format_ms(stats.wait.worst_ms)} | "
        f"input {format_ms(stats.input.worst_ms)}"
    )


def _format_card_ranks(stats: StepTimeCardStats) -> Optional[str]:
    """Render the compact Step Time `Ranks` line for distributed runs."""
    if not stats.is_multi_rank:
        return None
    step = _format_rank_pair(stats.step.median_rank, stats.step.worst_rank)
    compute = _format_rank_pair(
        stats.compute.median_rank,
        stats.compute.worst_rank,
    )
    wait = _format_rank_pair(stats.wait.median_rank, stats.wait.worst_rank)
    input_rank = _format_rank_pair(
        stats.input.median_rank,
        stats.input.worst_rank,
    )
    return (
        "- Ranks: median/worst | "
        f"step {step} | compute {compute} | "
        f"wait {wait} | input {input_rank}"
    )


def _largest_compute_phase(
    summary: Optional[RankStepSummary],
) -> Optional[str]:
    """Return the largest compute bucket for one rank summary."""
    if summary is None:
        return None
    values = {
        "forward": _finite_float(summary.avg_forward_ms),
        "backward": _finite_float(summary.avg_backward_ms),
        "optimizer": _finite_float(summary.avg_optimizer_ms),
    }
    return max(values, key=values.get) if values else None


def _step_time_card_reason(
    diagnosis: Optional[Any],
    *,
    stats: Optional[StepTimeCardStats],
    per_rank_summary: Dict[int, RankStepSummary],
) -> str:
    """Build the short `Why` line used only by the human card."""
    kind = str(getattr(diagnosis, "kind", "") or "")
    if diagnosis is None or kind == "NO_DATA":
        return "Need more step-time samples."
    if stats is None:
        return str(getattr(diagnosis, "reason", "") or "n/a")

    if kind == "BALANCED":
        return "No clear timing bottleneck."
    if kind == "INPUT_STRAGGLER":
        evidence = _format_ms_pair(stats.input.worst_ms, stats.input.median_ms)
        return (
            f"{_rank_label(stats.input.worst_rank)} input was slower than "
            f"median rank ({evidence})."
        )
    if kind == "COMPUTE_STRAGGLER":
        evidence = _format_ms_pair(
            stats.compute.worst_ms,
            stats.compute.median_ms,
        )
        return (
            f"{_rank_label(stats.compute.worst_rank)} compute was slower than "
            f"median rank ({evidence})."
        )
    if kind == "STRAGGLER":
        return "Input and compute varied across ranks."
    if kind == "INPUT_BOUND":
        evidence = (
            f"{format_ms(stats.input.worst_ms)}/"
            f"{format_ms(stats.step.worst_ms)}"
        )
        return f"Input loading took a large share ({evidence})."
    if kind == "WAIT_HEAVY":
        evidence = (
            f"{format_ms(stats.wait.worst_ms)}/"
            f"{format_ms(stats.step.worst_ms)}"
        )
        return f"Wait time was high ({evidence})."
    if kind == "COMPUTE_BOUND":
        summary = per_rank_summary.get(stats.compute.worst_rank)
        phase = _largest_compute_phase(summary)
        suffix = f"; {phase} was largest" if phase else ""
        evidence = (
            f"{format_ms(stats.compute.worst_ms)}/"
            f"{format_ms(stats.step.worst_ms)}"
        )
        return f"Compute dominated ({evidence}){suffix}."

    reason = str(getattr(diagnosis, "reason", "") or "").strip()
    return reason or "No clear timing bottleneck."


def build_step_time_card(
    *,
    training_steps: int,
    latest_step_observed: Optional[int],
    per_rank_summary: Dict[int, RankStepSummary],
    per_rank_step_metrics: Dict[int, Dict[int, Dict[str, float]]],
    max_rows: int,
) -> tuple[str, Dict[str, Any]]:
    """Build the Step Time section payload and compact card text."""
    ranks_present = sorted(per_rank_summary.keys())
    overview = _build_overview(per_rank_summary=per_rank_summary)

    representative_rank = overview["representative_rank"]
    worst_rank = overview["worst_rank"]
    representative_summary = (
        per_rank_summary.get(representative_rank)
        if representative_rank is not None
        else None
    )
    worst_summary = (
        per_rank_summary.get(worst_rank) if worst_rank is not None else None
    )
    primary_summary = representative_summary or worst_summary

    summary_diag_result = build_summary_step_diagnosis_result(
        rank_signals=_to_rank_signals(per_rank_summary),
        max_rows=max_rows,
        per_rank_step_metrics=per_rank_step_metrics,
    )
    summary_diag = (
        summary_diag_result.primary
        if summary_diag_result is not None
        else None
    )
    summary_diag_presented = present_step_time_summary_diagnosis(summary_diag)
    issues = summary_diag_result.issues if summary_diag_result else ()
    issues_by_rank, unassigned_issues = issues_by_rank_json(
        issues,
        rank_keys=ranks_present,
    )
    issues_by_metric, metric_unassigned = issues_by_metric_json(issues)

    global_rollup = _build_global_rollup(
        per_rank_summary=per_rank_summary,
        representative_rank=representative_rank,
        bottleneck_rank=worst_rank,
        imbalance_gap_pct=overview["worst_vs_representative_pct"],
    )
    card_stats = _build_card_stats(per_rank_summary)
    primary_rollup = (
        _timing_rollup_from_summary(primary_summary)
        if primary_summary is not None
        else _timing_rollup_from_summary(None)
    )

    steps_analyzed_by_rank = {
        str(rank): int(s.steps_analyzed)
        for rank, s in sorted(per_rank_summary.items())
    }
    analyzed_counts = list(steps_analyzed_by_rank.values())
    min_steps_analyzed = min(analyzed_counts) if analyzed_counts else 0
    max_steps_analyzed = max(analyzed_counts) if analyzed_counts else 0

    title = (
        f"TraceML Step Timing Summary | steps {training_steps} | "
        f"ranks {len(ranks_present)}"
    )
    lines = [title, "Step Time"]
    diagnosis_status = (
        summary_diag_presented.status
        if summary_diag_presented is not None
        else "NO DATA"
    )
    diagnosis_why = _step_time_card_reason(
        summary_diag,
        stats=card_stats,
        per_rank_summary=per_rank_summary,
    )

    if not per_rank_summary:
        latest_step_text = latest_step_observed or "n/a"
        lines.extend(
            [
                f"- Diagnosis: {diagnosis_status}",
                f"- Scope: latest step {latest_step_text}",
                "- Stats: n/a",
                f"- Why: {diagnosis_why}",
            ]
        )
    elif len(ranks_present) == 1 and primary_summary is not None:
        only_rank = ranks_present[0]
        lines.extend(
            [
                f"- Diagnosis: {diagnosis_status}",
                (
                    f"- Scope: last {primary_summary.steps_analyzed} "
                    f"steps on rank r{only_rank}"
                ),
            ]
        )
        if card_stats is not None:
            lines.append(_format_card_stats(card_stats))
        lines.append(f"- Why: {diagnosis_why}")
    else:
        analyzed_text = (
            f"last {max_steps_analyzed} steps per rank"
            if min_steps_analyzed == max_steps_analyzed
            else f"last {min_steps_analyzed}-{max_steps_analyzed} steps per rank"
        )
        lines.append(f"- Diagnosis: {diagnosis_status}")
        lines.append(f"- Scope: compared over {analyzed_text}")
        if card_stats is not None:
            lines.append(_format_card_stats(card_stats))
            ranks_line = _format_card_ranks(card_stats)
            if ranks_line:
                lines.append(ranks_line)
        lines.append(f"- Why: {diagnosis_why}")

    card = "\n".join(lines)
    per_rank_json = {
        str(rank): {
            **_rank_entry_to_json(rank, s),
            "issues": issues_by_rank.get(str(rank), []),
        }
        for rank, s in sorted(per_rank_summary.items())
    }

    summary = {
        "overview": {
            "mode": overview["mode"],
            "training_steps": training_steps,
            "latest_step_observed": latest_step_observed,
            "ranks_seen": len(ranks_present),
            "max_steps_analyzed_per_rank": int(max_rows),
            "steps_used_primary": int(primary_rollup["steps_analyzed"]),
            "steps_analyzed_min": int(min_steps_analyzed),
            "steps_analyzed_max": int(max_steps_analyzed),
            "steps_analyzed_per_rank": steps_analyzed_by_rank,
        },
        "primary_diagnosis": diagnosis_presentation_to_json(
            summary_diag_presented,
            include_action=False,
        ),
        "issues": issues_to_json(issues),
        "issues_by_rank": issues_by_rank,
        "issues_by_metric": issues_by_metric,
        "unassigned_issues": unassigned_issues + metric_unassigned,
        "global": global_rollup,
        "per_rank": per_rank_json,
        "units": {"time": "ms"},
        "card": card,
    }
    return card, summary


def build_step_time_section_payload(
    data: StepTimeSectionData,
) -> Dict[str, Any]:
    """Build the JSON-safe step-time section payload from loaded data."""
    _, payload = build_step_time_card(
        training_steps=data.training_steps,
        latest_step_observed=data.latest_step_observed,
        per_rank_summary=data.per_rank_summary,
        per_rank_step_metrics=data.per_rank_step_metrics,
        max_rows=data.max_rows,
    )
    return payload


__all__ = [
    "StepTimeCardStats",
    "StepTimeMetricPair",
    "build_step_time_card",
    "build_step_time_section_payload",
]
