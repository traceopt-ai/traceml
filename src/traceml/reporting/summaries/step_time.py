"""
Compact end-of-run step-time summary generation.

This module reads `step_time_samples`, builds per-rank timing summaries, and
produces:

1. a compact text summary for end-of-run display and sharing
2. a structured JSON payload for automation, comparison, and future dashboards

Design goals
------------
- Keep the printed summary concise and easy to scan
- Make future extension straightforward for:
  - richer global rollups
  - per-rank detail
  - optional per-device / per-GPU detail when the data model supports it

Notes
-----
- The printed text intentionally remains compact.
- The JSON summary is the richer source of truth for downstream systems.
- The current persistent step-time data is rank-based, so this module exposes
  `per_rank` as the canonical detailed dimension today.
- A `per_device` extension point is included, but it is intentionally empty
  until canonical step-time rows carry device identity reliably.
"""

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from traceml.reporting.summaries.diagnosis_presentation import (
    diagnosis_presentation_to_json,
    present_step_time_summary_diagnosis,
)
from traceml.reporting.summaries.step_time_diagnosis import (
    RankStepSignals,
    build_summary_step_diagnosis_result,
    diagnosis_result_to_json,
    diagnosis_to_json,
)
from traceml.reporting.summaries.summary_formatting import (
    format_ms,
    format_percent,
    safe_float,
    share_percent,
)
from traceml.reporting.summaries.summary_io import (
    append_text,
    load_json_or_empty,
    write_json,
)


def _finite_float(x: Any) -> float:
    """Convert to float; coerce non-finite values to 0.0."""
    v = safe_float(x)
    return v if np.isfinite(v) else 0.0


def _event_total_ms(by_dev: Any) -> float:
    """
    Sum duration_ms across all devices for one event.

    Expected restricted JSON shape:
        {
            "<device>": {
                "is_gpu": bool | null,
                "duration_ms": float | null,
                "n_calls": int | null
            }
        }
    """
    if not isinstance(by_dev, dict):
        return 0.0

    total = 0.0
    for stats in by_dev.values():
        if not isinstance(stats, dict):
            continue
        total += _finite_float(stats.get("duration_ms"))
    return total


def _event_bucket(name: str) -> Optional[str]:
    """
    Map raw event names to canonical step-time buckets.

    Returns one of:
      - dataloader
      - forward
      - backward
      - optimizer
      - step_time
      - None
    """
    n = str(name).lower()

    if "step_time" in n:
        return "step_time"
    if "dataloader_next" in n:
        return "dataloader"
    if "forward_time" in n:
        return "forward"
    if "backward_time" in n:
        return "backward"
    if "optimizer_step" in n:
        return "optimizer"

    if "data" in n or "dataloader" in n or "input" in n or "batch" in n:
        return "dataloader"
    if "forward" in n or n == "fwd":
        return "forward"
    if "backward" in n or "bwd" in n:
        return "backward"
    if "optim" in n or "optimizer" in n or n in {"step", "update"}:
        return "optimizer"

    return None


def _closest_rank_to_median(rank_to_value: Dict[int, float]) -> Optional[int]:
    """
    Return the rank whose value is closest to the median of all values.

    Tie-breaker order:
    1) closest absolute distance to median
    2) smaller metric value
    3) smaller rank id
    """
    if not rank_to_value:
        return None

    vals = np.asarray(
        [_finite_float(v) for v in rank_to_value.values()],
        dtype=np.float64,
    )
    if vals.size == 0:
        return None

    median_val = float(np.median(vals))

    return min(
        rank_to_value.keys(),
        key=lambda r: (
            abs(_finite_float(rank_to_value[r]) - median_val),
            _finite_float(rank_to_value[r]),
            r,
        ),
    )


@dataclass
class RankStepSummary:
    """
    Per-rank averaged step-time summary over the analyzed step window.
    """

    steps_analyzed: int
    avg_dataloader_ms: float
    avg_forward_ms: float
    avg_backward_ms: float
    avg_optimizer_ms: float
    avg_step_cpu_ms: float
    avg_gpu_compute_ms: float
    avg_total_step_ms: float


@dataclass
class RankStepAnalysis:
    """
    Per-rank summary plus per-step canonical metrics.

    `per_step_metrics` shape:
      step -> {
        "dataloader_fetch": ms,
        "forward": ms,
        "backward": ms,
        "optimizer_step": ms,
        "step_time": ms,
        "wait_proxy": ms,
      }
    """

    summary: RankStepSummary
    per_step_metrics: Dict[int, Dict[str, float]]


def _to_rank_signals(
    per_rank_summary: Dict[int, RankStepSummary],
) -> Dict[int, RankStepSignals]:
    """
    Convert local rank summaries to diagnosis adapter input objects.
    """
    return {
        int(rank): RankStepSignals(
            steps_analyzed=int(s.steps_analyzed),
            dataloader_ms=_finite_float(s.avg_dataloader_ms),
            forward_ms=_finite_float(s.avg_forward_ms),
            backward_ms=_finite_float(s.avg_backward_ms),
            optimizer_ms=_finite_float(s.avg_optimizer_ms),
            step_cpu_ms=_finite_float(s.avg_step_cpu_ms),
        )
        for rank, s in per_rank_summary.items()
    }


def _load_rank_step_rows(
    conn: sqlite3.Connection,
    *,
    rank: int,
    max_rows: int,
) -> list[Dict[str, Any]]:
    """
    Load the latest up to `max_rows` step-time rows for one rank.

    Assumes one projected row per step in `step_time_samples`.
    """
    cur = conn.execute(
        """
        SELECT step, events_json
        FROM step_time_samples
        WHERE rank = ?
        ORDER BY step DESC, id DESC
        LIMIT ?;
        """,
        (int(rank), int(max_rows)),
    )

    rows: list[Dict[str, Any]] = []
    for step, events_json in cur:
        if step is None or not events_json:
            continue
        try:
            events = json.loads(events_json)
        except Exception:
            continue
        if not isinstance(events, dict):
            continue
        rows.append({"step": int(step), "events": events})
    return rows


def _row_metrics(events: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Convert one step's event map into canonical timing buckets.

    Returns
    -------
    dict with keys:
      - dataloader
      - forward
      - backward
      - optimizer
      - step_time

    or None if nothing usable was found.
    """
    metrics = {
        "dataloader": 0.0,
        "forward": 0.0,
        "backward": 0.0,
        "optimizer": 0.0,
        "step_time": 0.0,
    }

    for evt_name, by_dev in events.items():
        bucket = _event_bucket(str(evt_name))
        if bucket is None:
            continue
        metrics[bucket] += _event_total_ms(by_dev)

    if (
        metrics["dataloader"] <= 0.0
        and metrics["forward"] <= 0.0
        and metrics["backward"] <= 0.0
        and metrics["optimizer"] <= 0.0
        and metrics["step_time"] <= 0.0
    ):
        return None

    return metrics


def _build_rank_summary(
    step_rows: list[Dict[str, Any]],
) -> Optional[RankStepAnalysis]:
    """
    Build a per-rank summary and per-step canonical metrics over provided rows.

    For each step:
        gpu_compute = forward + backward + optimizer
        step_effective = max(step_time, gpu_compute)
        total_step = dataloader + step_effective
        wait_proxy = max(0, step_effective - gpu_compute)
    """
    if not step_rows:
        return None

    sum_dl = 0.0
    sum_fwd = 0.0
    sum_bwd = 0.0
    sum_opt = 0.0
    sum_step_cpu = 0.0
    sum_total = 0.0
    n = 0

    per_step_metrics: Dict[int, Dict[str, float]] = {}

    for row in step_rows:
        step_id = row.get("step")
        metrics = _row_metrics(row["events"])
        if metrics is None or step_id is None:
            continue

        dl = _finite_float(metrics["dataloader"])
        fwd = _finite_float(metrics["forward"])
        bwd = _finite_float(metrics["backward"])
        opt = _finite_float(metrics["optimizer"])
        step_cpu = _finite_float(metrics["step_time"])

        gpu_compute = fwd + bwd + opt
        step_effective = max(step_cpu, gpu_compute)
        wait_proxy = max(0.0, step_effective - gpu_compute)
        total_step = dl + step_effective

        per_step_metrics[int(step_id)] = {
            "dataloader_fetch": dl,
            "forward": fwd,
            "backward": bwd,
            "optimizer_step": opt,
            "step_time": step_effective,
            "wait_proxy": wait_proxy,
        }

        sum_dl += dl
        sum_fwd += fwd
        sum_bwd += bwd
        sum_opt += opt
        sum_step_cpu += step_cpu
        sum_total += total_step
        n += 1

    if n == 0:
        return None

    summary = RankStepSummary(
        steps_analyzed=n,
        avg_dataloader_ms=sum_dl / n,
        avg_forward_ms=sum_fwd / n,
        avg_backward_ms=sum_bwd / n,
        avg_optimizer_ms=sum_opt / n,
        avg_step_cpu_ms=sum_step_cpu / n,
        avg_gpu_compute_ms=(sum_fwd + sum_bwd + sum_opt) / n,
        avg_total_step_ms=sum_total / n,
    )
    return RankStepAnalysis(summary=summary, per_step_metrics=per_step_metrics)


def _split_ms(s: RankStepSummary) -> Dict[str, float]:
    """Return the main timing split in milliseconds for one rank summary."""
    return {
        "dataloader": s.avg_dataloader_ms,
        "forward": s.avg_forward_ms,
        "backward": s.avg_backward_ms,
        "optimizer": s.avg_optimizer_ms,
    }


def _split_pct(s: RankStepSummary) -> Dict[str, Optional[float]]:
    """Return the main timing split as percentage share of average total step."""
    return {
        "dataloader": share_percent(s.avg_dataloader_ms, s.avg_total_step_ms),
        "forward": share_percent(s.avg_forward_ms, s.avg_total_step_ms),
        "backward": share_percent(s.avg_backward_ms, s.avg_total_step_ms),
        "optimizer": share_percent(s.avg_optimizer_ms, s.avg_total_step_ms),
    }


def _compute_wait_avg_ms(s: RankStepSummary) -> float:
    """
    Return average wait proxy for one rank summary.

    Wait is derived from:
        total_step - (dataloader + forward + backward + optimizer)
    """
    return max(
        0.0,
        _finite_float(s.avg_total_step_ms)
        - (
            _finite_float(s.avg_dataloader_ms)
            + _finite_float(s.avg_forward_ms)
            + _finite_float(s.avg_backward_ms)
            + _finite_float(s.avg_optimizer_ms)
        ),
    )


def _dominant_bucket(split_ms: Dict[str, float]) -> Optional[str]:
    """Return the timing bucket with the largest ms contribution."""
    if not split_ms:
        return None
    return max(split_ms, key=split_ms.get)


def _dominant_line(
    median_split_ms: Optional[Dict[str, float]],
    worst_split_ms: Optional[Dict[str, float]],
) -> str:
    """
    Build one concise dominant-phase takeaway line.
    """
    if median_split_ms is None and worst_split_ms is None:
        return "n/a"

    if median_split_ms is not None and worst_split_ms is None:
        dom = _dominant_bucket(median_split_ms)
        return f"{dom} is the largest phase"

    if median_split_ms is None and worst_split_ms is not None:
        dom = _dominant_bucket(worst_split_ms)
        return f"{dom} is the largest phase"

    dom_median = _dominant_bucket(median_split_ms or {})
    dom_worst = _dominant_bucket(worst_split_ms or {})

    if dom_median == dom_worst:
        return (
            f"{dom_median} is the largest phase on both median and worst rank"
        )

    return (
        f"median rank is dominated by {dom_median}; "
        f"worst rank is dominated by {dom_worst}"
    )


def _empty_timing_rollup() -> Dict[str, Any]:
    """
    Return an empty timing rollup block with stable keys.

    Using a stable empty structure makes downstream consumers simpler and keeps
    missing-data handling explicit rather than forcing callers to guard every
    field access.
    """
    return {
        "steps_analyzed": 0,
        "step_avg_ms": None,
        "compute_avg_ms": None,
        "compute_share_pct": None,
        "wait_avg_ms": None,
        "wait_share_pct": None,
        "split_ms": None,
        "split_pct": None,
        "dominant_phase": None,
    }


def _timing_rollup_from_summary(
    summary: Optional[RankStepSummary],
) -> Dict[str, Any]:
    """
    Build one canonical timing rollup from a per-rank summary.

    The chosen metrics are intentionally conservative and user-oriented:
    - average milliseconds per phase over the analyzed window
    - percentage share of the average total step
    - dominant phase
    - derived wait proxy

    This avoids noisy sums and preserves the fields most useful for diagnosis,
    run-to-run comparison, and future dashboards.
    """
    if summary is None:
        return _empty_timing_rollup()

    split_ms = _split_ms(summary)
    split_pct = _split_pct(summary)
    step_avg_ms = _finite_float(summary.avg_total_step_ms)
    compute_avg_ms = _finite_float(summary.avg_gpu_compute_ms)
    wait_avg_ms = _compute_wait_avg_ms(summary)

    return {
        "steps_analyzed": int(summary.steps_analyzed),
        "step_avg_ms": step_avg_ms,
        "compute_avg_ms": compute_avg_ms,
        "compute_share_pct": share_percent(compute_avg_ms, step_avg_ms),
        "wait_avg_ms": wait_avg_ms,
        "wait_share_pct": share_percent(wait_avg_ms, step_avg_ms),
        "split_ms": split_ms,
        "split_pct": split_pct,
        "dominant_phase": _dominant_bucket(split_ms),
    }


def _rank_entry_to_json(
    rank: int,
    summary: RankStepSummary,
) -> Dict[str, Any]:
    """
    Serialize one rank summary into the canonical machine-readable structure.

    Device identity is intentionally nullable today because the current
    `step_time_samples` projection is rank-based and does not reliably persist
    canonical GPU identity for end-of-run timing summaries.
    """
    return {
        "identity": {
            "rank": int(rank),
            "local_rank": None,
            "gpu_idx": None,
        },
        "timing": _timing_rollup_from_summary(summary),
    }


def _timing_rollup_with_rank(
    rank: Optional[int],
    summary: Optional[RankStepSummary],
) -> Dict[str, Any]:
    """
    Attach rank identity to a timing rollup block.
    """
    rollup = _timing_rollup_from_summary(summary)
    rollup["rank"] = int(rank) if rank is not None else None
    return rollup


def _build_global_rollup(
    *,
    per_rank_summary: Dict[int, RankStepSummary],
    representative_rank: Optional[int],
    bottleneck_rank: Optional[int],
    imbalance_gap_pct: Optional[float],
) -> Dict[str, Any]:
    """
    Build the canonical top-level step-time rollup for the analyzed run window.

    Semantics
    ---------
    - `typical` is the representative rank for the run window. In distributed
      runs this is the rank closest to the median average step time. In
      single-rank runs it is the only available rank.
    - `bottleneck` is the slowest rank in the analyzed window.
    - `imbalance_gap_pct` captures worst-vs-typical spread and is one of the
      most useful signals for distributed straggler diagnosis.

    Notes
    -----
    This is intentionally rank-based today. A future `per_device` view can be
    added cleanly once canonical step-time rows include reliable device
    identity.
    """
    if not per_rank_summary:
        return {
            "mode": "no_data",
            "ranks_seen": 0,
            "representative_rank": None,
            "bottleneck_rank": None,
            "imbalance_gap_pct": None,
            "typical": _empty_timing_rollup(),
            "bottleneck": _empty_timing_rollup(),
        }

    representative_summary = (
        per_rank_summary.get(representative_rank)
        if representative_rank is not None
        else None
    )
    bottleneck_summary = (
        per_rank_summary.get(bottleneck_rank)
        if bottleneck_rank is not None
        else None
    )

    return {
        "mode": (
            "single_rank" if len(per_rank_summary) <= 1 else "distributed"
        ),
        "ranks_seen": len(per_rank_summary),
        "representative_rank": representative_rank,
        "bottleneck_rank": bottleneck_rank,
        "imbalance_gap_pct": imbalance_gap_pct,
        "typical": _timing_rollup_from_summary(representative_summary),
        "bottleneck": _timing_rollup_from_summary(bottleneck_summary),
    }


def _build_overview(
    *,
    per_rank_summary: Dict[int, RankStepSummary],
) -> Dict[str, Any]:
    """
    Build high-level comparison-friendly overview fields from per-rank timing
    summaries.

    Notes
    -----
    The overview intentionally centers on representative-vs-bottleneck behavior
    rather than cross-rank mean values. For distributed training, bottlenecks
    and imbalance matter more than arithmetic averages across ranks.
    """
    if not per_rank_summary:
        return {
            "mode": "no_data",
            "representative_rank": None,
            "worst_rank": None,
            "representative_avg_step_ms": None,
            "worst_avg_step_ms": None,
            "worst_vs_representative_pct": None,
        }

    avg_total_by_rank = {
        rank: s.avg_total_step_ms for rank, s in per_rank_summary.items()
    }
    worst_rank = max(avg_total_by_rank, key=avg_total_by_rank.get)
    representative_rank = _closest_rank_to_median(avg_total_by_rank)

    worst_avg_step_ms = avg_total_by_rank.get(worst_rank)
    representative_avg_step_ms = (
        avg_total_by_rank.get(representative_rank)
        if representative_rank is not None
        else None
    )

    worst_vs_representative_pct = None
    if (
        worst_avg_step_ms is not None
        and representative_avg_step_ms is not None
        and representative_avg_step_ms > 0.0
        and worst_rank != representative_rank
    ):
        worst_vs_representative_pct = (
            100.0
            * (worst_avg_step_ms - representative_avg_step_ms)
            / representative_avg_step_ms
        )

    return {
        "mode": (
            "single_rank" if len(per_rank_summary) <= 1 else "distributed"
        ),
        "representative_rank": representative_rank,
        "worst_rank": worst_rank,
        "representative_avg_step_ms": representative_avg_step_ms,
        "worst_avg_step_ms": worst_avg_step_ms,
        "worst_vs_representative_pct": worst_vs_representative_pct,
    }


def _build_step_time_card(
    *,
    training_steps: int,
    latest_step_observed: Optional[int],
    per_rank_summary: Dict[int, RankStepSummary],
    per_rank_step_metrics: Dict[int, Dict[int, Dict[str, float]]],
    max_rows: int,
) -> tuple[str, Dict[str, Any]]:
    """
    Build a compact, shareable end-of-run step-time summary.

    Printed output is intentionally concise:
    - one scope line
    - one timing line
    - optional distributed comparison lines
    - one dominant-phase takeaway
    - diagnosis + next action

    The JSON payload is richer than the printed text and is the canonical
    machine-readable representation for compare, logging, and dashboards.

    Schema notes
    ------------
    The canonical structured blocks are:
    - `overview`
    - `global_rollup`
    - `rank_rollup`
    - `per_rank`
    - `per_device` (reserved for future extension)
    - `diagnosis`

    Compatibility notes
    -------------------
    `timing_primary`, `timing_median_rank`, and `timing_worst_rank` are kept so
    existing compare and rendering logic remain stable while the internal schema
    becomes cleaner.
    """
    ranks_present = sorted(per_rank_summary.keys())
    overview = _build_overview(per_rank_summary=per_rank_summary)

    representative_rank = overview["representative_rank"]
    worst_rank = overview["worst_rank"]
    representative_avg_step_ms = overview["representative_avg_step_ms"]
    worst_avg_step_ms = overview["worst_avg_step_ms"]
    worst_vs_representative_pct = overview["worst_vs_representative_pct"]

    representative_summary = (
        per_rank_summary.get(representative_rank)
        if representative_rank is not None
        else None
    )
    worst_summary = (
        per_rank_summary.get(worst_rank) if worst_rank is not None else None
    )

    primary_summary = (
        representative_summary
        if representative_summary is not None
        else worst_summary
    )

    representative_split_ms = (
        _split_ms(representative_summary)
        if representative_summary is not None
        else None
    )
    worst_split_ms = (
        _split_ms(worst_summary) if worst_summary is not None else None
    )
    representative_split_pct = (
        _split_pct(representative_summary)
        if representative_summary is not None
        else None
    )
    worst_split_pct = (
        _split_pct(worst_summary) if worst_summary is not None else None
    )

    dominant_text = _dominant_line(
        representative_split_ms,
        worst_split_ms,
    )

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

    primary_rollup = _timing_rollup_from_summary(primary_summary)
    representative_rollup = _timing_rollup_with_rank(
        representative_rank,
        representative_summary,
    )
    worst_rollup = _timing_rollup_with_rank(worst_rank, worst_summary)

    global_rollup = _build_global_rollup(
        per_rank_summary=per_rank_summary,
        representative_rank=representative_rank,
        bottleneck_rank=worst_rank,
        imbalance_gap_pct=worst_vs_representative_pct,
    )

    lines = [
        f"TraceML Step Timing Summary | steps {training_steps} | ranks {len(ranks_present)}",
        "Step Time",
    ]

    if not per_rank_summary:
        lines.extend(
            [
                f"- Scope: latest step {latest_step_observed if latest_step_observed is not None else 'n/a'}",
                "- Step avg: n/a",
                "- Split: n/a",
                "- Dominant: n/a",
            ]
        )
    elif len(ranks_present) == 1 and primary_summary is not None:
        only_rank = ranks_present[0]
        only_split_ms = _split_ms(primary_summary)

        lines.extend(
            [
                f"- Scope: last {primary_summary.steps_analyzed} steps on rank r{only_rank}",
                f"- Step avg: {format_ms(primary_summary.avg_total_step_ms)}",
                (
                    f"- Split: DL {format_ms(only_split_ms['dataloader'])} | "
                    f"FWD {format_ms(only_split_ms['forward'])} | "
                    f"BWD {format_ms(only_split_ms['backward'])} | "
                    f"OPT {format_ms(only_split_ms['optimizer'])} | "
                    f"WAIT {format_ms(primary_rollup['wait_avg_ms'])}"
                ),
                f"- Dominant: {_dominant_line(only_split_ms, None)}",
            ]
        )
    else:
        lines.extend(
            [
                (
                    f"- Scope: ranks {len(ranks_present)} | compared over last up to {int(max_rows)} steps per rank"
                ),
                (
                    f"- Step avg: median r{representative_rank} {format_ms(representative_avg_step_ms)} | "
                    f"worst r{worst_rank} {format_ms(worst_avg_step_ms)} | "
                    f"gap {format_percent(worst_vs_representative_pct)}"
                ),
            ]
        )

        if representative_split_ms is not None:
            lines.append(
                (
                    f"- Median split: DL {format_ms(representative_split_ms['dataloader'])} | "
                    f"FWD {format_ms(representative_split_ms['forward'])} | "
                    f"BWD {format_ms(representative_split_ms['backward'])} | "
                    f"OPT {format_ms(representative_split_ms['optimizer'])} | "
                    f"WAIT {format_ms(representative_rollup['wait_avg_ms'])}"
                )
            )

        if worst_split_ms is not None:
            lines.append(
                (
                    f"- Worst split: DL {format_ms(worst_split_ms['dataloader'])} | "
                    f"FWD {format_ms(worst_split_ms['forward'])} | "
                    f"BWD {format_ms(worst_split_ms['backward'])} | "
                    f"OPT {format_ms(worst_split_ms['optimizer'])} | "
                    f"WAIT {format_ms(worst_rollup['wait_avg_ms'])}"
                )
            )

        lines.append(f"- Dominant: {dominant_text}")

    if summary_diag_presented is not None:
        lines.append(f"- Diagnosis: {summary_diag_presented.status}")
        lines.append(f"- Why: {summary_diag_presented.reason}")
        lines.append(f"- Next: {summary_diag_presented.action}")
        if summary_diag_presented.note:
            lines.append(f"- Note: {summary_diag_presented.note}")

    card = "\n".join(lines)

    summary = {
        # Existing top-level fields kept for compatibility and to minimize risk.
        "training_steps": training_steps,
        "latest_step_observed": latest_step_observed,
        "ranks_seen": len(ranks_present),
        "max_steps_analyzed_per_rank": int(max_rows),
        "worst_rank": worst_rank,
        "median_rank": representative_rank,
        "worst_vs_median_pct": worst_vs_representative_pct,
        "worst_avg_step_ms": worst_avg_step_ms,
        "median_avg_step_ms": representative_avg_step_ms,
        "dominant_text": dominant_text,
        "median_split_ms": representative_split_ms,
        "worst_split_ms": worst_split_ms,
        "median_split_pct": representative_split_pct,
        "worst_split_pct": worst_split_pct,
        "diagnosis": diagnosis_to_json(summary_diag),
        "diagnosis_result": diagnosis_result_to_json(summary_diag_result),
        "diagnosis_presented": diagnosis_presentation_to_json(
            summary_diag_presented
        ),
        "timing_primary": primary_rollup,
        "timing_median_rank": representative_rollup,
        "timing_worst_rank": worst_rollup,
        # Canonical structured fields.
        "overview": {
            "mode": overview["mode"],
            "training_steps": training_steps,
            "latest_step_observed": latest_step_observed,
            "ranks_seen": len(ranks_present),
            "max_steps_analyzed_per_rank": int(max_rows),
            "steps_used_primary": int(primary_rollup["steps_analyzed"]),
        },
        "global_rollup": global_rollup,
        "rank_rollup": {
            "representative_rank": representative_rank,
            "bottleneck_rank": worst_rank,
            "representative_avg_step_ms": representative_avg_step_ms,
            "bottleneck_avg_step_ms": worst_avg_step_ms,
            "imbalance_gap_pct": worst_vs_representative_pct,
        },
        "per_rank": {
            str(rank): _rank_entry_to_json(rank, s)
            for rank, s in sorted(per_rank_summary.items())
        },
        "per_device": {},
        "extensibility": {
            "per_device_available": False,
            "per_device_reason": (
                "step_time_samples is currently rank-based and does not "
                "persist canonical device identity for end-of-run timing "
                "summaries"
            ),
            "future_extension": (
                "populate per_device when canonical step-time rows include "
                "stable device identity"
            ),
        },
        "notes": {
            "step_basis": (
                "step_avg = dataloader + max(step_time, forward + backward + optimizer)"
            ),
            "comparison_mode": (
                f"per-rank averages over each rank's last up to {int(max_rows)} steps"
            ),
            "representative_rank_definition": (
                "rank closest to the median average step time across analyzed ranks"
            ),
            "bottleneck_rank_definition": (
                "rank with the highest average step time across analyzed ranks"
            ),
            "wait_definition": (
                "wait_proxy = max(0, effective_step - (forward + backward + optimizer))"
            ),
        },
        "card": card,
    }
    return card, summary


def generate_step_time_summary_card(
    db_path: str,
    *,
    max_rows: int = 50_000,
    print_to_stdout: bool = True,
) -> Dict[str, Any]:
    """
    Generate a compact STEP TIME summary from `step_time_samples`.

    Parameters
    ----------
    db_path:
        Path to the SQLite DB file.
    max_rows:
        Maximum number of latest steps analyzed per rank.
    print_to_stdout:
        If True, print the rendered summary.

    Returns
    -------
    Dict[str, Any]
        Structured summary JSON including the rendered `card`.

    Notes
    -----
    - Uses `step_time_samples`, not raw event transport tables.
    - Assumes one projected row per step per rank.
    - Diagnosis is intentionally reused from the shared step-time diagnosis
      engine so live views and end-of-run summaries stay consistent.
    """
    conn = sqlite3.connect(db_path)

    try:
        latest_step_observed_row = conn.execute(
            "SELECT MAX(step) FROM step_time_samples;"
        ).fetchone()
        latest_step_observed = (
            int(latest_step_observed_row[0])
            if latest_step_observed_row[0] is not None
            else None
        )
        training_steps = (
            latest_step_observed + 1 if latest_step_observed is not None else 0
        )

        rank_rows = conn.execute(
            "SELECT DISTINCT rank FROM step_time_samples ORDER BY rank ASC;"
        ).fetchall()
        ranks_present = [int(r[0]) for r in rank_rows if r[0] is not None]

        per_rank_summary: Dict[int, RankStepSummary] = {}
        per_rank_step_metrics: Dict[int, Dict[int, Dict[str, float]]] = {}

        for rank in ranks_present:
            step_rows = _load_rank_step_rows(
                conn,
                rank=rank,
                max_rows=max_rows,
            )
            analysis = _build_rank_summary(step_rows)
            if analysis is not None:
                per_rank_summary[rank] = analysis.summary
                per_rank_step_metrics[rank] = analysis.per_step_metrics
    finally:
        conn.close()

    card, step_summary = _build_step_time_card(
        training_steps=training_steps,
        latest_step_observed=latest_step_observed,
        per_rank_summary=per_rank_summary,
        per_rank_step_metrics=per_rank_step_metrics,
        max_rows=max_rows,
    )

    append_text(db_path + "_summary_card.txt", card)

    existing = load_json_or_empty(db_path + "_summary_card.json")
    existing["step_time"] = step_summary
    write_json(db_path + "_summary_card.json", existing)

    if print_to_stdout:
        print(card)

    return step_summary
