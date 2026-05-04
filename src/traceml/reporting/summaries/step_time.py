"""End-of-run step-time summary generation."""

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from traceml.reporting.summaries.step_time_diagnosis import RankStepSignals
from traceml.reporting.summaries.summary_formatting import (
    safe_float,
    share_percent,
)
from traceml.reporting.summaries.summary_io import (
    append_text,
    load_json_or_empty,
    write_json,
)

MAX_SUMMARY_WINDOW_ROWS = 10_000


def _finite_float(x: Any) -> float:
    """Convert to float; coerce non-finite values to 0.0."""
    v = safe_float(x)
    return v if np.isfinite(v) else 0.0


def _event_total_ms(by_dev: Any) -> float:
    """Sum ``duration_ms`` across all devices for one event."""
    if not isinstance(by_dev, dict):
        return 0.0

    total = 0.0
    for stats in by_dev.values():
        if not isinstance(stats, dict):
            continue
        total += _finite_float(stats.get("duration_ms"))
    return total


def _event_bucket(name: str) -> Optional[str]:
    """Map a raw event name to a step-time bucket."""
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
    """Per-rank averaged step-time summary."""

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
    """Per-rank summary plus per-step metrics."""

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
    """Build one timing rollup from a per-rank summary."""
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
    """Serialize one rank summary."""
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
    """Build the top-level step-time rollup for the run window."""
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
    """Build high-level overview fields from per-rank timing summaries."""
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


def generate_step_time_summary_card(
    db_path: str,
    *,
    max_rows: int = MAX_SUMMARY_WINDOW_ROWS,
    print_to_stdout: bool = True,
) -> Dict[str, Any]:
    """Generate the end-of-run step-time summary."""
    from traceml.reporting.sections.step_time import StepTimeSummarySection

    result = StepTimeSummarySection(max_rows=max_rows).build(db_path)
    card = result.text
    step_summary = result.payload

    append_text(db_path + "_summary_card.txt", card)

    existing = load_json_or_empty(db_path + "_summary_card.json")
    existing["step_time"] = step_summary
    write_json(db_path + "_summary_card.json", existing)

    if print_to_stdout:
        print(card)

    return step_summary
