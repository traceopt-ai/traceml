"""
STEP TIME post-run summary generator.

This module reads the `step_time_samples` SQLite projection table, builds
per-rank timing summaries, and writes:
1) a text card (`*_summary_card.txt`)
2) a structured JSON payload (`*_summary_card.json`)

It also augments the summary with a diagnosis produced by a dedicated
summary-mode adapter.
"""

import json
import sqlite3
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from traceml.aggregator.summaries.step_time_diagnosis import (
    RankStepSignals,
    build_summary_step_diagnosis,
    diagnosis_to_json,
)


def _append_text(path: str, text: str) -> None:
    """
    Append text to a file, inserting a blank line first if the file already
    contains content.
    """
    with open(path, "a+", encoding="utf-8") as f:
        f.seek(0, 2)
        if f.tell() > 0:
            f.write("\n")
        f.write(text.rstrip() + "\n")


def _load_json_or_empty(path: str) -> Dict[str, Any]:
    """Load JSON if present; otherwise return an empty dict."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    """Write JSON with indentation."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _safe_float(x: Any) -> float:
    """Best-effort float conversion; returns 0.0 on failure."""
    try:
        return float(x)
    except Exception:
        return 0.0


def _finite_float(x: Any) -> float:
    """Convert to float; coerce non-finite values to 0.0."""
    v = _safe_float(x)
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
    Map raw event names to canonical buckets.

    Returns one of:
      - dataloader
      - forward
      - backward
      - optimizer
      - step_time
      - None
    """
    n = str(name).lower()

    # Exact / internal aliases first
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

    # Permissive fallbacks
    if "data" in n or "dataloader" in n or "input" in n or "batch" in n:
        return "dataloader"
    if "forward" in n or n == "fwd":
        return "forward"
    if "backward" in n or "bwd" in n:
        return "backward"
    if "optim" in n or "optimizer" in n or n in {"step", "update"}:
        return "optimizer"

    return None


def _fmt_ms(x: Optional[float]) -> str:
    """Format milliseconds for card output."""
    return "n/a" if x is None else f"{x:.1f}ms"


def _fmt_pct(x: Optional[float]) -> str:
    """Format percentage for card output."""
    return "n/a" if x is None else f"{x:.1f}%"


def _share(num: float, denom: float) -> Optional[float]:
    """Return percentage share num/denom, or None if denom is non-positive."""
    if denom <= 0.0:
        return None
    return 100.0 * num / denom


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

    Assumes one row per step in `step_time_samples`.
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
        rows.append(
            {
                "step": int(step),
                "events": events,
            }
        )
    return rows


def _row_metrics(events: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Convert one step's dynamic event map into canonical bucket metrics.

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
) -> Optional[RankStepSummary]:
    """
    Build a per-rank summary over the provided step rows.

    For each step:
        gpu_compute = forward + backward + optimizer
        total_step  = dataloader + max(step_time, gpu_compute)

    This keeps step-time comparable across ranks even when step_time excludes
    dataloader and GPU compute is recorded separately.
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

    for row in step_rows:
        metrics = _row_metrics(row["events"])
        if metrics is None:
            continue

        dl = _finite_float(metrics["dataloader"])
        fwd = _finite_float(metrics["forward"])
        bwd = _finite_float(metrics["backward"])
        opt = _finite_float(metrics["optimizer"])
        step_cpu = _finite_float(metrics["step_time"])

        gpu_compute = fwd + bwd + opt
        total_step = dl + max(step_cpu, gpu_compute)

        sum_dl += dl
        sum_fwd += fwd
        sum_bwd += bwd
        sum_opt += opt
        sum_step_cpu += step_cpu
        sum_total += total_step
        n += 1

    if n == 0:
        return None

    return RankStepSummary(
        steps_analyzed=n,
        avg_dataloader_ms=sum_dl / n,
        avg_forward_ms=sum_fwd / n,
        avg_backward_ms=sum_bwd / n,
        avg_optimizer_ms=sum_opt / n,
        avg_step_cpu_ms=sum_step_cpu / n,
        avg_gpu_compute_ms=(sum_fwd + sum_bwd + sum_opt) / n,
        avg_total_step_ms=sum_total / n,
    )


def _split_ms(s: RankStepSummary) -> Dict[str, float]:
    """Return the main timing split in ms for one rank summary."""
    return {
        "dataloader": s.avg_dataloader_ms,
        "forward": s.avg_forward_ms,
        "backward": s.avg_backward_ms,
        "optimizer": s.avg_optimizer_ms,
    }


def _split_pct(s: RankStepSummary) -> Dict[str, Optional[float]]:
    """Return the main timing split as percentage share of avg total step."""
    return {
        "dataloader": _share(s.avg_dataloader_ms, s.avg_total_step_ms),
        "forward": _share(s.avg_forward_ms, s.avg_total_step_ms),
        "backward": _share(s.avg_backward_ms, s.avg_total_step_ms),
        "optimizer": _share(s.avg_optimizer_ms, s.avg_total_step_ms),
    }


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
    Build one concise takeaway line from the median/worst rank splits.
    """
    if median_split_ms is None and worst_split_ms is None:
        return "n/a"

    if median_split_ms is not None and worst_split_ms is None:
        dom = _dominant_bucket(median_split_ms)
        return f"{dom} is the largest part of step time"

    if median_split_ms is None and worst_split_ms is not None:
        dom = _dominant_bucket(worst_split_ms)
        return f"{dom} is the largest part of step time"

    dom_median = _dominant_bucket(median_split_ms or {})
    dom_worst = _dominant_bucket(worst_split_ms or {})

    if dom_median == dom_worst:
        return (
            f"{dom_median} is the largest part on both median and worst rank"
        )

    return (
        f"median rank is dominated by {dom_median}; "
        f"worst rank is dominated by {dom_worst}"
    )


def _build_step_time_card(
    *,
    training_steps: int,
    latest_step_observed: Optional[int],
    per_rank_summary: Dict[int, RankStepSummary],
    max_rows: int,
) -> tuple[str, Dict[str, Any]]:
    """
    Build a clean, shareable STEP TIME summary card.

    The card prioritizes immediate understanding:
    - who the straggler is
    - how large the gap is
    - where time is spent on the median and worst rank
    - one clear dominant timing takeaway

    Detailed percentages remain available in JSON output.
    """
    ranks_present = sorted(per_rank_summary.keys())

    worst_rank: Optional[int] = None
    median_rank: Optional[int] = None
    worst_avg_step_ms: Optional[float] = None
    median_avg_step_ms: Optional[float] = None
    worst_vs_median_pct: Optional[float] = None

    if per_rank_summary:
        avg_total_by_rank = {
            rank: s.avg_total_step_ms for rank, s in per_rank_summary.items()
        }
        worst_rank = max(avg_total_by_rank, key=avg_total_by_rank.get)
        median_rank = _closest_rank_to_median(avg_total_by_rank)

        worst_avg_step_ms = avg_total_by_rank.get(worst_rank)
        median_avg_step_ms = (
            avg_total_by_rank.get(median_rank)
            if median_rank is not None
            else None
        )

        if (
            worst_avg_step_ms is not None
            and median_avg_step_ms is not None
            and median_avg_step_ms > 0.0
            and worst_rank is not None
            and median_rank is not None
            and worst_rank != median_rank
        ):
            worst_vs_median_pct = (
                100.0
                * (worst_avg_step_ms - median_avg_step_ms)
                / median_avg_step_ms
            )

    worst_summary = (
        per_rank_summary.get(worst_rank) if worst_rank is not None else None
    )
    median_summary = (
        per_rank_summary.get(median_rank) if median_rank is not None else None
    )

    median_split_ms = _split_ms(median_summary) if median_summary else None
    worst_split_ms = _split_ms(worst_summary) if worst_summary else None
    median_split_pct = _split_pct(median_summary) if median_summary else None
    worst_split_pct = _split_pct(worst_summary) if worst_summary else None
    dominant_text = _dominant_line(median_split_ms, worst_split_ms)

    summary_diag = build_summary_step_diagnosis(
        rank_signals=_to_rank_signals(per_rank_summary),
        max_rows=max_rows,
    )

    width = 78
    inner_width = width - 4

    def border() -> str:
        return "+" + "-" * (width - 2) + "+"

    def row(text: str = "") -> str:
        return f"|  {text:<{inner_width}}|"

    def wrapped_row(label: str, text: str) -> None:
        """
        Render a labeled row with wrapping while preserving card width.
        """
        prefix = f"{label:<13}"
        wrapped = textwrap.wrap(
            text,
            width=max(10, inner_width - len(prefix)),
        ) or [""]
        lines.append(row(f"{prefix}{wrapped[0]}"))
        for part in wrapped[1:]:
            lines.append(row(f"{'':<{len(prefix)}}{part}"))

    header = (
        f"TraceML Step Timing Summary | steps {training_steps} | "
        f"ranks {len(ranks_present)}"
    )

    lines = [
        border(),
        row(header),
        border(),
        row("STEP TIME"),
        row(),
    ]

    if not per_rank_summary:
        lines.append(
            row(
                f"Latest step   {latest_step_observed if latest_step_observed is not None else 'n/a'}"
            )
        )
        lines.append(row("Steps used    n/a"))
        lines.append(row("Step avg      n/a"))
        lines.append(row())
        lines.append(row("Dominant      n/a"))
    elif len(ranks_present) == 1:
        only_rank = ranks_present[0]
        only = per_rank_summary[only_rank]
        only_split_ms = _split_ms(only)

        lines.append(
            row(
                f"Steps used    last {only.steps_analyzed:,} on rank r{only_rank}"
            )
        )
        lines.append(
            row(
                f"Step avg      rank r{only_rank} {_fmt_ms(only.avg_total_step_ms)}"
            )
        )
        lines.append(
            row(
                f"Split         DL {_fmt_ms(only_split_ms['dataloader'])} | "
                f"FWD {_fmt_ms(only_split_ms['forward'])} | "
                f"BWD {_fmt_ms(only_split_ms['backward'])} | "
                f"OPT {_fmt_ms(only_split_ms['optimizer'])}"
            )
        )
        lines.append(row())
        lines.append(
            row(f"Dominant      {_dominant_line(only_split_ms, None)}")
        )
    else:
        lines.append(
            row(
                f"Steps used    last {worst_summary.steps_analyzed:,} / rank"
                if worst_summary is not None
                else "Steps used    n/a"
            )
        )
        lines.append(
            row(
                f"Straggler     worst rank r{worst_rank} {_fmt_ms(worst_avg_step_ms)} | "
                f"median rank r{median_rank} {_fmt_ms(median_avg_step_ms)} | "
                f"gap {_fmt_pct(worst_vs_median_pct)}"
            )
        )

        if median_split_ms is not None:
            lines.append(
                row(
                    f"Median split  DL {_fmt_ms(median_split_ms['dataloader'])} | "
                    f"FWD {_fmt_ms(median_split_ms['forward'])} | "
                    f"BWD {_fmt_ms(median_split_ms['backward'])} | "
                    f"OPT {_fmt_ms(median_split_ms['optimizer'])}"
                )
            )

        if worst_split_ms is not None:
            lines.append(
                row(
                    f"Worst split   DL {_fmt_ms(worst_split_ms['dataloader'])} | "
                    f"FWD {_fmt_ms(worst_split_ms['forward'])} | "
                    f"BWD {_fmt_ms(worst_split_ms['backward'])} | "
                    f"OPT {_fmt_ms(worst_split_ms['optimizer'])}"
                )
            )

        lines.append(row())
        lines.append(row(f"Dominant      {dominant_text}"))

    if summary_diag is not None:
        lines.append(row())
        wrapped_row(
            "Diagnosis",
            f"{summary_diag.status}: {summary_diag.reason}",
        )
        wrapped_row("Action", summary_diag.action)
        if summary_diag.note:
            wrapped_row("Note", summary_diag.note)

    lines.append(border())
    card = "\n".join(lines)

    summary = {
        "training_steps": training_steps,
        "latest_step_observed": latest_step_observed,
        "ranks_seen": len(ranks_present),
        "max_steps_analyzed_per_rank": int(max_rows),
        "worst_rank": worst_rank,
        "median_rank": median_rank,
        "worst_vs_median_pct": worst_vs_median_pct,
        "worst_avg_step_ms": worst_avg_step_ms,
        "median_avg_step_ms": median_avg_step_ms,
        "dominant_text": dominant_text,
        "median_split_ms": median_split_ms,
        "worst_split_ms": worst_split_ms,
        "median_split_pct": median_split_pct,
        "worst_split_pct": worst_split_pct,
        "diagnosis": diagnosis_to_json(summary_diag),
        "per_rank": {
            str(rank): {
                "steps_analyzed": s.steps_analyzed,
                "avg_dataloader_ms": s.avg_dataloader_ms,
                "avg_forward_ms": s.avg_forward_ms,
                "avg_backward_ms": s.avg_backward_ms,
                "avg_optimizer_ms": s.avg_optimizer_ms,
                "avg_step_cpu_ms": s.avg_step_cpu_ms,
                "avg_gpu_compute_ms": s.avg_gpu_compute_ms,
                "avg_total_step_ms": s.avg_total_step_ms,
            }
            for rank, s in per_rank_summary.items()
        },
        "notes": {
            "step_basis": (
                "step_avg = dataloader + max(step_time, forward + backward + optimizer)"
            ),
            "comparison_mode": (
                f"per-rank averages over each rank's last up to {int(max_rows)} steps"
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
    Generate a clean STEP TIME summary card from `step_time_samples`.

    Parameters
    ----------
    db_path:
        Path to the SQLite DB file.
    max_rows:
        Maximum number of latest steps analyzed per rank.
    print_to_stdout:
        If True, print the rendered card.

    Returns
    -------
    Dict[str, Any]
        Structured summary JSON including the rendered `card`.

    Notes
    -----
    - Uses `step_time_samples`, not `raw_messages`.
    - Assumes one projected row per step per rank.
    - The dynamic event map is read from `events_json`.
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
        for rank in ranks_present:
            step_rows = _load_rank_step_rows(
                conn,
                rank=rank,
                max_rows=max_rows,
            )
            summary = _build_rank_summary(step_rows)
            if summary is not None:
                per_rank_summary[rank] = summary

    finally:
        conn.close()

    card, step_summary = _build_step_time_card(
        training_steps=training_steps,
        latest_step_observed=latest_step_observed,
        per_rank_summary=per_rank_summary,
        max_rows=max_rows,
    )

    _append_text(db_path + "_summary_card.txt", card)

    existing = _load_json_or_empty(db_path + "_summary_card.json")
    existing["step_time"] = step_summary
    _write_json(db_path + "_summary_card.json", existing)

    if print_to_stdout:
        print(card)

    return step_summary
