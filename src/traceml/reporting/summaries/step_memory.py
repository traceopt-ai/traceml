"""
Compact end-of-run step-memory summary generation.

This module reads aligned step-memory telemetry from `step_memory_samples`,
reuses the shared step-memory diagnosis engine, and produces:

1. a compact text summary for end-of-run display and sharing
2. a structured JSON payload for automation

Design goals
------------
- Keep the printed summary short and actionable
- Reuse the same diagnosis logic as live CLI and dashboard views
- Use one clear canonical schema for end-of-run step-memory data
- Preserve richer machine-readable fields in JSON
"""

import sqlite3
from typing import Any, Dict, Optional

from traceml.diagnostics.step_memory import (
    StepMemoryDiagnosis,
    build_step_memory_diagnosis,
)
from traceml.diagnostics.trends import compute_trend_evidence
from traceml.renderers.step_memory.common import (
    StepMemoryMetricsDB,
    build_step_memory_combined_result,
)
from traceml.renderers.step_memory.schema import StepMemoryCombinedMetric
from traceml.reporting.summaries.diagnosis_presentation import (
    diagnosis_presentation_to_json,
    present_step_memory_summary_diagnosis,
)
from traceml.reporting.summaries.summary_formatting import (
    format_ratio_percent,
    safe_float,
)
from traceml.reporting.summaries.summary_io import (
    append_text,
    load_json_or_empty,
    write_json,
)
from traceml.utils.formatting import fmt_mem_new


def _metric_sort_key(metric: StepMemoryCombinedMetric) -> int:
    """Stable metric ordering: allocated first, reserved second."""
    if metric.metric == "peak_allocated":
        return 0
    if metric.metric == "peak_reserved":
        return 1
    return 99


def _metric_label(metric_name: str) -> str:
    """Human-friendly label for one step-memory metric key."""
    if metric_name == "peak_allocated":
        return "peak allocated"
    if metric_name == "peak_reserved":
        return "peak reserved"
    return metric_name.replace("_", " ")


def _diagnosis_to_json(
    diagnosis: Optional[StepMemoryDiagnosis],
) -> Optional[Dict[str, Any]]:
    """Serialize StepMemoryDiagnosis into a compare-friendly JSON shape."""
    if diagnosis is None:
        return None

    return {
        "kind": diagnosis.kind,
        "status": diagnosis.status,
        "severity": diagnosis.severity,
        "metric": diagnosis.metric,
        "steps_used": diagnosis.steps_used,
        "worst_rank": diagnosis.worst_rank,
        "reason": diagnosis.reason,
        "action": diagnosis.action,
        "note": diagnosis.note,
        "confidence": diagnosis.confidence,
    }


def _latest_step_observed(conn: sqlite3.Connection) -> Optional[int]:
    """Return the latest observed memory step across all ranks."""
    row = conn.execute("SELECT MAX(step) FROM step_memory_samples;").fetchone()
    if not row or row[0] is None:
        return None
    return int(row[0])


def _gpu_total_bytes(conn: sqlite3.Connection) -> Optional[float]:
    """
    Best-effort device total memory from process telemetry.

    This is optional but useful for HIGH_PRESSURE diagnosis.
    """
    try:
        row = conn.execute(
            "SELECT MAX(gpu_mem_total_bytes) FROM process_samples;"
        ).fetchone()
    except Exception:
        return None

    if not row or row[0] is None:
        return None
    try:
        value = float(row[0])
    except Exception:
        return None
    return value if value > 0.0 else None


def _head_tail_trend(series: list[float]) -> Dict[str, Optional[float]]:
    """
    Compute the canonical trend summary for one memory series.

    Compatibility note:
    - `head_avg_bytes` maps to the canonical baseline average
    - `tail_avg_bytes` maps to the canonical recent average
    """
    values = [max(0.0, safe_float(v)) for v in series]
    evidence = compute_trend_evidence(values)

    if evidence is None:
        return {
            "head_avg_bytes": None,
            "mid_avg_bytes": None,
            "tail_avg_bytes": None,
            "baseline_avg_bytes": None,
            "recent_avg_bytes": None,
            "delta_bytes": None,
            "growth_pct": None,
        }

    return {
        "head_avg_bytes": evidence.baseline_avg,
        "mid_avg_bytes": evidence.mid_avg,
        "tail_avg_bytes": evidence.recent_avg,
        "baseline_avg_bytes": evidence.baseline_avg,
        "recent_avg_bytes": evidence.recent_avg,
        "delta_bytes": evidence.delta_vs_baseline,
        "growth_pct": evidence.delta_pct_vs_baseline,
    }


def _window_variability(series: list[float]) -> Dict[str, Optional[float]]:
    """
    Compute compact within-window variability statistics for one memory series.

    Peak memory alone does not distinguish stable high usage from noisy or
    jittery behavior. This summary intentionally keeps the variability signal
    compact and easy to interpret:
    - `peak_bytes` captures the maximum observed value in the analyzed window
    - `latest_bytes` captures the most recent value in the window
    - `window_min_bytes` and `window_max_bytes` provide context
    - `window_range_bytes` approximates instability / jitter magnitude
    - `window_range_pct_of_peak` normalizes that range to the local peak
    """
    values = [max(0.0, safe_float(v)) for v in series]
    if not values:
        return {
            "peak_bytes": None,
            "latest_bytes": None,
            "window_min_bytes": None,
            "window_max_bytes": None,
            "window_range_bytes": None,
            "window_range_pct_of_peak": None,
        }

    min_value = min(values)
    max_value = max(values)
    latest_value = values[-1]
    range_value = max_value - min_value

    return {
        "peak_bytes": max_value,
        "latest_bytes": latest_value,
        "window_min_bytes": min_value,
        "window_max_bytes": max_value,
        "window_range_bytes": range_value,
        "window_range_pct_of_peak": (
            (range_value / max_value) if max_value > 0.0 else None
        ),
    }


def _metric_to_json(metric: StepMemoryCombinedMetric) -> Dict[str, Any]:
    """
    Serialize one combined step-memory metric into the canonical JSON shape.

    The combined metric remains useful because it answers the run-level story:
    - `median` describes a typical rank
    - `worst` describes the gating / most memory-heavy rank
    - `skew` quantifies rank imbalance
    """
    worst_trend = _head_tail_trend(metric.series.worst)
    median_trend = _head_tail_trend(metric.series.median)

    worst_variability = _window_variability(metric.series.worst)
    median_variability = _window_variability(metric.series.median)

    return {
        "metric": metric.metric,
        "device": metric.device,
        "coverage": {
            "expected_steps": metric.coverage.expected_steps,
            "steps_used": metric.coverage.steps_used,
            "completed_step": metric.coverage.completed_step,
            "world_size": metric.coverage.world_size,
            "ranks_present": metric.coverage.ranks_present,
            "incomplete": metric.coverage.incomplete,
        },
        "summary": {
            "window_size": metric.summary.window_size,
            "steps_used": metric.summary.steps_used,
            "median_peak_bytes": metric.summary.median_peak,
            "worst_peak_bytes": metric.summary.worst_peak,
            "worst_rank": metric.summary.worst_rank,
            "skew_ratio": metric.summary.skew_ratio,
            "skew_pct": metric.summary.skew_pct,
        },
        "series_summary": {
            "median": {
                "trend": median_trend,
                "variability": median_variability,
            },
            "worst": {
                "trend": worst_trend,
                "variability": worst_variability,
            },
        },
    }


def _primary_metric(
    metrics: list[StepMemoryCombinedMetric],
    diagnosis: Optional[StepMemoryDiagnosis],
) -> Optional[StepMemoryCombinedMetric]:
    """
    Pick the primary metric to surface in the printed summary.

    Preference order:
    1. diagnosis.metric, if available
    2. peak_reserved
    3. peak_allocated
    4. first metric
    """
    if not metrics:
        return None

    by_name = {m.metric: m for m in metrics}

    if diagnosis is not None and diagnosis.metric in by_name:
        return by_name[diagnosis.metric]
    if "peak_reserved" in by_name:
        return by_name["peak_reserved"]
    if "peak_allocated" in by_name:
        return by_name["peak_allocated"]
    return metrics[0]


def _common_suffix_steps(
    per_rank_steps: Dict[int, Dict[int, float]],
    *,
    completed_step: int,
    window_size: int,
) -> list[int]:
    """
    Return the last `window_size` common steps across all provided ranks.

    This intentionally mirrors the alignment semantics used by the shared
    step-memory renderer path:
    - aligned on completed steps only
    - use the largest common suffix up to `window_size`
    - avoid misleading partial-rank windows
    """
    maps = list(per_rank_steps.values())
    if not maps or window_size <= 0 or completed_step < 0:
        return []

    reference = maps[0]
    out_rev: list[int] = []

    step = int(completed_step)
    scan_cap = max(window_size * 20, window_size + 1)
    scanned = 0

    while step >= 0 and len(out_rev) < window_size:
        scanned += 1
        if scanned > scan_cap:
            break

        if step in reference:
            if all(step in step_map for step_map in maps[1:]):
                out_rev.append(step)

        step -= 1

    if not out_rev:
        return []

    out_rev.reverse()
    return out_rev


def _parse_gpu_idx(device: Optional[str]) -> Optional[int]:
    """
    Best-effort parse of a CUDA device string such as `cuda:0`.
    """
    if not device:
        return None

    text = str(device).strip().lower()
    if not text.startswith("cuda:"):
        return None

    try:
        return int(text.split(":", 1)[1])
    except Exception:
        return None


def _load_per_rank_summary(
    conn: sqlite3.Connection,
    *,
    db: StepMemoryMetricsDB,
    metrics: list[StepMemoryCombinedMetric],
    window_size: int,
) -> Dict[str, Any]:
    """
    Build per-rank step-memory summaries for the analyzed aligned tail window.

    Combined worst/median summaries are excellent for end-of-run diagnosis, but
    one often need to answer:
    - which rank is gating or drifting?
    - is one rank materially less stable than others?
    - do allocated and reserved tell the same story?

    This function provides that machine-readable rank-level view while keeping
    the printed summary compact.
    """
    latest_per_rank = db.fetch_latest_step_per_rank(conn)
    if not latest_per_rank:
        return {}

    completed_step = min(latest_per_rank.values())
    scan_span = max(int(window_size) * 20, int(window_size) + 1)
    start_step = max(0, int(completed_step) - scan_span + 1)

    per_rank: Dict[str, Any] = {}

    for metric in metrics:
        rank_maps, rank_devices = db.fetch_rank_step_maps(
            conn,
            metric_key=metric.metric,
            start_step=start_step,
            end_step=int(completed_step),
            max_unique_steps_per_rank=scan_span,
        )

        if not rank_maps:
            continue

        common_steps = _common_suffix_steps(
            rank_maps,
            completed_step=int(completed_step),
            window_size=int(window_size),
        )
        if not common_steps:
            continue

        for rank in sorted(rank_maps.keys()):
            step_map = rank_maps.get(rank, {})
            values = [
                float(step_map[s]) for s in common_steps if s in step_map
            ]
            if len(values) != len(common_steps) or not values:
                continue

            rank_key = str(rank)
            rank_device = rank_devices.get(rank)

            entry = per_rank.setdefault(
                rank_key,
                {
                    "identity": {
                        "rank": int(rank),
                        "device": rank_device,
                        "gpu_idx": _parse_gpu_idx(rank_device),
                    },
                    "metrics": {},
                },
            )

            if (
                entry["identity"].get("device") is None
                and rank_device is not None
            ):
                entry["identity"]["device"] = rank_device
                entry["identity"]["gpu_idx"] = _parse_gpu_idx(rank_device)

            entry["metrics"][metric.metric] = {
                "coverage": {
                    "steps_used": len(common_steps),
                    "window_size": int(window_size),
                    "completed_step": int(completed_step),
                },
                "summary": _window_variability(values),
                "trend": _head_tail_trend(values),
            }

    return dict(sorted(per_rank.items(), key=lambda item: int(item[0])))


def _empty_global_rollup() -> Dict[str, Any]:
    """
    Return a stable empty global rollup for missing-data cases.
    """
    return {
        "primary_metric": None,
        "diagnosis_status": None,
        "analysis_window": {
            "steps_used": 0,
            "window_size": None,
            "completed_step": None,
            "ranks_seen": 0,
        },
        "typical": None,
        "bottleneck": None,
        "imbalance": None,
    }


def _build_global_rollup(
    *,
    primary: Optional[StepMemoryCombinedMetric],
    diagnosis: Optional[StepMemoryDiagnosis],
) -> Dict[str, Any]:
    """
    Build the canonical global memory rollup for the analyzed window.

    Semantics
    ---------
    - `typical` describes the median-across-ranks memory behavior
    - `bottleneck` describes the worst-across-ranks memory behavior
    - `imbalance` captures how far the gating rank is from the typical rank
    """
    if primary is None:
        return _empty_global_rollup()

    typical_trend = _head_tail_trend(primary.series.median)
    bottleneck_trend = _head_tail_trend(primary.series.worst)

    typical_variability = _window_variability(primary.series.median)
    bottleneck_variability = _window_variability(primary.series.worst)

    return {
        "primary_metric": primary.metric,
        "diagnosis_status": (
            diagnosis.status if diagnosis is not None else None
        ),
        "analysis_window": {
            "steps_used": primary.summary.steps_used,
            "window_size": primary.summary.window_size,
            "completed_step": primary.coverage.completed_step,
            "ranks_seen": primary.coverage.ranks_present,
        },
        "typical": {
            "metric": primary.metric,
            "peak_bytes": primary.summary.median_peak,
            "trend": typical_trend,
            "variability": typical_variability,
        },
        "bottleneck": {
            "metric": primary.metric,
            "worst_rank": primary.summary.worst_rank,
            "peak_bytes": primary.summary.worst_peak,
            "trend": bottleneck_trend,
            "variability": bottleneck_variability,
        },
        "imbalance": {
            "worst_rank": primary.summary.worst_rank,
            "skew_ratio": primary.summary.skew_ratio,
            "skew_pct": primary.summary.skew_pct,
        },
    }


def _build_step_memory_card(
    *,
    training_steps: int,
    latest_step_observed: Optional[int],
    metrics: list[StepMemoryCombinedMetric],
    diagnosis: Optional[StepMemoryDiagnosis],
    per_rank: Dict[str, Any],
) -> tuple[str, Dict[str, Any]]:
    """
    Build a compact, shareable end-of-run step-memory summary.

    The printed summary is intentionally concise:
    - scope
    - diagnosis
    - primary metric details
    - one stable trend hint

    The JSON payload is richer than the printed text and is the canonical
    machine-readable representation for compare, logging, and dashboards.
    """
    sorted_metrics = sorted(metrics, key=_metric_sort_key)
    primary = _primary_metric(sorted_metrics, diagnosis)
    diagnosis_presented = present_step_memory_summary_diagnosis(diagnosis)

    if not sorted_metrics or primary is None:
        card = "\n".join(
            [
                f"TraceML Step Memory Summary | steps {training_steps} | ranks 0",
                "Step Memory",
                (
                    f"- Scope: latest step "
                    f"{latest_step_observed if latest_step_observed is not None else 'n/a'}"
                ),
                "- Diagnosis: NO DATA",
                "- Why: No step-memory data was collected.",
                "- Next: Run longer or collect more memory steps.",
            ]
        )

        summary = {
            "training_steps": training_steps,
            "latest_step_observed": latest_step_observed,
            "ranks_seen": 0,
            "diagnosis": None,
            "diagnosis_presented": None,
            "primary_metric": None,
            "overview": {
                "training_steps": training_steps,
                "latest_step_observed": latest_step_observed,
                "ranks_seen": 0,
                "metric_names": [],
                "window_size": None,
                "steps_used": 0,
            },
            "global_rollup": _empty_global_rollup(),
            "metric_rollup": {},
            "metrics": {},
            "per_rank": {},
            "units": {"memory": "bytes"},
            "notes": {
                "window_basis": (
                    "aligned tail window across ranks using the largest common "
                    "suffix of completed steps"
                ),
                "trend_definition": (
                    "trend compares recent average against baseline average "
                    "within the analyzed window"
                ),
                "variability_definition": (
                    "window_range_bytes approximates within-window instability "
                    "or jitter for the analyzed series"
                ),
            },
            "card": card,
        }
        return card, summary

    primary_trend_worst = _head_tail_trend(primary.series.worst)
    primary_trend_median = _head_tail_trend(primary.series.median)
    primary_worst_variability = _window_variability(primary.series.worst)
    primary_median_variability = _window_variability(primary.series.median)

    steps_used = int(primary.summary.steps_used)
    ranks_seen = int(primary.coverage.ranks_present)

    lines = [
        f"TraceML Step Memory Summary | steps {training_steps} | ranks {ranks_seen}",
        "Step Memory",
        f"- Scope: last {steps_used} aligned steps",
    ]

    if diagnosis_presented is not None:
        lines.append(f"- Diagnosis: {diagnosis_presented.status}")
        lines.append(f"- Why: {diagnosis_presented.reason}")
        lines.append(f"- Next: {diagnosis_presented.action}")

    lines.append(
        (
            f"- Primary: {_metric_label(primary.metric)} | "
            f"worst {fmt_mem_new(primary.summary.worst_peak)}"
            f" on r{primary.summary.worst_rank if primary.summary.worst_rank is not None else 'n/a'} | "
            f"skew {format_ratio_percent(primary.summary.skew_pct)}"
        )
    )

    if primary_trend_worst["delta_bytes"] is not None:
        trend_text = (
            f"- Trend: worst "
            f"{'+' if float(primary_trend_worst['delta_bytes']) >= 0.0 else '-'}"
            f"{fmt_mem_new(abs(float(primary_trend_worst['delta_bytes'])))}"
        )
        if primary_trend_worst["growth_pct"] is not None:
            trend_text += (
                f" (~{float(primary_trend_worst['growth_pct']) * 100.0:.0f}%)"
            )
        lines.append(trend_text)

    if diagnosis_presented is not None and diagnosis_presented.note:
        lines.append(f"- Note: {diagnosis_presented.note}")

    card = "\n".join(lines)

    metric_rollup = {
        metric.metric: _metric_to_json(metric) for metric in sorted_metrics
    }

    summary = {
        "training_steps": training_steps,
        "latest_step_observed": latest_step_observed,
        "ranks_seen": ranks_seen,
        "diagnosis": _diagnosis_to_json(diagnosis),
        "diagnosis_presented": diagnosis_presentation_to_json(
            diagnosis_presented
        ),
        "primary_metric": {
            "metric": primary.metric,
            "device": primary.device,
            "steps_used": primary.summary.steps_used,
            "worst_peak_bytes": primary.summary.worst_peak,
            "median_peak_bytes": primary.summary.median_peak,
            "worst_rank": primary.summary.worst_rank,
            "skew_pct": primary.summary.skew_pct,
            "trend": {
                "worst": primary_trend_worst,
                "median": primary_trend_median,
            },
            "variability": {
                "worst": primary_worst_variability,
                "median": primary_median_variability,
            },
        },
        "overview": {
            "training_steps": training_steps,
            "latest_step_observed": latest_step_observed,
            "ranks_seen": ranks_seen,
            "metric_names": [metric.metric for metric in sorted_metrics],
            "window_size": primary.summary.window_size,
            "steps_used": primary.summary.steps_used,
            "completed_step": primary.coverage.completed_step,
        },
        "global_rollup": _build_global_rollup(
            primary=primary,
            diagnosis=diagnosis,
        ),
        "metric_rollup": metric_rollup,
        # Kept as an alias to reduce downstream risk while callers migrate to
        # the clearer `metric_rollup` name.
        "metrics": metric_rollup,
        "per_rank": per_rank,
        "units": {"memory": "bytes"},
        "notes": {
            "window_basis": (
                "aligned tail window across ranks using the largest common "
                "suffix of completed steps"
            ),
            "primary_metric_definition": (
                "diagnosis-selected metric when available; otherwise reserved "
                "memory is preferred, then allocated memory"
            ),
            "trend_definition": (
                "trend compares recent average against baseline average within "
                "the analyzed window"
            ),
            "variability_definition": (
                "window_range_bytes approximates within-window instability or "
                "jitter for the analyzed series"
            ),
            "imbalance_definition": (
                "skew_pct compares worst rank peak against median rank peak "
                "for the analyzed aligned window"
            ),
        },
        "card": card,
    }
    return card, summary


def generate_step_memory_summary_card(
    db_path: str,
    *,
    window_size: int = 400,
    print_to_stdout: bool = True,
) -> Dict[str, Any]:
    """
    Generate a compact STEP MEMORY summary from `step_memory_samples`.

    Parameters
    ----------
    db_path:
        Path to the SQLite DB file.
    window_size:
        Number of aligned tail steps to use for end-of-run memory diagnosis and
        summary. A larger default than live CLI helps make end summaries more
        stable without switching to a misleading full-run average.
    print_to_stdout:
        If True, print the rendered summary.

    Returns
    -------
    Dict[str, Any]
        Structured summary JSON including the rendered `card`.

    Notes
    -----
    - This summary intentionally uses a stable aligned tail window rather than
      a naive whole-run average. Memory issues are often end-heavy, so tail
      behavior is more actionable than full-run dilution.
    - Diagnosis is reused from the shared step-memory diagnosis engine to keep
      live CLI, dashboard, and end-of-run summaries consistent.
    """
    db = StepMemoryMetricsDB(db_path=db_path)
    conn = db.connect()

    try:
        latest_step_observed = _latest_step_observed(conn)
        training_steps = (
            latest_step_observed + 1 if latest_step_observed is not None else 0
        )

        gpu_total_bytes = _gpu_total_bytes(conn)

        result = build_step_memory_combined_result(
            conn,
            db=db,
            window_size=max(1, int(window_size)),
        )
        metrics = sorted(result.metrics, key=_metric_sort_key)

        diagnosis = None
        if metrics:
            diagnosis = build_step_memory_diagnosis(
                metrics,
                gpu_total_bytes=gpu_total_bytes,
            )

        per_rank = _load_per_rank_summary(
            conn,
            db=db,
            metrics=metrics,
            window_size=max(1, int(window_size)),
        )
    finally:
        conn.close()

    card, step_memory_summary = _build_step_memory_card(
        training_steps=training_steps,
        latest_step_observed=latest_step_observed,
        metrics=metrics,
        diagnosis=diagnosis,
        per_rank=per_rank,
    )

    append_text(db_path + "_summary_card.txt", card)

    existing = load_json_or_empty(db_path + "_summary_card.json")
    existing["step_memory"] = step_memory_summary
    write_json(db_path + "_summary_card.json", existing)

    if print_to_stdout:
        print(card)

    return step_memory_summary
