"""Step-memory data shaping for the final-report section."""

import sqlite3
from typing import Any, Dict, Optional

from traceml.diagnostics.step_memory import StepMemoryDiagnosis
from traceml.diagnostics.trends import compute_trend_evidence
from traceml.renderers.step_memory.common import StepMemoryMetricsDB
from traceml.renderers.step_memory.schema import StepMemoryCombinedMetric
from traceml.reporting.summaries.diagnosis_presentation import (
    diagnosis_presentation_to_json,
    present_step_memory_summary_diagnosis,
)
from traceml.reporting.summaries.issue_summary import (
    issues_by_metric_json,
    issues_by_rank_json,
    issues_to_json,
)
from traceml.reporting.summaries.summary_formatting import (
    format_ratio_percent,
    safe_float,
)
from traceml.utils.formatting import fmt_mem_new

MAX_SUMMARY_WINDOW_ROWS = 10_000


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


def _no_gpu_diagnosis_json() -> Dict[str, Any]:
    """
    Stable summary diagnosis block for CPU-only / no-GPU runs.
    """
    return {
        "kind": "NO_GPU",
        "status": "NO GPU",
        "severity": "info",
        "metric": None,
        "steps_used": 0,
        "worst_rank": None,
        "reason": (
            "No GPU detected. Step memory uses torch-based GPU memory telemetry."
        ),
        "action": "Treat step memory as not applicable for this run.",
        "note": None,
        "confidence": 1.0,
    }


def _no_gpu_diagnosis_presented() -> Dict[str, Any]:
    """
    Stable end-of-run presentation block for CPU-only / no-GPU runs.
    """
    return {
        "status": "NO GPU",
        "reason": (
            "No GPU detected. Step memory uses torch-based GPU memory telemetry."
        ),
        "action": "Step memory is not applicable for this run.",
        "note": None,
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
    """Compute a head/tail trend summary for one memory series."""
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
    """Build the global memory rollup for the analyzed window."""
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
    diagnosis_result: Optional[Any],
    no_gpu_detected: bool,
    per_rank: Dict[str, Any],
) -> tuple[str, Dict[str, Any]]:
    """Build the end-of-run step-memory summary payload and text card."""
    sorted_metrics = sorted(metrics, key=_metric_sort_key)
    primary = _primary_metric(sorted_metrics, diagnosis)
    diagnosis_presented = present_step_memory_summary_diagnosis(diagnosis)
    issues = tuple(getattr(diagnosis_result, "issues", ()) or ())
    issues_by_rank, unassigned_issues = issues_by_rank_json(
        issues,
        rank_keys=per_rank.keys(),
    )
    issues_by_metric, metric_unassigned = issues_by_metric_json(issues)
    per_rank_with_issues = {
        rank_key: {
            **entry,
            "issues": issues_by_rank.get(rank_key, []),
        }
        for rank_key, entry in per_rank.items()
    }

    if not sorted_metrics or primary is None:
        no_gpu_diagnosis = (
            _no_gpu_diagnosis_json() if no_gpu_detected else None
        )
        no_gpu_presented = (
            _no_gpu_diagnosis_presented() if no_gpu_detected else None
        )
        diagnosis_status = (
            no_gpu_diagnosis["status"]
            if no_gpu_diagnosis is not None
            else "NO DATA"
        )
        diagnosis_reason = (
            no_gpu_diagnosis["reason"]
            if no_gpu_diagnosis is not None
            else "No step-memory data was collected."
        )
        latest_step_text = (
            latest_step_observed if latest_step_observed is not None else "n/a"
        )
        card = "\n".join(
            [
                f"TraceML Step Memory Summary | steps {training_steps} | ranks 0",
                "Step Memory",
                f"- Diagnosis: {diagnosis_status}",
                f"- Scope: latest step {latest_step_text}",
                "- Stats: n/a",
                f"- Why: {diagnosis_reason}",
            ]
        )

        summary = {
            "overview": {
                "training_steps": training_steps,
                "latest_step_observed": latest_step_observed,
                "ranks_seen": 0,
                "metric_names": [],
                "window_size": None,
                "steps_used": 0,
            },
            "primary_diagnosis": diagnosis_presentation_to_json(
                no_gpu_presented,
                include_action=False,
            ),
            "issues": [],
            "issues_by_rank": {},
            "issues_by_metric": {},
            "unassigned_issues": [],
            "global": {
                **_empty_global_rollup(),
                "metric_rollup": {},
            },
            "per_rank": {},
            "units": {"memory": "bytes"},
            "card": card,
        }
        return card, summary

    primary_trend_worst = _head_tail_trend(primary.series.worst)
    primary_trend_median = _head_tail_trend(primary.series.median)
    primary_worst_variability = _window_variability(primary.series.worst)
    primary_median_variability = _window_variability(primary.series.median)

    steps_used = int(primary.summary.steps_used)
    ranks_seen = int(primary.coverage.ranks_present)
    single_rank = ranks_seen <= 1

    diagnosis_status = (
        diagnosis_presented.status
        if diagnosis_presented is not None
        else "NO DATA"
    )
    diagnosis_reason = (
        diagnosis_presented.reason
        if diagnosis_presented is not None
        else "Need more step-memory samples."
    )
    title = (
        f"TraceML Step Memory Summary | steps {training_steps} | "
        f"ranks {ranks_seen}"
    )
    lines = [title, "Step Memory"]
    lines.append(f"- Diagnosis: {diagnosis_status}")
    lines.append(f"- Scope: last {steps_used} aligned steps")
    if single_rank:
        stats_text = (
            f"{_metric_label(primary.metric)} peak "
            f"{fmt_mem_new(primary.summary.worst_peak)}"
        )
    else:
        worst_rank = (
            f"r{primary.summary.worst_rank}"
            if primary.summary.worst_rank is not None
            else "rn/a"
        )
        stats_text = (
            f"{_metric_label(primary.metric)} worst "
            f"{fmt_mem_new(primary.summary.worst_peak)} on {worst_rank} | "
            f"skew {format_ratio_percent(primary.summary.skew_pct)}"
        )
    lines.append(f"- Stats: {stats_text}")
    lines.append(f"- Why: {diagnosis_reason}")

    card = "\n".join(lines)

    metric_rollup = {
        metric.metric: _metric_to_json(metric) for metric in sorted_metrics
    }
    primary_metric = {
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
    }
    global_summary = {
        **_build_global_rollup(
            primary=primary,
            diagnosis=diagnosis,
        ),
        "primary_metric": primary_metric,
        "metric_rollup": metric_rollup,
    }

    summary = {
        "overview": {
            "training_steps": training_steps,
            "latest_step_observed": latest_step_observed,
            "ranks_seen": ranks_seen,
            "metric_names": [metric.metric for metric in sorted_metrics],
            "window_size": primary.summary.window_size,
            "steps_used": primary.summary.steps_used,
            "completed_step": primary.coverage.completed_step,
        },
        "primary_diagnosis": diagnosis_presentation_to_json(
            diagnosis_presented,
            include_action=False,
        ),
        "issues": issues_to_json(issues),
        "issues_by_rank": issues_by_rank,
        "issues_by_metric": issues_by_metric,
        "unassigned_issues": unassigned_issues + metric_unassigned,
        "global": global_summary,
        "per_rank": per_rank_with_issues,
        "units": {"memory": "bytes"},
        "card": card,
    }
    return card, summary
