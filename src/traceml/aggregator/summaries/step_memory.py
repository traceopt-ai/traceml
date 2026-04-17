"""
Compact end-of-run step-memory summary generation.

This module reads aligned step-memory telemetry from `step_memory_samples`,
reuses the shared step-memory diagnosis engine, and produces:

1. a compact text summary for end-of-run display and sharing
2. a structured JSON payload suitable for future compare features

Design goals
------------
- Keep the printed summary short and actionable
- Reuse the same diagnosis logic as live CLI and dashboard views
- Use a stable aligned tail window for printed diagnosis
"""

import sqlite3
from typing import Any, Dict, Optional

from traceml.aggregator.summaries.diagnosis_presentation import (
    diagnosis_presentation_to_json,
    present_step_memory_summary_diagnosis,
)
from traceml.aggregator.summaries.summary_formatting import (
    format_ratio_percent,
    safe_float,
)
from traceml.aggregator.summaries.summary_io import (
    append_text,
    load_json_or_empty,
    write_json,
)
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
    - ``head_avg_bytes`` maps to the canonical baseline average
    - ``tail_avg_bytes`` maps to the canonical recent average
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


def _metric_to_json(metric: StepMemoryCombinedMetric) -> Dict[str, Any]:
    """
    Serialize one step-memory metric into a compare-friendly JSON shape.
    """
    worst_trend = _head_tail_trend(metric.series.worst)
    median_trend = _head_tail_trend(metric.series.median)

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
        "trend": {
            "worst": worst_trend,
            "median": median_trend,
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


def _build_step_memory_card(
    *,
    training_steps: int,
    latest_step_observed: Optional[int],
    metrics: list[StepMemoryCombinedMetric],
    diagnosis: Optional[StepMemoryDiagnosis],
) -> tuple[str, Dict[str, Any]]:
    """
    Build a compact, shareable end-of-run step-memory summary.

    The printed summary is intentionally concise:
    - scope
    - diagnosis
    - primary metric details
    - one stable trend hint

    The JSON payload retains both allocated and reserved detail for compare.
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
            "primary_metric": None,
            "metrics": {},
            "units": {"memory": "bytes"},
            "card": card,
        }
        return card, summary

    primary_trend_worst = _head_tail_trend(primary.series.worst)

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
                "median": _head_tail_trend(primary.series.median),
            },
        },
        "metrics": {
            metric.metric: _metric_to_json(metric) for metric in sorted_metrics
        },
        "units": {"memory": "bytes"},
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
    finally:
        conn.close()

    card, step_memory_summary = _build_step_memory_card(
        training_steps=training_steps,
        latest_step_observed=latest_step_observed,
        metrics=metrics,
        diagnosis=diagnosis,
    )

    append_text(db_path + "_summary_card.txt", card)

    existing = load_json_or_empty(db_path + "_summary_card.json")
    existing["step_memory"] = step_memory_summary
    write_json(db_path + "_summary_card.json", existing)

    if print_to_stdout:
        print(card)

    return step_memory_summary
