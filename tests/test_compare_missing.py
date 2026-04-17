from pathlib import Path
from typing import Optional

from traceml.compare.core import build_compare_payload
from traceml.compare.render import build_compare_text


def _base_payload() -> dict:
    return {
        "duration_s": 120.0,
        "system": {
            "cpu_avg_percent": 25.0,
            "ram_peak_gb": 4.0,
            "gpu_available": True,
            "gpu_count": 1,
        },
        "process": {
            "cpu_avg_percent": 80.0,
            "ram_peak_gb": 2.0,
            "takeaway": "CPU usage looks normal.",
        },
    }


def _step_time_section(
    *,
    status: str = "BALANCED",
    reason: str = "No clear timing issue.",
    action: str = "Keep monitoring.",
    step_avg_ms: float = 300.0,
    wait_share_pct: float = 10.0,
    compute_share_pct: float = 90.0,
    dominant_phase: str = "backward",
    split_pct: Optional[dict] = None,
    split_ms: Optional[dict] = None,
) -> dict:
    return {
        "diagnosis": {
            "status": status,
        },
        "diagnosis_presented": {
            "status": status,
            "reason": reason,
            "action": action,
        },
        "timing_primary": {
            "step_avg_ms": step_avg_ms,
            "wait_share_pct": wait_share_pct,
            "compute_share_pct": compute_share_pct,
            "dominant_phase": dominant_phase,
            "split_pct": split_pct
            or {
                "dataloader": 8.0,
                "forward": 20.0,
                "backward": 60.0,
                "optimizer": 12.0,
            },
            "split_ms": split_ms
            or {
                "dataloader": 24.0,
                "forward": 60.0,
                "backward": 180.0,
                "optimizer": 36.0,
            },
        },
    }


def _step_memory_section(
    *,
    status: str = "BALANCED",
    reason: str = "No clear pressure, imbalance, or creep signal.",
    action: str = "Keep monitoring.",
    metric: str = "peak_reserved",
    worst_peak_bytes: float = 256.0 * 1024.0 * 1024.0,
    median_peak_bytes: float = 220.0 * 1024.0 * 1024.0,
    skew_pct: float = 0.0,
    trend_delta_bytes: float = 0.0,
) -> dict:
    return {
        "diagnosis": {
            "status": status,
        },
        "diagnosis_presented": {
            "status": status,
            "reason": reason,
            "action": action,
        },
        "primary_metric": {
            "metric": metric,
            "worst_peak_bytes": worst_peak_bytes,
            "median_peak_bytes": median_peak_bytes,
            "worst_rank": 0,
            "skew_pct": skew_pct,
            "trend": {
                "worst": {
                    "delta_bytes": trend_delta_bytes,
                },
                "median": {
                    "delta_bytes": trend_delta_bytes * 0.5,
                },
            },
        },
    }


def _payload_with_sections(
    *,
    include_step_time: bool = True,
    include_step_memory: bool = True,
    step_time: Optional[dict] = None,
    step_memory: Optional[dict] = None,
) -> dict:
    payload = _base_payload()
    if include_step_time:
        payload["step_time"] = step_time or _step_time_section()
    if include_step_memory:
        payload["step_memory"] = step_memory or _step_memory_section()
    return payload


def _build_compare(lhs: dict, rhs: dict) -> dict:
    return build_compare_payload(
        lhs_payload=lhs,
        rhs_payload=rhs,
        lhs_path=Path("/tmp/run_a/final_summary.json"),
        rhs_path=Path("/tmp/run_b/final_summary.json"),
    )


def test_compare_missing_both_primary_sections_on_lhs_is_unclear() -> None:
    lhs = _payload_with_sections(
        include_step_time=False,
        include_step_memory=False,
    )
    rhs = _payload_with_sections(
        step_time=_step_time_section(
            status="INPUT STRAGGLER",
            reason="r0 has excess dataloader burden.",
            action="Inspect dataloader imbalance.",
            step_avg_ms=296.5,
            wait_share_pct=13.5,
        ),
        step_memory=_step_memory_section(
            status="BALANCED",
            worst_peak_bytes=194.0 * 1024.0 * 1024.0,
            skew_pct=0.0,
        ),
    )

    compare_payload = _build_compare(lhs, rhs)
    verdict = compare_payload["verdict"]

    assert verdict["outcome"] == "unclear"
    assert verdict["severity"] == "info"
    assert verdict["comparability"]["overall"]["state"] == "insufficient"
    assert verdict["comparability"]["step_time"]["state"] == "missing_one_side"
    assert (
        verdict["comparability"]["step_memory"]["state"] == "missing_one_side"
    )
    assert "missing on run A" in verdict["why"]
    assert "matching TraceML summary coverage" in verdict["action"]

    top_changes = verdict["top_changes"]
    assert top_changes
    assert top_changes[0]["domain"] == "compare"
    assert "not comparable" in top_changes[0]["summary"]


def test_compare_render_shows_unavailable_in_a_for_missing_numeric_fields() -> (
    None
):
    lhs = _payload_with_sections(
        include_step_time=False,
        include_step_memory=False,
    )
    rhs = _payload_with_sections(
        step_time=_step_time_section(
            status="INPUT STRAGGLER",
            reason="r0 has excess dataloader burden (~86.6% of a typical local step).",
            step_avg_ms=296.5,
            wait_share_pct=13.5,
        ),
        step_memory=_step_memory_section(
            status="BALANCED",
            worst_peak_bytes=194.0 * 1024.0 * 1024.0,
            skew_pct=0.0,
        ),
    )

    compare_payload = _build_compare(lhs, rhs)
    text = build_compare_text(compare_payload)

    assert "No clear comparison outcome." in text
    assert "Step avg: unavailable in A; B = 296.5ms" in text
    assert "Wait share: unavailable in A; B = 13.5%" in text
    assert "Worst peak: unavailable in A; B = 194 MB" in text


def test_compare_partial_step_time_stays_unclear_and_surfaces_partial_comparability() -> (
    None
):
    lhs = _payload_with_sections(
        step_time={
            "diagnosis": {"status": "BALANCED"},
            "diagnosis_presented": {
                "status": "BALANCED",
                "reason": "No clear timing issue.",
                "action": "Keep monitoring.",
            },
            "timing_primary": {
                "step_avg_ms": 300.0,
                "dominant_phase": "backward",
                "split_pct": {
                    "dataloader": 8.0,
                    "forward": 20.0,
                    "backward": 60.0,
                    "optimizer": 12.0,
                },
                "split_ms": {
                    "dataloader": 24.0,
                    "forward": 60.0,
                    "backward": 180.0,
                    "optimizer": 36.0,
                },
            },
        },
        step_memory=_step_memory_section(),
    )
    rhs = _payload_with_sections(
        step_time=_step_time_section(
            status="WAIT-HEAVY",
            reason="WAIT dominates the traced step.",
            action="Inspect synchronization and host stalls.",
            step_avg_ms=301.0,
            wait_share_pct=22.0,
        ),
        step_memory=_step_memory_section(),
    )

    compare_payload = _build_compare(lhs, rhs)
    verdict = compare_payload["verdict"]

    assert verdict["outcome"] == "unclear"
    assert verdict["severity"] == "info"
    assert verdict["comparability"]["step_time"]["state"] == "partial"
    assert verdict["comparability"]["overall"]["state"] == "partial"
    assert "partial" in verdict["summary"].lower()
    assert (
        "partial" in verdict["why"].lower()
        or "missing" in verdict["why"].lower()
    )

    top_changes = verdict["top_changes"]
    assert top_changes
    assert any(
        item["domain"] == "compare"
        and "partially comparable" in item["summary"]
        for item in top_changes
    )


def test_compare_fully_comparable_stable_runs_can_still_be_equivalent() -> (
    None
):
    lhs = _payload_with_sections(
        step_time=_step_time_section(
            status="BALANCED",
            step_avg_ms=300.0,
            wait_share_pct=10.0,
        ),
        step_memory=_step_memory_section(
            status="BALANCED",
            worst_peak_bytes=256.0 * 1024.0 * 1024.0,
            skew_pct=0.0,
            trend_delta_bytes=0.0,
        ),
    )
    rhs = _payload_with_sections(
        step_time=_step_time_section(
            status="BALANCED",
            step_avg_ms=301.0,
            wait_share_pct=10.1,
        ),
        step_memory=_step_memory_section(
            status="BALANCED",
            worst_peak_bytes=258.0 * 1024.0 * 1024.0,
            skew_pct=0.1,
            trend_delta_bytes=8.0 * 1024.0 * 1024.0,
        ),
    )

    compare_payload = _build_compare(lhs, rhs)
    verdict = compare_payload["verdict"]

    assert verdict["comparability"]["overall"]["state"] == "comparable"
    assert verdict["outcome"] == "equivalent"
    assert verdict["severity"] == "info"
    assert (
        "unchanged" in verdict["why"].lower()
        or "stable" in verdict["why"].lower()
    )


def test_compare_one_comparable_domain_and_one_missing_domain_is_not_equivalent() -> (
    None
):
    lhs = _payload_with_sections(
        include_step_time=True,
        include_step_memory=False,
        step_time=_step_time_section(
            status="BALANCED",
            step_avg_ms=300.0,
            wait_share_pct=10.0,
        ),
    )
    rhs = _payload_with_sections(
        include_step_time=True,
        include_step_memory=True,
        step_time=_step_time_section(
            status="BALANCED",
            step_avg_ms=300.5,
            wait_share_pct=10.2,
        ),
        step_memory=_step_memory_section(
            status="BALANCED",
            worst_peak_bytes=220.0 * 1024.0 * 1024.0,
        ),
    )

    compare_payload = _build_compare(lhs, rhs)
    verdict = compare_payload["verdict"]

    assert verdict["comparability"]["step_time"]["state"] == "comparable"
    assert (
        verdict["comparability"]["step_memory"]["state"] == "missing_one_side"
    )
    assert verdict["comparability"]["overall"]["state"] == "partial"
    assert verdict["outcome"] == "unclear"
    assert verdict["outcome"] != "equivalent"
