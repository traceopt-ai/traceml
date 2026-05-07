from pathlib import Path
from typing import Optional

from traceml.reporting.compare import build_compare_payload
from traceml.reporting.compare import build_compare_text
from traceml.reporting.compare.formatters import CompareTextFormatter


def _base_payload() -> dict:
    return {
        "duration_s": 120.0,
        "system": {
            "global": {
                "cpu": {"avg_percent": 25.0},
                "ram": {"peak_gb": 4.0},
                "gpu_rollup": {"available": True, "count": 1},
            },
        },
        "process": {
            "global": {
                "cpu": {"avg_percent": 80.0},
                "ram": {"peak_gb": 2.0},
                "takeaway": "CPU usage looks normal.",
            },
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
        "primary_diagnosis": {
            "status": status,
            "reason": reason,
            "action": action,
        },
        "global": {
            "typical": {
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
        "primary_diagnosis": {
            "status": status,
            "reason": reason,
            "action": action,
        },
        "global": {
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


def _legacy_payload(
    *,
    ranks_seen: int,
    step_avg_ms: float,
    wait_share_pct: float,
    status: Optional[str],
    cpu_avg_percent: float,
    process_cpu_avg_percent: float,
    ram_peak_gb: float = 9.8,
    process_ram_peak_gb: float = 2.5,
) -> dict:
    diagnosis = (
        None
        if status is None
        else {
            "status": status,
            "reason": "legacy diagnosis reason",
            "action": "legacy diagnosis action",
            "note": None,
        }
    )
    return {
        "duration_s": 34.1,
        "system": {
            "cpu_avg_percent": cpu_avg_percent,
            "ram_peak_gb": ram_peak_gb,
            "gpu_available": False,
            "gpu_count": 0,
        },
        "process": {
            "cpu_avg_percent": process_cpu_avg_percent,
            "ram_peak_gb": process_ram_peak_gb,
            "takeaway": "stable overall",
        },
        "step_time": {
            "ranks_seen": ranks_seen,
            "diagnosis_presented": diagnosis,
            "timing_primary": {
                "step_avg_ms": step_avg_ms,
                "compute_share_pct": 98.0,
                "wait_share_pct": wait_share_pct,
                "dominant_phase": "backward",
                "split_ms": {
                    "dataloader": 1.0,
                    "forward": 2.0,
                    "backward": 3.0,
                    "optimizer": 4.0,
                },
                "split_pct": {
                    "dataloader": 10.0,
                    "forward": 20.0,
                    "backward": 30.0,
                    "optimizer": 40.0,
                },
            },
            "median_split_ms": {
                "dataloader": 1.0,
                "forward": 2.0,
                "backward": 3.0,
                "optimizer": 4.0,
            },
            "median_split_pct": {
                "dataloader": 10.0,
                "forward": 20.0,
                "backward": 30.0,
                "optimizer": 40.0,
            },
        },
        "step_memory": {
            "diagnosis_presented": {
                "status": "NO DATA",
                "reason": "Need more aligned steps.",
                "action": "Too little aligned memory data was collected.",
                "note": None,
            },
            "primary_metric": {
                "metric": "peak_allocated",
                "worst_peak_bytes": 0.0,
                "median_peak_bytes": 0.0,
                "worst_rank": 0,
                "skew_pct": 0.0,
                "trend": {
                    "worst": {"delta_bytes": None},
                    "median": {"delta_bytes": None},
                },
            },
        },
    }


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

    assert "Verdict: INCONCLUSIVE" in text
    assert "Step avg" in text
    assert "n/a" in text
    assert "296.5 ms" in text
    assert "Peak reserved" in text
    assert "194 MB" in text


def test_compare_reads_legacy_flat_final_summary_schema() -> None:
    """
    Compare should read older flat final_summary.json artifacts while users may
    still compare historical runs.

    Remove this test when the legacy fallback readers in
    traceml.reporting.compare.core are intentionally deleted.
    """
    lhs = _legacy_payload(
        ranks_seen=3,
        step_avg_ms=2499.7,
        wait_share_pct=0.6,
        status=None,
        cpu_avg_percent=42.5,
        process_cpu_avg_percent=70.8,
    )
    rhs = _legacy_payload(
        ranks_seen=1,
        step_avg_ms=621.1,
        wait_share_pct=1.5,
        status="COMPUTE-BOUND",
        cpu_avg_percent=23.2,
        process_cpu_avg_percent=143.9,
        ram_peak_gb=10.9,
        process_ram_peak_gb=4.0,
    )

    compare_payload = _build_compare(lhs, rhs)
    text = build_compare_text(compare_payload)

    assert compare_payload["step_time"]["step_avg_ms"]["lhs"] == 2499.7
    assert compare_payload["step_time"]["step_avg_ms"]["rhs"] == 621.1
    assert compare_payload["step_time"]["wait_share_pct"]["lhs"] == 0.6
    assert compare_payload["step_time"]["wait_share_pct"]["rhs"] == 1.5
    assert compare_payload["system"]["cpu_avg_percent"]["lhs"] == 42.5
    assert compare_payload["system"]["cpu_avg_percent"]["rhs"] == 23.2
    assert compare_payload["process"]["cpu_avg_percent"]["lhs"] == 70.8
    assert compare_payload["process"]["cpu_avg_percent"]["rhs"] == 143.9
    assert "Step avg" in text
    assert "2499.7 ms" in text
    assert "621.1 ms" in text
    assert "unavailable in both runs" not in text


def test_compare_text_formatter_matches_public_wrapper() -> None:
    lhs = _payload_with_sections()
    rhs = _payload_with_sections(
        step_time=_step_time_section(step_avg_ms=330.0),
    )
    compare_payload = _build_compare(lhs, rhs)

    assert CompareTextFormatter().format(
        compare_payload
    ) == build_compare_text(compare_payload)


def test_compare_text_wrapper_returns_fallback_if_formatter_fails(
    monkeypatch,
) -> None:
    lhs = _payload_with_sections()
    rhs = _payload_with_sections()
    compare_payload = _build_compare(lhs, rhs)

    def _raise(_self, _payload):
        raise RuntimeError("boom")

    monkeypatch.setattr(CompareTextFormatter, "format", _raise)

    text = build_compare_text(compare_payload)

    assert "TraceML Compare" in text
    assert "detailed compare text formatting failed" in text


def test_compare_partial_step_time_stays_unclear() -> None:
    lhs = _payload_with_sections(
        step_time={
            "primary_diagnosis": {
                "status": "BALANCED",
                "reason": "No clear timing issue.",
                "action": "Keep monitoring.",
            },
            "global": {
                "typical": {
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
    assert any("partial" in item["summary"].lower() for item in top_changes)


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


def test_compare_payload_has_section_based_json_and_table_text() -> None:
    lhs = _payload_with_sections(
        step_time=_step_time_section(
            status="COMPUTE-BOUND",
            step_avg_ms=621.1,
            wait_share_pct=1.4,
            split_ms={
                "dataloader": 1.3,
                "forward": 228.4,
                "backward": 313.4,
                "optimizer": 69.0,
            },
        ),
        step_memory=_step_memory_section(
            status="NORMAL",
            worst_peak_bytes=6.2 * 1024.0 * 1024.0 * 1024.0,
            skew_pct=4.0,
        ),
    )
    rhs = _payload_with_sections(
        step_time=_step_time_section(
            status="COMPUTE-BOUND",
            step_avg_ms=735.2,
            wait_share_pct=1.6,
            split_ms={
                "dataloader": 1.9,
                "forward": 300.0,
                "backward": 350.0,
                "optimizer": 70.0,
            },
        ),
        step_memory=_step_memory_section(
            status="HIGH MEMORY",
            worst_peak_bytes=8.9 * 1024.0 * 1024.0 * 1024.0,
            skew_pct=12.0,
        ),
    )

    compare_payload = _build_compare(lhs, rhs)
    text = build_compare_text(compare_payload)

    assert compare_payload["schema_version"] == 2
    assert set(compare_payload["sections"]) == {
        "step_time",
        "step_memory",
        "process",
        "system",
    }
    assert (
        compare_payload["sections"]["step_time"]["metrics"]["step_avg_ms"][
            "pct_change"
        ]
        == compare_payload["step_time"]["step_avg_ms"]["pct_change"]
    )
    assert compare_payload["verdict"]["status"] == "REGRESSION"
    assert compare_payload["verdict"]["primary_domain"] == "step_time"
    assert "Metric" in text
    assert "Step time diagnosis" in text
    assert "Step avg" in text
    assert "621.1 ms" in text
    assert "735.2 ms" in text
    assert "+114.1 ms (+18.4%)" in text
    assert "Peak reserved" in text
    assert "+2.70 GB (+43.5%)" in text


def test_compare_verdict_uses_priority_for_mixed_primary_signals() -> None:
    lhs = _payload_with_sections(
        step_time=_step_time_section(step_avg_ms=700.0),
        step_memory=_step_memory_section(
            worst_peak_bytes=4.0 * 1024.0 * 1024.0 * 1024.0,
        ),
    )
    rhs = _payload_with_sections(
        step_time=_step_time_section(step_avg_ms=600.0),
        step_memory=_step_memory_section(
            worst_peak_bytes=6.0 * 1024.0 * 1024.0 * 1024.0,
        ),
    )

    verdict = _build_compare(lhs, rhs)["verdict"]

    assert verdict["status"] == "MIXED"
    assert verdict["outcome"] == "mixed"
    assert verdict["findings"][0]["status"] == "MIXED"
