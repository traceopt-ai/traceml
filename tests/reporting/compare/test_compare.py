from pathlib import Path
from typing import Optional

from traceml_ai.reporting.compare import build_compare_payload
from traceml_ai.reporting.compare import build_compare_text
from traceml_ai.reporting.compare.formatters import CompareTextFormatter

BYTES_PER_GB = 1024.0**3


def _base_payload() -> dict:
    return {
        "schema_version": 1,
        "duration_s": 120.0,
        "system": {
            "global": {
                "average": {
                    "cpu_percent": 25.0,
                    "ram_bytes": 4.0 * BYTES_PER_GB,
                    "gpu_util_percent": 50.0,
                    "gpu_mem_percent": 20.0,
                }
            },
        },
        "process": {
            "global": {
                "average": {
                    "cpu_percent": 80.0,
                    "ram_bytes": 2.0 * BYTES_PER_GB,
                }
            },
        },
    }


def _step_time_section(
    *,
    status: str = "BALANCED",
    reason: str = "No clear timing issue.",
    action: str = "Keep monitoring.",
    total_step_ms: float = 300.0,
    h2d_ms: Optional[float] = None,
    wait_ms: Optional[float] = None,
    split_ms: Optional[dict] = None,
) -> dict:
    splits = split_ms or {
        "dataloader": 24.0,
        "forward": 60.0,
        "backward": 180.0,
        "optimizer": 36.0,
    }
    compute_ms = splits["forward"] + splits["backward"] + splits["optimizer"]
    resolved_wait_ms = (
        max(
            0.0,
            total_step_ms
            - splits["dataloader"]
            - float(h2d_ms or 0.0)
            - compute_ms,
        )
        if wait_ms is None
        else wait_ms
    )
    average = {
        "total_step_ms": total_step_ms,
        "dataloader_ms": splits["dataloader"],
        "compute_ms": compute_ms,
        "wait_ms": resolved_wait_ms,
        "forward_ms": splits["forward"],
        "backward_ms": splits["backward"],
        "optimizer_ms": splits["optimizer"],
    }
    if h2d_ms is not None:
        average["h2d_ms"] = h2d_ms
    return {
        "diagnosis": {
            "status": status,
            "reason": reason,
            "action": action,
        },
        "global": {"average": average},
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
    median_peak = (
        median_peak_bytes
        if skew_pct <= 0.0
        else worst_peak_bytes / (1.0 + skew_pct / 100.0)
    )
    return {
        "diagnosis": {
            "status": status,
            "reason": reason,
            "action": action,
        },
        "global": {
            "median": {
                "peak_reserved_bytes": {
                    "value": median_peak,
                    "idx": "0",
                }
            },
            "worst": {
                "peak_reserved_bytes": {
                    "value": worst_peak_bytes,
                    "idx": "0",
                }
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
            total_step_ms=296.5,
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
            total_step_ms=296.5,
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
    assert "Total step" in text
    assert "n/a" in text
    assert "296.5 ms" in text
    assert "Peak reserved" in text
    assert "194 MB" in text


def test_compare_text_formatter_matches_public_wrapper() -> None:
    lhs = _payload_with_sections()
    rhs = _payload_with_sections(
        step_time=_step_time_section(total_step_ms=330.0),
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
            "diagnosis": {
                "status": "BALANCED",
                "reason": "No clear timing issue.",
                "action": "Keep monitoring.",
            },
            "global": {
                "average": {
                    "total_step_ms": 300.0,
                    "dataloader_ms": 24.0,
                    "forward_ms": 60.0,
                    "backward_ms": 180.0,
                    "optimizer_ms": 36.0,
                    "compute_ms": 276.0,
                },
            },
        },
        step_memory=_step_memory_section(),
    )
    rhs = _payload_with_sections(
        step_time=_step_time_section(
            status="WAIT-HEAVY",
            reason="Wait dominates total step.",
            action="Inspect synchronization and host stalls.",
            total_step_ms=301.0,
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
            total_step_ms=300.0,
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
            total_step_ms=301.0,
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
            total_step_ms=300.0,
        ),
    )
    rhs = _payload_with_sections(
        include_step_time=True,
        include_step_memory=True,
        step_time=_step_time_section(
            status="BALANCED",
            total_step_ms=300.5,
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
            total_step_ms=621.1,
            h2d_ms=2.0,
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
            total_step_ms=735.2,
            h2d_ms=2.4,
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
    step_time_metrics = compare_payload["sections"]["step_time"]["metrics"]
    assert step_time_metrics["total_step_ms"]["pct_change"] is not None
    assert "forward_ms" in step_time_metrics
    assert "backward_ms" in step_time_metrics
    assert "optimizer_ms" in step_time_metrics
    assert compare_payload["verdict"]["status"] == "REGRESSION"
    assert compare_payload["verdict"]["primary_domain"] == "step_time"
    assert "Metric" in text
    assert "Step time diagnosis" in text
    assert "Total step" in text
    assert "Input" in text
    assert "H2D" in text
    assert "Compute" in text
    assert "Wait" in text
    assert "Forward" not in text
    assert "Backward" not in text
    assert "Optimizer" not in text
    assert "621.1 ms" in text
    assert "735.2 ms" in text
    assert "+114.1 ms (+18.4%)" in text
    assert "Peak reserved" in text
    assert "+2.70 GB (+43.5%)" in text


def test_compare_shows_system_gpu_utilization_diagnosis_change() -> None:
    lhs = _payload_with_sections()
    rhs = _payload_with_sections()
    lhs["system"]["diagnosis"] = {"status": "NORMAL"}
    rhs["system"]["diagnosis"] = {"status": "MODERATE GPU UTILIZATION"}
    lhs["system"]["global"]["average"]["gpu_util_percent"] = 86.9
    rhs["system"]["global"]["average"]["gpu_util_percent"] = 37.8

    compare_payload = _build_compare(lhs, rhs)
    system = compare_payload["sections"]["system"]
    text = build_compare_text(compare_payload)

    assert system["diagnosis"] == {
        "lhs": "NORMAL",
        "rhs": "MODERATE GPU UTILIZATION",
        "changed": True,
    }
    assert (
        round(system["metrics"]["gpu_util_avg_percent"]["delta"], 1) == -49.1
    )
    assert "System diagnosis" in text
    assert "MODERATE GPU UTILIZATION" in text
    assert "GPU util avg" in text
    assert "-49.1 pp" in text


def test_compare_verdict_uses_priority_for_mixed_primary_signals() -> None:
    lhs = _payload_with_sections(
        step_time=_step_time_section(total_step_ms=700.0),
        step_memory=_step_memory_section(
            worst_peak_bytes=4.0 * 1024.0 * 1024.0 * 1024.0,
        ),
    )
    rhs = _payload_with_sections(
        step_time=_step_time_section(total_step_ms=600.0),
        step_memory=_step_memory_section(
            worst_peak_bytes=6.0 * 1024.0 * 1024.0 * 1024.0,
        ),
    )

    verdict = _build_compare(lhs, rhs)["verdict"]

    assert verdict["status"] == "MIXED"
    assert verdict["outcome"] == "mixed"
    assert verdict["findings"][0]["status"] == "MIXED"
