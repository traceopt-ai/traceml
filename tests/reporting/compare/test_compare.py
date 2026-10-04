import json
from pathlib import Path
from typing import Optional

import pytest

import traceml_ai.launcher.commands as launcher_commands
from traceml_ai.diagnostics.step_time.api import _STATUS_BY_KIND
from traceml_ai.launcher.cli import build_parser
from traceml_ai.reporting.compare import (
    build_compare_payload,
    build_compare_text,
    compare_summaries,
)
from traceml_ai.reporting.compare.formatters import CompareTextFormatter
from traceml_ai.reporting.compare.io import load_summary_json
from traceml_ai.reporting.compare.policy import (
    _STEP_TIME_STATUS_RANK,
    build_step_time_ci_policy,
    ci_policy_ineligibility_reason,
    evaluate_step_time_ci_result,
    extract_step_time_ci_evidence,
    parse_step_time_regression_threshold,
)

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
    input_wait_ms: Optional[float] = None,
    h2d_ms: Optional[float] = None,
    residual_ms: Optional[float] = None,
    split_ms: Optional[dict] = None,
) -> dict:
    splits = split_ms or {
        "dataloader": 24.0,
        "forward": 60.0,
        "backward": 180.0,
        "optimizer": 36.0,
    }
    compute_ms = splits["forward"] + splits["backward"] + splits["optimizer"]
    resolved_residual_ms = (
        max(
            0.0,
            total_step_ms
            - splits["dataloader"]
            - float(h2d_ms or 0.0)
            - compute_ms,
        )
        if residual_ms is None
        else residual_ms
    )
    average = {
        "total_step_ms": total_step_ms,
        "dataloader_ms": splits["dataloader"],
        "compute_ms": compute_ms,
        "residual_ms": resolved_residual_ms,
        "forward_ms": splits["forward"],
        "backward_ms": splits["backward"],
        "optimizer_ms": splits["optimizer"],
    }
    if input_wait_ms is not None:
        average["input_wait_ms"] = input_wait_ms
    if h2d_ms is not None:
        average["h2d_ms"] = h2d_ms
    return {
        "diagnosis": {
            "status": status,
            "reason": reason,
            "action": action,
        },
        "global": {
            "window": {"diagnosis_clock": "cpu"},
            "average": average,
        },
    }


def _canonical_step_time_section(
    *,
    step_time_ms: float,
    diagnosis_clock: str,
    step_time_cpu_ms: Optional[float],
    step_time_gpu_ms: Optional[float],
    dataloader_fetch_cpu_ms: Optional[float] = None,
) -> dict:
    """Build a minimal schema-1.7 Step Time section for compare coverage."""
    return {
        "diagnosis": {"status": "BALANCED"},
        "global": {
            "window": {"diagnosis_clock": diagnosis_clock},
            "average": {
                "step_time_ms": step_time_ms,
                "traced_step_time_ms": step_time_ms - 10.0,
                "step_time_cpu_ms": step_time_cpu_ms,
                "step_time_gpu_ms": step_time_gpu_ms,
                "dataloader_fetch_cpu_ms": dataloader_fetch_cpu_ms,
                "input_wait_ms": 10.0,
                "h2d_ms": 5.0,
                "compute_ms": step_time_ms - 15.0,
                "residual_ms": 0.0,
                "forward_ms": step_time_ms - 15.0,
                "backward_ms": 0.0,
                "optimizer_ms": 0.0,
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


def _set_primary(payload: dict, status: str) -> dict:
    payload["primary_diagnosis"] = {
        "kind": status.replace("-", "_").replace(" ", "_"),
        "status": status,
        "severity": "warn",
        "section": "step_time",
        "scope": "performance",
        "summary": "primary summary",
        "action": "primary action",
        "evidence": {},
    }
    return payload


def _build_compare(lhs: dict, rhs: dict) -> dict:
    return build_compare_payload(
        lhs_payload=lhs,
        rhs_payload=rhs,
        lhs_path=Path("/tmp/run_a/final_summary.json"),
        rhs_path=Path("/tmp/run_b/final_summary.json"),
    )


def _ci_payload() -> dict:
    payload = _payload_with_sections()
    payload["run_context"] = {
        "run": {"status": "completed", "profile": "run"},
        "declaration": {
            "schema_version": 1,
            "workload": {"name": "image-training", "parameters": {}},
            "measurement": {"start_step": 1, "completed_steps": 20},
        },
        "execution": {
            "expected_nodes": 1,
            "processes_per_node": 2,
            "expected_world_size": 2,
            "launcher_completion": {
                "status": "completed",
                "nodes_observed": 1,
                "reason_codes": [],
            },
        },
    }
    return payload


def _ci_ineligibility_reason(lhs: dict, rhs: dict) -> Optional[str]:
    return ci_policy_ineligibility_reason(
        lhs_payload=lhs,
        rhs_payload=rhs,
        compare_payload=_build_compare(lhs, rhs),
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", 0.0),
        ("5", 5.0),
        ("2.5", 2.5),
    ],
)
def test_ci_policy_parses_supported_thresholds(value, expected) -> None:
    assert parse_step_time_regression_threshold(value) == expected


@pytest.mark.parametrize("value", [None, True, "", "-1", "nan", "inf", "x"])
def test_ci_policy_rejects_invalid_thresholds(value) -> None:
    with pytest.raises(RuntimeError, match="finite, nonnegative"):
        parse_step_time_regression_threshold(value)


@pytest.mark.parametrize(
    ("delta_pct", "expected"),
    [
        (5.01, "SLOWER_IN_THIS_PAIR"),
        (5.0, "WITHIN_THRESHOLD_IN_THIS_PAIR"),
        (0.0, "WITHIN_THRESHOLD_IN_THIS_PAIR"),
        (-5.0, "WITHIN_THRESHOLD_IN_THIS_PAIR"),
        (-5.01, "FASTER_IN_THIS_PAIR"),
    ],
)
def test_ci_policy_classifies_threshold_boundaries(
    delta_pct, expected
) -> None:
    # These are the v0.1 decisions; broader statistical analysis is out of scope.
    assert (
        evaluate_step_time_ci_result(delta_pct=delta_pct, threshold_pct=5.0)
        == expected
    )


@pytest.mark.parametrize(
    ("reference_ms", "candidate_ms"),
    [(0.3, 0.315), (0.7, 0.735), (0.3, 0.285)],
)
def test_ci_policy_keeps_computed_threshold_boundaries_inclusive(
    reference_ms, candidate_ms
) -> None:
    lhs = _ci_payload()
    rhs = _ci_payload()
    lhs["step_time"]["global"]["average"]["total_step_ms"] = reference_ms
    rhs["step_time"]["global"]["average"]["total_step_ms"] = candidate_ms

    policy = build_step_time_ci_policy(
        lhs_payload=lhs,
        rhs_payload=rhs,
        compare_payload=_build_compare(lhs, rhs),
        threshold="5",
    )

    assert policy["result"] == "WITHIN_THRESHOLD_IN_THIS_PAIR"


def test_ci_policy_accepts_complete_matching_evidence() -> None:
    assert _ci_ineligibility_reason(_ci_payload(), _ci_payload()) is None


def test_ci_policy_reuses_existing_step_time_evidence() -> None:
    lhs = _ci_payload()
    rhs = _ci_payload()
    lhs["step_time"]["global"]["window"]["steps_analyzed"] = 20
    rhs["step_time"]["global"]["window"]["steps_analyzed"] = 17
    rhs["step_time"]["global"]["average"]["total_step_ms"] = 330.0

    evidence = extract_step_time_ci_evidence(
        lhs_payload=lhs,
        rhs_payload=rhs,
        compare_payload=_build_compare(lhs, rhs),
    )

    assert evidence == {
        "selected_clock": "cpu",
        "reference_step_time_ms": 300.0,
        "candidate_step_time_ms": 330.0,
        "pct_change": pytest.approx(10.0),
        "reference_steps_analyzed": 20,
        "candidate_steps_analyzed": 17,
    }


def test_ci_policy_builds_one_block_from_existing_evidence() -> None:
    lhs = _ci_payload()
    rhs = _ci_payload()
    lhs["step_time"]["global"]["window"]["steps_analyzed"] = 20
    rhs["step_time"]["global"]["window"]["steps_analyzed"] = 17
    rhs["step_time"]["global"]["average"]["total_step_ms"] = 330.0
    compare_payload = _build_compare(lhs, rhs)

    policy = build_step_time_ci_policy(
        lhs_payload=lhs,
        rhs_payload=rhs,
        compare_payload=compare_payload,
        threshold="5",
    )

    assert policy == {
        "threshold_pct": 5.0,
        "selected_clock": "cpu",
        "reference_step_time_ms": 300.0,
        "candidate_step_time_ms": 330.0,
        "pct_change": pytest.approx(10.0),
        "reference_steps_analyzed": 20,
        "candidate_steps_analyzed": 17,
        "result": "SLOWER_IN_THIS_PAIR",
    }


def test_ci_policy_records_inconclusive_reason() -> None:
    lhs = _ci_payload()
    rhs = _ci_payload()
    rhs["run_context"].pop("declaration")

    policy = build_step_time_ci_policy(
        lhs_payload=lhs,
        rhs_payload=rhs,
        compare_payload=_build_compare(lhs, rhs),
        threshold="5",
    )

    assert policy["result"] == "INCONCLUSIVE"
    assert policy["reason"] == (
        "Both summaries must contain a valid guard declaration."
    )


def test_compare_writes_ci_policy_only_when_requested(
    tmp_path, capsys
) -> None:
    lhs = _ci_payload()
    rhs = _ci_payload()
    lhs["step_time"]["global"]["window"]["steps_analyzed"] = 20
    rhs["step_time"]["global"]["window"]["steps_analyzed"] = 17
    rhs["step_time"]["global"]["average"]["total_step_ms"] = 330.0
    lhs_path = tmp_path / "reference.json"
    rhs_path = tmp_path / "candidate.json"
    lhs_path.write_text(json.dumps(lhs), encoding="utf-8")
    rhs_path.write_text(json.dumps(rhs), encoding="utf-8")

    exploratory = compare_summaries(
        lhs_path,
        rhs_path,
        output=tmp_path / "exploratory",
        print_to_stdout=False,
    )
    evaluated = compare_summaries(
        lhs_path,
        rhs_path,
        output=tmp_path / "evaluated",
        max_step_time_regression_pct="5",
        print_to_stdout=True,
    )
    stdout = capsys.readouterr().out

    assert "ci_policy" not in exploratory
    assert "CI Policy" not in exploratory["text"]
    assert evaluated["ci_policy"]["result"] == "SLOWER_IN_THIS_PAIR"
    assert "CI Policy" in evaluated["text"]
    assert "Result: SLOWER_IN_THIS_PAIR" in evaluated["text"]
    assert "Threshold: 5.0% maximum Step Time regression" in evaluated["text"]
    assert "Evidence: CPU Step Time 300.0 ms -> 330.0 ms (+10.0%)" in (
        evaluated["text"]
    )
    assert "Analyzed steps: 20 -> 17" in evaluated["text"]
    assert evaluated["text"] in stdout
    persisted = json.loads((tmp_path / "evaluated.json").read_text())
    assert persisted["ci_policy"] == evaluated["ci_policy"]
    assert (tmp_path / "evaluated.txt").read_text().rstrip() == evaluated[
        "text"
    ]


def test_ci_regression_exits_after_writing_compare_artifacts(tmp_path) -> None:
    lhs = _ci_payload()
    rhs = _ci_payload()
    rhs["step_time"]["global"]["average"]["total_step_ms"] = 330.0
    lhs_path = tmp_path / "reference.json"
    rhs_path = tmp_path / "candidate.json"
    output = tmp_path / "ci-result"
    lhs_path.write_text(json.dumps(lhs), encoding="utf-8")
    rhs_path.write_text(json.dumps(rhs), encoding="utf-8")
    args = build_parser().parse_args(
        [
            "compare",
            str(lhs_path),
            str(rhs_path),
            "--max-step-time-regression-pct",
            "5",
            "--output",
            str(output),
        ]
    )

    # The command must preserve its evidence before returning the CI failure.
    with pytest.raises(SystemExit) as exc_info:
        launcher_commands.run_compare(args)

    assert exc_info.value.code == 4
    assert output.with_suffix(".txt").is_file()
    persisted = json.loads(output.with_suffix(".json").read_text())
    assert persisted["ci_policy"]["result"] == "SLOWER_IN_THIS_PAIR"


def test_compare_artifacts_keep_dotted_run_labels(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    paths = []
    for run_name in ("lr_0.1", "lr_0.01"):
        run_dir = tmp_path / "logs" / run_name
        run_dir.mkdir(parents=True)
        path = run_dir / "final_summary.json"
        path.write_text(json.dumps(_ci_payload()), encoding="utf-8")
        paths.append(path)

    result = compare_summaries(*paths, print_to_stdout=False)

    compare_dir = (tmp_path / "compare").resolve()
    assert result["artifacts"]["json"] == str(
        compare_dir / "lr_0.1_vs_lr_0.01.json"
    )
    assert result["artifacts"]["txt"] == str(
        compare_dir / "lr_0.1_vs_lr_0.01.txt"
    )
    assert sorted(p.name for p in compare_dir.iterdir()) == [
        "lr_0.1_vs_lr_0.01.json",
        "lr_0.1_vs_lr_0.01.txt",
    ]


def test_compare_text_renders_ci_policy_inconclusive_reason() -> None:
    lhs = _ci_payload()
    rhs = _ci_payload()
    rhs["run_context"].pop("declaration")
    compare_payload = _build_compare(lhs, rhs)
    compare_payload["ci_policy"] = build_step_time_ci_policy(
        lhs_payload=lhs,
        rhs_payload=rhs,
        compare_payload=compare_payload,
        threshold="5",
    )

    text = build_compare_text(compare_payload)

    assert "Result: INCONCLUSIVE" in text
    assert (
        "Reason: Both summaries must contain a valid guard declaration."
        in text
    )


def test_ci_policy_requires_matching_declarations() -> None:
    lhs = _ci_payload()
    rhs = _ci_payload()
    rhs["run_context"].pop("declaration")
    assert _ci_ineligibility_reason(lhs, rhs) == (
        "Both summaries must contain a valid guard declaration."
    )

    rhs = _ci_payload()
    rhs["run_context"]["declaration"]["workload"]["name"] = "other-training"
    assert _ci_ineligibility_reason(lhs, rhs) == (
        "The guard declarations do not match."
    )


def test_ci_policy_requires_matching_topology() -> None:
    lhs = _ci_payload()
    rhs = _ci_payload()
    rhs["run_context"]["execution"].pop("expected_world_size")
    assert _ci_ineligibility_reason(lhs, rhs) == (
        "Both summaries must contain a valid expected topology."
    )

    rhs = _ci_payload()
    rhs["run_context"]["execution"].update(
        expected_nodes=2,
        processes_per_node=2,
        expected_world_size=4,
    )
    assert _ci_ineligibility_reason(lhs, rhs) == (
        "The expected topologies do not match."
    )


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_ci_policy_requires_completed_training(status) -> None:
    lhs = _ci_payload()
    rhs = _ci_payload()
    rhs["run_context"]["run"]["status"] = status

    assert _ci_ineligibility_reason(lhs, rhs) == (
        "Training must be completed in both summaries."
    )


def test_ci_policy_requires_completed_launcher_outcome() -> None:
    lhs = _ci_payload()
    rhs = _ci_payload()
    rhs["run_context"]["execution"]["launcher_completion"][
        "status"
    ] = "incomplete"

    assert _ci_ineligibility_reason(lhs, rhs) == (
        "Launcher completion must be recorded for both summaries."
    )


@pytest.mark.parametrize("step_time_ms", [None, 0.0, float("inf")])
def test_ci_policy_requires_positive_finite_common_clock_step_time(
    step_time_ms,
) -> None:
    lhs = _ci_payload()
    rhs = _ci_payload()
    rhs["step_time"]["global"]["average"]["total_step_ms"] = step_time_ms

    assert _ci_ineligibility_reason(lhs, rhs) == (
        "Positive Step Time is required on a common CPU or GPU clock."
    )


def test_compare_ignores_additive_run_context() -> None:
    lhs = _payload_with_sections()
    rhs = _payload_with_sections()
    expected = _build_compare(lhs, rhs)
    lhs["run_context"] = {"run": {"status": "completed"}}
    rhs["run_context"] = {"run": {"status": "failed"}}

    actual = _build_compare(lhs, rhs)

    assert actual["sections"] == expected["sections"]
    assert actual["overview"] == expected["overview"]
    assert actual["verdict"] == expected["verdict"]


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
    assert verdict["comparability"]["step_time"]["state"] == "missing_both"
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
    assert "Step Time" in text
    assert "n/a" in text
    assert "no common comparison clock" in text
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


def test_compare_loader_still_requires_final_summary_sections(
    tmp_path,
) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1.4,
                "system": {},
                "process": {},
                "step_time": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="missing required section 'step_memory'",
    ):
        load_summary_json(summary_path)


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
            status="RESIDUAL-HEAVY",
            reason="Residual time dominates total step.",
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
    assert step_time_metrics["step_time_ms"]["pct_change"] is not None
    assert "forward_ms" in step_time_metrics
    assert "backward_ms" in step_time_metrics
    assert "optimizer_ms" in step_time_metrics
    assert compare_payload["verdict"]["status"] == "REGRESSION"
    assert compare_payload["verdict"]["primary_domain"] == "step_time"
    assert "Metric" in text
    assert "Step time diagnosis" in text
    assert "Step Time" in text
    assert "Input" not in text
    assert "DataLoader Fetch (CPU)" in text
    assert "H2D" in text
    assert "Compute" in text
    assert "Residual" in text
    assert "Forward" not in text
    assert "Backward" not in text
    assert "Optimizer" not in text
    assert "621.1 ms" in text
    assert "735.2 ms" in text
    assert "+114.1 ms (+18.4%)" in text
    assert "Peak reserved" in text
    assert "+2.70 GB (+43.5%)" in text


def test_compare_text_scales_a_memory_decrease_like_an_increase() -> None:
    lhs = _payload_with_sections(
        step_memory=_step_memory_section(
            worst_peak_bytes=8.9 * 1024.0 * 1024.0 * 1024.0,
        ),
    )
    rhs = _payload_with_sections(
        step_memory=_step_memory_section(
            worst_peak_bytes=6.2 * 1024.0 * 1024.0 * 1024.0,
        ),
    )

    text = build_compare_text(_build_compare(lhs, rhs))

    assert "-2.70 GB (-30.3%)" in text


def test_compare_warns_when_summary_schema_versions_differ() -> None:
    lhs = _payload_with_sections()
    rhs = _payload_with_sections()
    lhs["schema_version"] = 1.5
    rhs["schema_version"] = 1.6

    compare_payload = _build_compare(lhs, rhs)
    text = build_compare_text(compare_payload)

    assert compare_payload["lhs"]["schema_version"] == 1.5
    assert compare_payload["rhs"]["schema_version"] == 1.6
    assert compare_payload["warnings"] == [
        (
            "Summary schema versions differ: A uses 1.5, B uses 1.6. "
            "Step Time fields changed in schema 1.6 and were made nullable "
            "and canonical in schema 1.7; compare uses a "
            "common measured GPU clock when possible and otherwise a "
            "common CPU clock."
        )
    ]
    assert "Notes" in text
    assert "Summary schema versions differ: A uses 1.5, B uses 1.6." in text


def test_compare_step_time_input_prefers_selected_input_wait() -> None:
    lhs = _payload_with_sections(
        step_time=_step_time_section(
            total_step_ms=120.0,
            input_wait_ms=5.0,
            split_ms={
                "dataloader": 100.0,
                "forward": 10.0,
                "backward": 1.0,
                "optimizer": 1.0,
            },
        )
    )
    rhs = _payload_with_sections(
        step_time=_step_time_section(
            total_step_ms=120.0,
            input_wait_ms=7.0,
            split_ms={
                "dataloader": 100.0,
                "forward": 10.0,
                "backward": 1.0,
                "optimizer": 1.0,
            },
        )
    )

    step_time = _build_compare(lhs, rhs)["sections"]["step_time"]

    assert step_time["metrics"]["input_ms"]["lhs"] == 5.0
    assert step_time["metrics"]["input_ms"]["rhs"] == 7.0
    assert step_time["metrics"]["dominant_phase"]["lhs"] == "forward"
    assert step_time["metrics"]["dominant_phase"]["rhs"] == "forward"


def test_compare_step_time_null_input_wait_is_not_borrowed_from_dataloader() -> (
    None
):
    # Schema >= 1.6 always carries the input_wait_ms key; a present-but-
    # null value means the signal was genuinely never measured this
    # window (e.g. GPU clock dropped it on some step), not "old payload
    # without the key" -- the fallback must not silently substitute
    # dataloader_ms for it.
    section = _step_time_section(total_step_ms=120.0)
    section["global"]["average"]["input_wait_ms"] = None

    payload = _payload_with_sections(step_time=section)
    step_time = _build_compare(payload, payload)["sections"]["step_time"]

    assert step_time["metrics"]["input_ms"]["lhs"] is None
    assert step_time["metrics"]["input_ms"]["rhs"] is None


def test_compare_legacy_dataloader_maps_to_cpu_fetch_not_input_wait() -> None:
    # True pre-1.6 shape: Input Wait did not exist. Historical dataloader_ms
    # is CPU supplemental DataLoader-fetch timing, not Input Wait.
    section = _step_time_section(total_step_ms=120.0)
    assert "input_wait_ms" not in section["global"]["average"]

    payload = _payload_with_sections(step_time=section)
    step_time = _build_compare(payload, payload)["sections"]["step_time"]

    assert step_time["metrics"]["input_ms"]["lhs"] is None
    assert step_time["metrics"]["input_ms"]["rhs"] is None
    assert step_time["metrics"]["dataloader_fetch_cpu_ms"]["lhs"] == (
        section["global"]["average"]["dataloader_ms"]
    )
    assert step_time["metrics"]["dataloader_fetch_cpu_ms"]["rhs"] == (
        section["global"]["average"]["dataloader_ms"]
    )


def test_compare_includes_top_level_primary_diagnosis_change() -> None:
    lhs = _set_primary(
        _payload_with_sections(
            step_time=_step_time_section(status="INPUT-BOUND"),
        ),
        "INPUT-BOUND",
    )
    rhs = _set_primary(
        _payload_with_sections(
            step_time=_step_time_section(status="COMPUTE-BOUND"),
        ),
        "COMPUTE-BOUND",
    )

    compare_payload = _build_compare(lhs, rhs)
    text = build_compare_text(compare_payload)

    assert compare_payload["overview"]["primary_diagnosis"] == {
        "lhs": "INPUT-BOUND",
        "rhs": "COMPUTE-BOUND",
        "changed": True,
    }
    assert "Primary diagnosis" in text
    assert "INPUT-BOUND -> COMPUTE-BOUND" in text
    assert compare_payload["verdict"]["primary_domain"] == "compare"


def test_compare_shows_system_gpu_utilization_diagnosis_change() -> None:
    lhs = _payload_with_sections()
    rhs = _payload_with_sections()
    lhs["system"]["diagnosis"] = {"status": "NORMAL"}
    rhs["system"]["diagnosis"] = {"status": "MODERATE GPU UTIL"}
    lhs["system"]["global"]["average"]["gpu_util_percent"] = 86.9
    rhs["system"]["global"]["average"]["gpu_util_percent"] = 37.8

    compare_payload = _build_compare(lhs, rhs)
    system = compare_payload["sections"]["system"]
    text = build_compare_text(compare_payload)

    assert system["diagnosis"] == {
        "lhs": "NORMAL",
        "rhs": "MODERATE GPU UTIL",
        "changed": True,
    }
    assert (
        round(system["metrics"]["gpu_util_avg_percent"]["delta"], 1) == -49.1
    )
    assert "System diagnosis" in text
    assert "MODERATE GPU UTIL" in text
    assert "GPU util avg" in text
    assert "-49.1 pp" in text


def test_every_emittable_step_time_status_has_a_rank() -> None:
    # Guard against silent rank-0 fallback: any diagnosis status the engine
    # can emit must be mapped, or step_time_status_rank() defaults it to 0 and
    # a real regression (e.g. BALANCED -> H2D STRAGGLER) reads as no change.
    unmapped = sorted(
        status
        for status in _STATUS_BY_KIND.values()
        if status not in _STEP_TIME_STATUS_RANK
    )
    assert unmapped == []


def test_compare_flags_regression_for_h2d_straggler() -> None:
    lhs = _payload_with_sections(
        step_time=_step_time_section(
            status="BALANCED",
            total_step_ms=300.0,
            h2d_ms=2.0,
        ),
    )
    rhs = _payload_with_sections(
        step_time=_step_time_section(
            status="H2D STRAGGLER",
            reason="r0 spends excess time on host-to-device copies.",
            action="Inspect H2D transfer imbalance.",
            total_step_ms=300.0,
            h2d_ms=2.0,
        ),
    )

    verdict = _build_compare(lhs, rhs)["verdict"]

    assert verdict["status"] == "REGRESSION"
    assert any(
        "H2D STRAGGLER" in finding.get("why", "")
        for finding in verdict["findings"]
    )


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


def test_compare_normalizes_legacy_and_schema_1_7_step_time() -> None:
    lhs = _payload_with_sections(
        step_time=_step_time_section(total_step_ms=155.0),
    )
    rhs = _payload_with_sections(
        step_time=_canonical_step_time_section(
            step_time_ms=100.0,
            diagnosis_clock="cpu",
            step_time_cpu_ms=100.0,
            step_time_gpu_ms=None,
        ),
    )
    rhs["schema_version"] = 1.7

    step_time = _build_compare(lhs, rhs)["sections"]["step_time"]

    assert step_time["metrics"]["step_time_ms"] == {
        "label": "CPU Step Time",
        "unit": "ms",
        "lhs": 155.0,
        "rhs": 100.0,
        "delta": -55.0,
        "pct_change": pytest.approx(-35.4838709677),
        "delta_unit": None,
        "direction": "higher_is_worse",
    }


def test_compare_uses_canonical_step_time_for_schema_1_7_artifacts() -> None:
    lhs = _payload_with_sections(
        step_time=_canonical_step_time_section(
            step_time_ms=155.0,
            diagnosis_clock="gpu",
            step_time_cpu_ms=165.0,
            step_time_gpu_ms=155.0,
        ),
    )
    rhs = _payload_with_sections(
        step_time=_canonical_step_time_section(
            step_time_ms=100.0,
            diagnosis_clock="gpu",
            step_time_cpu_ms=110.0,
            step_time_gpu_ms=100.0,
        ),
    )
    lhs["schema_version"] = 1.7
    rhs["schema_version"] = 1.7

    metric = _build_compare(lhs, rhs)["sections"]["step_time"]["metrics"][
        "step_time_ms"
    ]

    assert metric["lhs"] == 155.0
    assert metric["rhs"] == 100.0
    assert metric["delta"] == -55.0
    assert metric["pct_change"] == pytest.approx(-35.4838709677)


def test_compare_prefers_gpu_despite_selected_clock_mismatch() -> None:
    lhs = _payload_with_sections(
        step_time=_canonical_step_time_section(
            step_time_ms=155.0,
            diagnosis_clock="cpu",
            step_time_cpu_ms=155.0,
            step_time_gpu_ms=140.0,
        ),
    )
    rhs = _payload_with_sections(
        step_time=_canonical_step_time_section(
            step_time_ms=100.0,
            diagnosis_clock="gpu",
            step_time_cpu_ms=115.0,
            step_time_gpu_ms=120.0,
        ),
    )
    lhs["schema_version"] = 1.7
    rhs["schema_version"] = 1.7

    step_time = _build_compare(lhs, rhs)["sections"]["step_time"]

    assert step_time["comparison_clock"] == "gpu"
    assert step_time["metrics"]["step_time_ms"]["label"] == "GPU Step Time"
    assert step_time["metrics"]["step_time_ms"]["lhs"] == 140.0
    assert step_time["metrics"]["step_time_ms"]["rhs"] == 120.0
    assert step_time["metrics"]["compute_ms"]["lhs"] is None
    assert step_time["metrics"]["compute_ms"]["rhs"] is None
    assert step_time["notes"] == [
        "Selected-clock phase metrics are unavailable because their clocks "
        "differ or are missing (A: cpu, B: gpu).",
    ]


def test_compare_falls_back_to_common_cpu_clock() -> None:
    lhs = _payload_with_sections(
        step_time=_canonical_step_time_section(
            step_time_ms=155.0,
            diagnosis_clock="gpu",
            step_time_cpu_ms=170.0,
            step_time_gpu_ms=None,
        ),
    )
    rhs = _payload_with_sections(
        step_time=_canonical_step_time_section(
            step_time_ms=100.0,
            diagnosis_clock="cpu",
            step_time_cpu_ms=130.0,
            step_time_gpu_ms=100.0,
        ),
    )
    lhs["schema_version"] = 1.7
    rhs["schema_version"] = 1.7

    compare_payload = _build_compare(lhs, rhs)
    step_time = compare_payload["sections"]["step_time"]

    assert step_time["comparison_clock"] == "cpu"
    assert step_time["metrics"]["step_time_ms"]["label"] == "CPU Step Time"
    assert step_time["metrics"]["step_time_ms"]["lhs"] == 170.0
    assert step_time["metrics"]["step_time_ms"]["rhs"] == 130.0
    assert step_time["metrics"]["input_ms"]["lhs"] is None
    assert "CPU Step Time decreased" in compare_payload["verdict"]["why"]
    text = build_compare_text(compare_payload)
    assert "Step Time (CPU comparison clock)" in text
    assert "CPU Step Time" in text
    assert "Total step" not in text
    assert "iteration" not in text.lower()


def test_compare_legacy_step_time_is_cpu_compatibility_evidence() -> None:
    lhs = _payload_with_sections(
        step_time=_step_time_section(total_step_ms=155.0),
    )
    rhs = _payload_with_sections(
        step_time=_canonical_step_time_section(
            step_time_ms=100.0,
            diagnosis_clock="gpu",
            step_time_cpu_ms=120.0,
            step_time_gpu_ms=100.0,
            dataloader_fetch_cpu_ms=24.0,
        ),
    )
    rhs["schema_version"] = 1.7

    step_time = _build_compare(lhs, rhs)["sections"]["step_time"]

    assert step_time["comparison_clock"] == "cpu"
    assert step_time["metrics"]["step_time_ms"]["lhs"] == 155.0
    assert step_time["metrics"]["step_time_ms"]["rhs"] == 120.0
    assert step_time["metrics"]["dataloader_fetch_cpu_ms"]["lhs"] == 24.0
    assert step_time["metrics"]["dataloader_fetch_cpu_ms"]["rhs"] == 24.0
    assert "total_step_ms" not in json.dumps(step_time)


def test_compare_reports_no_common_step_time_clock() -> None:
    lhs = _payload_with_sections(
        step_time=_canonical_step_time_section(
            step_time_ms=155.0,
            diagnosis_clock="gpu",
            step_time_cpu_ms=None,
            step_time_gpu_ms=155.0,
        ),
    )
    rhs = _payload_with_sections(
        step_time=_canonical_step_time_section(
            step_time_ms=100.0,
            diagnosis_clock="cpu",
            step_time_cpu_ms=100.0,
            step_time_gpu_ms=None,
        ),
    )
    lhs["schema_version"] = 1.7
    rhs["schema_version"] = 1.7

    compare_payload = _build_compare(lhs, rhs)
    step_time = compare_payload["sections"]["step_time"]

    assert step_time["comparison_clock"] is None
    assert step_time["metrics"]["step_time_ms"]["lhs"] is None
    assert step_time["metrics"]["step_time_ms"]["rhs"] is None
    assert step_time["notes"][0] == (
        "Step Time metrics are unavailable because the summaries have no "
        "common measured GPU or CPU clock."
    )
    assert compare_payload["verdict"]["status"] == "INCONCLUSIVE"


def test_compare_accepts_measured_zero_for_common_gpu_clock() -> None:
    lhs = _payload_with_sections(
        step_time=_canonical_step_time_section(
            step_time_ms=0.0,
            diagnosis_clock="gpu",
            step_time_cpu_ms=1.0,
            step_time_gpu_ms=0.0,
        ),
    )
    rhs = _payload_with_sections(
        step_time=_canonical_step_time_section(
            step_time_ms=5.0,
            diagnosis_clock="gpu",
            step_time_cpu_ms=6.0,
            step_time_gpu_ms=5.0,
        ),
    )
    lhs["schema_version"] = 1.7
    rhs["schema_version"] = 1.7

    metric = _build_compare(lhs, rhs)["sections"]["step_time"]["metrics"][
        "step_time_ms"
    ]

    assert metric["lhs"] == 0.0
    assert metric["rhs"] == 5.0
    assert metric["pct_change"] is None
