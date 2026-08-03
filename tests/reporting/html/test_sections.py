from traceml_ai.reporting.html import render_html_report
from traceml_ai.reporting.html.sections import render_sections


def test_all_four_section_titles_present(make_payload) -> None:
    out = render_sections(make_payload())
    for title in ("Step time", "Step memory", "System", "Process"):
        assert title in out


def test_units_resolved_by_metric_name_suffix(make_payload) -> None:
    out = render_sections(make_payload())
    assert "ms" in out  # *_ms metrics
    assert "GB" in out  # *_bytes -> decimal GB
    assert "%" in out  # *_percent


def test_per_rank_rows_iterated_from_object(make_payload) -> None:
    # step_time fixture has two ranks keyed "0" and "1" in a JSON object.
    out = render_sections(make_payload())
    assert "Per-rank window stats (2)" in out


def test_diagnosis_card_uses_severity_class(
    make_payload, make_section
) -> None:
    payload = make_payload(
        step_memory=make_section(
            metric_names=["peak_reserved_bytes"],
            average={"peak_reserved_bytes": 2.3e9},
            severity="crit",
            kind="HIGH_PRESSURE",
            status="HIGH PRESSURE",
            units={"memory": "bytes"},
        ),
    )
    assert 'class="diag crit"' in render_sections(payload)


def test_extra_issues_are_listed(make_payload, make_section) -> None:
    primary = {
        "kind": "INPUT_BOUND",
        "status": "INPUT-BOUND",
        "severity": "warn",
        "summary": "Dataloader heavy.",
        "action": None,
    }
    secondary = {
        "kind": "COMPUTE_STRAGGLER",
        "status": "COMPUTE STRAGGLER",
        "severity": "crit",
        "summary": "Rank 2 slow compute.",
        "action": None,
    }
    section = make_section(
        metric_names=["total_step_ms"],
        average={"total_step_ms": 100.0},
        severity="warn",
        kind="INPUT_BOUND",
        status="INPUT-BOUND",
        issues=[primary, secondary],
    )
    out = render_sections(make_payload(step_time=section))
    assert "Dataloader heavy." in out
    assert "Rank 2 slow compute." in out


def test_friendly_metric_labels_and_raw_fallback(
    make_payload, make_section
) -> None:
    section = make_section(
        metric_names=["total_step_ms", "weird_custom_metric"],
        average={"total_step_ms": 100.0, "weird_custom_metric": 1.0},
    )
    out = render_sections(make_payload(step_time=section))
    assert "Total step" in out  # mapped label
    assert "weird_custom_metric" in out  # unknown key rendered raw


def test_missing_section_does_not_crash(make_payload) -> None:
    out = render_sections(make_payload(system={}))
    assert "System" in out


def test_render_html_report_includes_sections(make_payload) -> None:
    assert "Step time" in render_html_report(make_payload())
