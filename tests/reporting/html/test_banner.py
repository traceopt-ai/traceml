from traceml_ai.reporting.html import render_html_report
from traceml_ai.reporting.html.banner import render_banner, select_verdict


def test_select_verdict_prefers_primary_diagnosis(
    make_payload, make_section
) -> None:
    payload = make_payload(
        schema_version=1.5,
        primary_diagnosis={
            "kind": "INPUT_BOUND",
            "status": "INPUT-BOUND",
            "severity": "warn",
            "summary": "Input pipeline dominates step time.",
            "action": "Inspect dataloader workers.",
        },
        step_memory=make_section(
            metric_names=["peak_reserved_bytes"],
            average={"peak_reserved_bytes": 2.3e9},
            severity="crit",
            kind="HIGH_PRESSURE",
            status="HIGH PRESSURE",
            summary="Section diagnosis should not win.",
        ),
    )

    source, diag = select_verdict(payload)

    assert source == "primary"
    assert diag["kind"] == "INPUT_BOUND"


def test_render_banner_uses_primary_diagnosis_for_schema_15(
    make_payload, make_section
) -> None:
    payload = make_payload(
        schema_version=1.5,
        primary_diagnosis={
            "kind": "INPUT_BOUND",
            "status": "INPUT-BOUND",
            "severity": "warn",
            "summary": "Input pipeline dominates step time.",
            "action": "Inspect dataloader workers.",
        },
        step_memory=make_section(
            metric_names=["peak_reserved_bytes"],
            average={"peak_reserved_bytes": 2.3e9},
            severity="crit",
            kind="HIGH_PRESSURE",
            status="HIGH PRESSURE",
            summary="Section diagnosis should not render in the banner.",
        ),
    )

    banner = render_banner(payload)

    assert 'class="banner warn"' in banner
    assert "INPUT-BOUND" in banner
    assert "Input pipeline dominates step time." in banner
    assert "Inspect dataloader workers." in banner
    assert "Primary diagnosis" in banner
    assert "Section diagnosis should not render in the banner." not in banner


def test_select_verdict_ignores_empty_primary_diagnosis(
    make_payload, make_section
) -> None:
    payload = make_payload(
        schema_version=1.5,
        primary_diagnosis={},
        step_memory=make_section(
            metric_names=["peak_reserved_bytes"],
            average={"peak_reserved_bytes": 2.3e9},
            severity="crit",
            kind="HIGH_PRESSURE",
            status="HIGH PRESSURE",
        ),
    )

    section, diag = select_verdict(payload)

    assert section == "step_memory"
    assert diag["kind"] == "HIGH_PRESSURE"


def test_select_verdict_ignores_non_dict_primary_diagnosis(
    make_payload, make_section
) -> None:
    payload = make_payload(
        schema_version=1.5,
        primary_diagnosis="INPUT_BOUND",
        step_memory=make_section(
            metric_names=["peak_reserved_bytes"],
            average={"peak_reserved_bytes": 2.3e9},
            severity="crit",
            kind="HIGH_PRESSURE",
            status="HIGH PRESSURE",
        ),
    )

    section, diag = select_verdict(payload)

    assert section == "step_memory"
    assert diag["kind"] == "HIGH_PRESSURE"


def test_render_banner_escapes_primary_diagnosis_fields(make_payload) -> None:
    payload = make_payload(
        schema_version=1.5,
        primary_diagnosis={
            "kind": "INPUT_BOUND",
            "status": "<b>INPUT</b>",
            "severity": "crit",
            "summary": "<script>alert(1)</script>",
            "action": "Use workers & pin memory.",
        },
    )

    banner = render_banner(payload)

    assert "<script>" not in banner
    assert "<b>INPUT</b>" not in banner
    assert "&lt;b&gt;INPUT&lt;/b&gt;" in banner
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in banner
    assert "Use workers &amp; pin memory." in banner


def test_select_verdict_picks_crit_over_warn(
    make_payload, make_section
) -> None:
    payload = make_payload(
        step_time=make_section(
            metric_names=["total_step_ms"],
            average={"total_step_ms": 100.0},
            severity="warn",
            kind="INPUT_BOUND",
            status="INPUT-BOUND",
        ),
        step_memory=make_section(
            metric_names=["peak_reserved_bytes"],
            average={"peak_reserved_bytes": 2.3e9},
            severity="crit",
            kind="HIGH_PRESSURE",
            status="HIGH PRESSURE",
        ),
    )
    section, diag = select_verdict(payload)
    assert section == "step_memory"
    assert diag["severity"] == "crit"


def test_select_verdict_tie_break_prefers_step_time(
    make_payload, make_section
) -> None:
    crit = dict(severity="crit", kind="X", status="X")
    payload = make_payload(
        step_time=make_section(
            metric_names=["total_step_ms"],
            average={"total_step_ms": 1.0},
            **crit,
        ),
        step_memory=make_section(
            metric_names=["peak_reserved_bytes"],
            average={"peak_reserved_bytes": 1.0},
            **crit,
        ),
    )
    section, _ = select_verdict(payload)
    assert section == "step_time"


def test_render_banner_crit_uses_crit_class(
    make_payload, make_section
) -> None:
    payload = make_payload(
        step_memory=make_section(
            metric_names=["peak_reserved_bytes"],
            average={"peak_reserved_bytes": 2.3e9},
            severity="crit",
            kind="HIGH_PRESSURE",
            status="HIGH PRESSURE",
            summary="Peak is 97% of capacity.",
            action="Reduce batch size.",
        ),
    )
    banner = render_banner(payload)
    assert 'class="banner crit"' in banner
    assert "Peak is 97% of capacity." in banner
    assert "Reduce batch size." in banner


def test_render_banner_healthy_info_kind_uses_good_class(make_payload) -> None:
    # Default payload is all-info with BALANCED/NORMAL kinds.
    assert 'class="banner good"' in render_banner(make_payload())


def test_render_banner_no_data_kind_uses_neutral_class(
    make_payload, make_section
) -> None:
    payload = make_payload(
        step_time=make_section(
            metric_names=["total_step_ms"],
            average={"total_step_ms": 0.0},
            severity="info",
            kind="NO_DATA",
            status="NO DATA",
        ),
        step_memory=make_section(
            metric_names=["peak_reserved_bytes"],
            average={"peak_reserved_bytes": 0.0},
            severity="info",
            kind="NO_DATA",
            status="NO DATA",
        ),
        system=make_section(
            metric_names=["gpu_util_percent"],
            average={"gpu_util_percent": 0.0},
            severity="info",
            kind="NO_DATA",
            status="NO DATA",
            by="node_rank",
        ),
        process=make_section(
            metric_names=["cpu_percent"],
            average={"cpu_percent": 0.0},
            severity="info",
            kind="NO_DATA",
            status="NO DATA",
        ),
    )
    assert 'class="banner neutral"' in render_banner(payload)


def test_render_banner_omits_action_when_absent(
    make_payload, make_section
) -> None:
    payload = make_payload(
        step_memory=make_section(
            metric_names=["peak_reserved_bytes"],
            average={"peak_reserved_bytes": 2.3e9},
            severity="crit",
            kind="HIGH_PRESSURE",
            status="HIGH PRESSURE",
            summary="Peak high.",
            action=None,
        ),
    )
    assert "Action:" not in render_banner(payload)


def test_render_banner_falls_back_to_status_when_summary_missing(
    make_payload, make_section
) -> None:
    payload = make_payload(
        step_memory=make_section(
            metric_names=["peak_reserved_bytes"],
            average={"peak_reserved_bytes": 2.3e9},
            severity="crit",
            kind="HIGH_PRESSURE",
            status="HIGH PRESSURE",
            summary=None,
        ),
    )
    banner = render_banner(payload)
    assert "HIGH PRESSURE" in banner
    assert "&mdash; &mdash;" not in banner  # never a bare placeholder headline


def test_render_html_report_includes_the_banner(make_payload) -> None:
    assert '<div class="banner' in render_html_report(make_payload())
