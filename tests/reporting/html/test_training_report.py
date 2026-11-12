"""Compact diagnosis card rendered into tracker runs (W&B, MLflow)."""

from html.parser import HTMLParser

from traceml_ai import __version__
from traceml_ai.reporting.html import render_training_report
from traceml_ai.reporting.html.style import PALETTE
from traceml_ai.reporting.html.textutils import fmt_ms_compact
from traceml_ai.reporting.sections.step_time.builder import (
    STEP_TIME_METRIC_NAMES,
)

_HOST = "gpu-host-secret-17"

# Averages of a real 60-step CPU run whose input wait is 97% of Step Time.
_INPUT_BOUND_AVERAGE = {
    "input_wait_ms": 153.958,
    "step_time_ms": 158.793,
    "traced_step_time_ms": 4.835,
    "step_time_cpu_ms": 158.793,
    "traced_step_time_cpu_ms": 4.835,
    "dataloader_fetch_cpu_ms": 153.958,
    "compute_ms": 4.163,
    "residual_ms": 0.671,
    "forward_ms": 0.849,
    "backward_ms": 1.681,
    "optimizer_ms": 1.633,
}

_INPUT_BOUND = {
    "kind": "INPUT_BOUND",
    "status": "INPUT-BOUND",
    "severity": "crit",
    "section": "step_time",
    "scope": "performance",
    "summary": "Input wait is 97.0% of the typical CPU Step Time.",
    "action": "Increase workers, prefetch, or storage throughput.",
    "evidence": {"type": "phase_share", "steps_analyzed": 60},
}

_INSUFFICIENT = {
    "kind": "INSUFFICIENT_STEP_TIME_DATA",
    "status": "INSUFFICIENT STEP-TIME DATA",
    "severity": "info",
    "section": "step_time",
    "scope": "performance",
    "summary": (
        "Not enough completed step-time samples were available for a "
        "stable performance diagnosis."
    ),
    "action": "Run for more steps or ensure step timing is recorded.",
    "evidence": {"type": "insufficient_data"},
}


def _step_time(make_section, average):
    values = {name: None for name in STEP_TIME_METRIC_NAMES}
    values.update(average)
    section = make_section(
        metric_names=STEP_TIME_METRIC_NAMES,
        average=values,
        kind="INPUT_BOUND",
        status="INPUT-BOUND",
        severity="crit",
        rows={
            "0": {
                "identity": {"global_rank": 0, "hostname": _HOST},
                "metrics": dict(values),
            }
        },
    )
    section["metadata"].update(
        {"global_ranks_seen": 1, "global_ranks_used": 1}
    )
    section["global"]["window"] = {
        "kind": "step_window",
        "alignment": "common_steps",
        "steps_analyzed": 60,
        "start_step": 1,
        "end_step": 60,
        "completed_step": 60,
        "diagnosis_clock": "cpu",
    }
    return section


def _payload(
    make_payload,
    make_section,
    *,
    primary=_INPUT_BOUND,
    average=None,
    run_name="mock-slow",
):
    payload = make_payload(
        schema_version=2.0,
        meta={
            "run_name": run_name,
            "mode": "single_node",
            "world_size": 1,
            "nodes_observed": 1,
            "gpus_observed": 0,
        },
        primary_diagnosis=primary,
        step_time=_step_time(
            make_section,
            _INPUT_BOUND_AVERAGE if average is None else average,
        ),
    )
    payload["_traceml_internal"] = {"session": "internal-only"}
    return payload


_VOID = frozenset(
    "area base br col embed hr img input link meta source track wbr".split()
)


class _TagBalance(HTMLParser):
    """Track open non-void elements; each end tag must close the innermost."""

    def __init__(self) -> None:
        super().__init__()
        self.stack = []

    def handle_starttag(self, tag, attrs) -> None:
        if tag not in _VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs) -> None:
        # A self-closed element (``<rect .../>``) opens and closes itself.
        pass

    def handle_endtag(self, tag) -> None:
        assert self.stack, f"</{tag}> closes nothing"
        innermost = self.stack.pop()
        assert innermost == tag, f"</{tag}> closes <{innermost}>"


def _parses(text: str) -> bool:
    parser = _TagBalance()
    parser.feed(text)
    parser.close()
    assert not parser.stack, f"unclosed tags: {parser.stack}"
    return True


def test_card_is_a_self_contained_document(make_payload, make_section) -> None:
    out = render_training_report(_payload(make_payload, make_section))
    assert out.startswith("<!DOCTYPE html>")
    assert '<meta charset="utf-8">' in out
    assert "<title>TraceML diagnosis</title>" in out
    assert _parses(out)


def test_card_has_no_scripts_links_or_host_details(
    make_payload, make_section
) -> None:
    out = render_training_report(_payload(make_payload, make_section))
    lowered = out.lower()
    assert "INPUT-BOUND" in out
    assert "<script" not in lowered
    assert "href=" not in lowered
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert "hostname" not in lowered
    assert _HOST not in out


def test_card_never_dumps_raw_json_or_internal_keys(
    make_payload, make_section
) -> None:
    out = render_training_report(_payload(make_payload, make_section))
    assert "mock-slow" in out
    assert "_traceml_internal" not in out
    assert "internal-only" not in out
    assert "phase_share" not in out
    assert "<pre" not in out


def test_card_renders_deterministically(make_payload, make_section) -> None:
    payload = _payload(make_payload, make_section)
    first = render_training_report(payload)
    assert first.startswith("<!DOCTYPE html>")
    assert first == render_training_report(payload)


def test_input_bound_card_shows_the_decided_verdict(
    make_payload, make_section
) -> None:
    out = render_training_report(_payload(make_payload, make_section))
    assert 'class="chip crit">INPUT-BOUND</span>' in out
    assert "<h1>Input wait is 97.0% of the typical CPU Step Time.</h1>" in out
    assert (
        "<b>Next</b> Increase workers, prefetch, or storage throughput." in out
    )


def test_input_bound_card_shows_coverage_line(
    make_payload, make_section
) -> None:
    out = render_training_report(_payload(make_payload, make_section))
    assert (
        '<p class="cov">mock-slow &middot; 1 rank &middot; 60 common steps '
        "&middot; steps 1–60 &middot; CPU clock</p>"
    ) in out


def test_input_bound_card_shows_phase_bar_and_three_rows(
    make_payload, make_section
) -> None:
    out = render_training_report(_payload(make_payload, make_section))
    assert "<svg" in out
    assert "input wait 154.0&thinsp;ms" in out
    assert "<tr><td>Step time</td><td>158.8&thinsp;ms</td></tr>" in out
    assert "<tr><td>Input wait</td><td>154.0&thinsp;ms</td></tr>" in out
    assert "<tr><td>Compute</td><td>4.2&thinsp;ms</td></tr>" in out


def test_card_reuses_the_full_report_palette(
    make_payload, make_section
) -> None:
    out = render_training_report(_payload(make_payload, make_section))
    assert PALETTE in out
    assert "--crit:#c4392f" in out
    assert "prefers-color-scheme: dark" in out
    # The chip colours are the banner's tokens, not card-local literals.
    assert ".chip.info{background:var(--info);}" in out
    assert ".chip.neutral{background:var(--neutral);}" in out


def test_insufficient_data_card_uses_info_tier_and_claims_no_share(
    make_payload, make_section
) -> None:
    average = {"step_time_ms": None, "input_wait_ms": None}
    out = render_training_report(
        _payload(
            make_payload, make_section, primary=_INSUFFICIENT, average=average
        )
    )
    assert 'class="chip info">INSUFFICIENT STEP-TIME DATA</span>' in out
    assert "Not enough completed step-time samples" in out
    assert "% of" not in out
    assert "<svg" not in out


def test_card_without_primary_diagnosis_shows_neutral_chip_only(
    make_payload,
) -> None:
    payload = make_payload()
    assert "primary_diagnosis" not in payload
    out = render_training_report(payload)
    assert 'class="chip neutral">NO PRIMARY DIAGNOSIS</span>' in out
    assert "<h1>" not in out
    assert "<b>Next</b>" not in out
    # Section diagnoses are never promoted into a verdict sentence.
    assert "Balanced step." not in out
    assert "Balanced memory." not in out


def test_unmeasured_step_time_is_stated_once_in_the_table(
    make_payload, make_section
) -> None:
    average = dict(_INPUT_BOUND_AVERAGE, step_time_ms=None)
    out = render_training_report(
        _payload(make_payload, make_section, average=average)
    )
    assert "<svg" not in out
    assert "Step time: Not measured" not in out
    assert "<tr><td>Step time</td><td>Not measured</td></tr>" in out
    assert out.count("Step time") == 1


def test_measured_zero_step_time_renders_zero_not_not_measured(
    make_payload, make_section
) -> None:
    average = dict(_INPUT_BOUND_AVERAGE, step_time_ms=0.0)
    out = render_training_report(
        _payload(make_payload, make_section, average=average)
    )
    assert "<tr><td>Step time</td><td>0.0&thinsp;ms</td></tr>" in out
    assert "<tr><td>Step time</td><td>Not measured</td></tr>" not in out


def test_zero_steps_and_zero_ranks_are_stated_in_coverage(
    make_payload, make_section
) -> None:
    payload = _payload(make_payload, make_section)
    step_time = payload["step_time"]
    step_time["metadata"].update(
        {"global_ranks_seen": 0, "global_ranks_used": 0}
    )
    step_time["groups"]["rows"] = {}
    step_time["global"]["window"] = {
        "kind": "step_window",
        "alignment": "common_steps",
        "steps_analyzed": 0,
        "diagnosis_clock": "cpu",
    }
    payload["process"]["groups"]["rows"] = {}
    out = render_training_report(payload)
    assert (
        '<p class="cov">mock-slow &middot; 0/1 rank &middot; 0 common steps '
        "&middot; CPU clock</p>"
    ) in out


def test_card_escapes_payload_strings(make_payload, make_section) -> None:
    primary = dict(_INPUT_BOUND, summary="Input <b>dominates</b> the step.")
    out = render_training_report(
        _payload(make_payload, make_section, primary=primary, run_name="a & b")
    )
    assert "<b>dominates</b>" not in out
    assert "Input &lt;b&gt;dominates&lt;/b&gt; the step." in out
    assert "a &amp; b" in out


def test_card_footer_states_version_schema_and_not_measured_rule(
    make_payload, make_section
) -> None:
    out = render_training_report(_payload(make_payload, make_section))
    assert f"Rendered by TraceML {__version__} &middot; schema 2.0" in out
    assert "Not measured means no signal, not zero" in out


def test_card_stays_compact(make_payload, make_section) -> None:
    out = render_training_report(_payload(make_payload, make_section))
    assert "<svg" in out
    assert len(out.encode("utf-8")) < 12_000


def test_fmt_ms_compact_marks_missing_values_not_measured() -> None:
    for value in (None, float("nan"), float("inf"), True, "x"):
        assert fmt_ms_compact(value) == "Not measured"


def test_fmt_ms_compact_formats_measured_values() -> None:
    assert fmt_ms_compact(0.05) == "&lt;0.1&thinsp;ms"
    assert fmt_ms_compact(0.0) == "0.0&thinsp;ms"
    assert fmt_ms_compact(1234.56) == "1,234.6&thinsp;ms"
