from __future__ import annotations

import pytest

pytest.importorskip("nicegui")

from traceml_ai.aggregator.display_drivers.nicegui_sections import theme
from traceml_ai.aggregator.display_drivers.nicegui_sections.model_combined_section import (
    update_model_combined_section,
)
from traceml_ai.renderers.step_time.schema import (
    StepCombinedTimeCoverage,
    StepCombinedTimeMetric,
    StepCombinedTimeResult,
    StepCombinedTimeSummary,
)


class _FakeSegment:
    def __init__(self) -> None:
        self.styles: list[str] = []

    def style(self, value: str) -> "_FakeSegment":
        self.styles.append(value)
        return self


class _FakeText:
    def __init__(self) -> None:
        self.text = ""


class _FakeHtml:
    def __init__(self) -> None:
        self.content = ""


def _metric(
    name: str,
    value: float,
    worst: float | None = None,
) -> StepCombinedTimeMetric:
    return StepCombinedTimeMetric(
        metric=name,
        clock="gpu",
        series=None,
        summary=StepCombinedTimeSummary(
            window_size=5,
            steps_used=5,
            median_total=value,
            worst_total=worst if worst is not None else value,
            worst_rank=0,
            skew_ratio=0.0,
            skew_pct=0.0,
        ),
        coverage=StepCombinedTimeCoverage(
            expected_steps=5,
            steps_used=5,
            completed_step=5,
            world_size=1,
            ranks_present=1,
            incomplete=False,
        ),
    )


def _panel() -> dict:
    return {
        "seg_divs": [_FakeSegment() for _ in theme.PHASES],
        "seg_labs": [_FakeText() for _ in theme.PHASES],
        "win": _FakeText(),
        "verdict": _FakeText(),
        "kpis": {
            "median": _FakeHtml(),
            "worst": _FakeHtml(),
            "gap": _FakeHtml(),
            "residual": _FakeHtml(),
            "rank": _FakeHtml(),
        },
        "_last_sig": None,
    }


def test_step_time_dashboard_hero_renders_sparse_metrics() -> None:
    # h2d and residual_proxy were never measured: the hero must still
    # update from the fresh sparse window instead of freezing on the last
    # complete view (issue #259).
    diagnosis_metrics = [
        _metric("input_wait", 10.0),
        _metric("forward", 20.0),
        _metric("backward", 30.0),
        _metric("optimizer_step", 20.0),
        _metric("step_time", 100.0),
    ]
    payload = StepCombinedTimeResult(
        diagnosis_metrics=diagnosis_metrics,
        diagnosis_clock="cpu",
    )
    panel = _panel()

    update_model_combined_section(panel, payload)

    # The panel updated (no freeze): window label reflects the sparse
    # window and flags the partial coverage.
    assert panel["win"].text == "5 aligned steps · partial signals"
    # Missing phases render as empty segments; measured phases scale
    # against the iteration envelope (input_wait 10 + step 100 = 110),
    # so unmeasured time stays visible as empty ribbon space.
    keys = [key for _, key, _ in theme.PHASES]
    by_key = dict(zip(keys, panel["seg_divs"]))
    assert by_key["h2d"].styles[-1] == "width:0.000%"
    assert by_key["residual_proxy"].styles[-1] == "width:0.000%"
    assert by_key["input_wait"].styles[-1] == "width:9.091%"
    # An underivable residual shows n/a, never a fake 0%.
    assert "n/a" in panel["kpis"]["residual"].content
    assert panel["kpis"]["median"].content.startswith("100")


def test_step_time_dashboard_hero_measured_zero_is_not_partial() -> None:
    diagnosis_metrics = [
        _metric("input_wait", 10.0),
        _metric("h2d", 0.0),
        _metric("forward", 20.0),
        _metric("backward", 30.0),
        _metric("optimizer_step", 20.0),
        _metric("residual_proxy", 30.0),
        _metric("step_time", 100.0),
    ]
    payload = StepCombinedTimeResult(
        diagnosis_metrics=diagnosis_metrics,
        diagnosis_clock="cpu",
    )
    panel = _panel()

    update_model_combined_section(panel, payload)

    # A measured-zero H2D is complete coverage, not partial signals.
    assert panel["win"].text == "5 aligned steps"
    assert "n/a" not in panel["kpis"]["residual"].content


def test_step_time_dashboard_hero_absent_h2d_is_not_partial() -> None:
    # H2D events are occurrence-driven: a run with no host-to-device
    # copies emits none, which is complete coverage, not partial.
    diagnosis_metrics = [
        _metric("input_wait", 10.0),
        _metric("forward", 20.0),
        _metric("backward", 30.0),
        _metric("optimizer_step", 20.0),
        _metric("residual_proxy", 30.0),
        _metric("step_time", 100.0),
    ]
    payload = StepCombinedTimeResult(
        diagnosis_metrics=diagnosis_metrics,
        diagnosis_clock="cpu",
    )
    panel = _panel()

    update_model_combined_section(panel, payload)

    assert panel["win"].text == "5 aligned steps"
    keys = [key for _, key, _ in theme.PHASES]
    by_key = dict(zip(keys, panel["seg_divs"]))
    assert by_key["h2d"].styles[-1] == "width:0.000%"


def test_step_time_dashboard_hero_step_time_only_extreme() -> None:
    payload = StepCombinedTimeResult(
        diagnosis_metrics=[_metric("step_time", 100.0)],
        diagnosis_clock="cpu",
    )
    panel = _panel()

    update_model_combined_section(panel, payload)

    assert panel["win"].text == "5 aligned steps · partial signals"
    assert all(seg.styles[-1] == "width:0.000%" for seg in panel["seg_divs"])
    assert "n/a" in panel["kpis"]["residual"].content
    assert panel["kpis"]["median"].content.startswith("100")


def test_step_time_dashboard_hero_ignores_payload_without_step_time() -> None:
    payload = StepCombinedTimeResult(
        diagnosis_metrics=[
            _metric("input_wait", 10.0),
            _metric("forward", 20.0),
        ],
        diagnosis_clock="cpu",
    )
    panel = _panel()

    update_model_combined_section(panel, payload)

    # Without the step envelope there is nothing coherent to draw: the
    # panel stays untouched instead of raising into the UI tick.
    assert panel["win"].text == ""


def test_step_time_dashboard_hero_uses_diagnosis_metrics() -> None:
    assert theme.PHASES[0][:2] == ("IW", "input_wait")

    # Self-consistent window shape: phases sum to input_wait + step.
    diagnosis_metrics = [
        _metric("input_wait", 10.0),
        _metric("h2d", 10.0),
        _metric("forward", 20.0),
        _metric("backward", 30.0),
        _metric("optimizer_step", 20.0),
        _metric("residual_proxy", 20.0),
        _metric("step_time", 100.0, worst=200.0),
    ]
    payload = StepCombinedTimeResult(
        diagnosis_metrics=diagnosis_metrics,
        diagnosis_clock="gpu",
    )
    panel = _panel()

    update_model_combined_section(panel, payload)

    assert panel["seg_labs"][0].text == "IW"
    # input_wait 10 of the 110 ms iteration envelope.
    assert panel["seg_divs"][0].styles[-1] == "width:9.091%"
    assert panel["win"].text == "5 aligned steps"
    assert panel["kpis"]["median"].content.startswith("100")
    assert panel["kpis"]["worst"].content.startswith("200")
    assert not panel["kpis"]["median"].content.startswith("20")


def test_diagnostics_rail_incomplete_data_is_neutral() -> None:
    from traceml_ai.aggregator.display_drivers.nicegui_sections import (
        model_diagnostics_section as mds,
    )

    assert (
        mds._row_sev({"severity": "info", "kind": "INCOMPLETE_DATA"})
        == "neutral"
    )
    assert mds._row_sev({"severity": "info", "kind": "BALANCED"}) == "healthy"
