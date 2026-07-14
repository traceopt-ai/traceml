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


def _metric(name: str, value: float) -> StepCombinedTimeMetric:
    return StepCombinedTimeMetric(
        metric=name,
        clock="gpu",
        series=None,
        summary=StepCombinedTimeSummary(
            window_size=5,
            steps_used=5,
            median_total=value,
            worst_total=value,
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


def test_step_time_dashboard_hero_uses_diagnosis_metrics() -> None:
    assert theme.PHASES[0][:2] == ("IW", "input_wait")

    diagnosis_metrics = [
        _metric("input_wait", 10.0),
        _metric("h2d", 10.0),
        _metric("forward", 20.0),
        _metric("backward", 30.0),
        _metric("optimizer_step", 20.0),
        _metric("residual_proxy", 10.0),
        _metric("step_time", 100.0),
    ]
    payload = StepCombinedTimeResult(
        diagnosis_metrics=diagnosis_metrics,
        diagnosis_clock="gpu",
    )
    panel = _panel()

    update_model_combined_section(panel, payload)

    assert panel["seg_labs"][0].text == "IW"
    assert panel["seg_divs"][0].styles[-1] == "width:10.000%"
    assert panel["win"].text == "5 aligned steps"
    assert panel["kpis"]["median"].content.startswith("100")
    assert panel["kpis"]["worst"].content.startswith("100")
    assert not panel["kpis"]["median"].content.startswith("20")
