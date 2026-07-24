from __future__ import annotations

from rich.console import Console

from traceml_ai.diagnostics.step_time.api import StepDiagnosis
from traceml_ai.renderers.step_time import renderer as renderer_module
from traceml_ai.renderers.step_time.renderer import StepCombinedRenderer
from traceml_ai.renderers.step_time.schema import (
    StepCombinedTimeCoverage,
    StepCombinedTimeMetric,
    StepCombinedTimeResult,
    StepCombinedTimeSummary,
)


def _metric(
    name: str,
    value: float,
    *,
    clock: str = "cpu",
) -> StepCombinedTimeMetric:
    return StepCombinedTimeMetric(
        metric=name,
        clock=clock,
        series=None,
        summary=StepCombinedTimeSummary(
            window_size=1,
            steps_used=1,
            median_total=value,
            worst_total=value,
            worst_rank=0,
            skew_ratio=0.0,
            skew_pct=0.0,
        ),
        coverage=StepCombinedTimeCoverage(
            expected_steps=1,
            steps_used=1,
            completed_step=1,
            world_size=1,
            ranks_present=1,
            incomplete=False,
        ),
    )


def _render_text(renderable) -> str:
    console = Console(record=True, width=140, color_system=None)
    console.print(renderable)
    return console.export_text()


def test_step_time_cli_diagnosis_uses_selected_metrics(monkeypatch) -> None:
    diagnosis_metrics = [
        _metric("input_wait", 40.0),
        _metric("step_time", 100.0),
        _metric("residual_proxy", 0.0),
    ]
    payload = StepCombinedTimeResult(
        per_rank_timing={0: {"input_wait": 40.0, "step_time": 100.0}},
        diagnosis_clock="gpu",
        training_strategy="fsdp",
        diagnosis_metrics=diagnosis_metrics,
    )
    seen = {}

    def fake_build_step_diagnosis(metrics, **kwargs):
        seen["metrics"] = metrics
        seen["diagnosis_clock"] = kwargs.get("diagnosis_clock")
        seen["training_strategy"] = kwargs.get("training_strategy")
        return StepDiagnosis(
            kind="INPUT_BOUND",
            status="INPUT-BOUND",
            severity="warn",
            reason="Input wait is high.",
            action="Increase workers.",
            steps_used=1,
        )

    monkeypatch.setattr(
        renderer_module,
        "build_step_diagnosis",
        fake_build_step_diagnosis,
    )

    renderer = StepCombinedRenderer(db_path=":memory:")
    monkeypatch.setattr(renderer, "_payload", lambda: payload)
    text = _render_text(renderer.get_panel_renderable())

    assert seen["metrics"] is diagnosis_metrics
    assert seen["diagnosis_clock"] == "gpu"
    assert seen["training_strategy"] == "fsdp"
    assert "IW" in text
    assert "DL" not in text
    assert "Average (1 steps)" in text
    assert "Sum (" not in text
    assert "40.0 ms" in text
    assert "12.0 ms" not in text


def test_step_time_cli_renders_zero_timings_as_zero(monkeypatch) -> None:
    diagnosis_metrics = [
        _metric("input_wait", 0.0),
        _metric("h2d", 0.0),
        _metric("forward", 4.5),
        _metric("backward", 8.7),
        _metric("optimizer_step", 12.0),
        _metric("step_time", 31.2),
        _metric("residual_proxy", 6.0),
    ]
    payload = StepCombinedTimeResult(
        per_rank_timing={0: {"input_wait": 0.0, "step_time": 31.2}},
        diagnosis_clock="cpu",
        diagnosis_metrics=diagnosis_metrics,
    )

    def fake_build_step_diagnosis(metrics, **kwargs):
        return StepDiagnosis(
            kind="RESIDUAL_HEAVY",
            status="RESIDUAL-HEAVY",
            severity="warn",
            reason="Residual time is high.",
            action="Inspect work outside traced phases.",
            steps_used=1,
        )

    monkeypatch.setattr(
        renderer_module,
        "build_step_diagnosis",
        fake_build_step_diagnosis,
    )

    renderer = StepCombinedRenderer(db_path=":memory:")
    monkeypatch.setattr(renderer, "_payload", lambda: payload)
    text = _render_text(renderer.get_panel_renderable())

    assert "IW" in text
    assert "H2D" in text
    assert text.count("0.0 ms") >= 2
    assert "4.5 ms" in text
