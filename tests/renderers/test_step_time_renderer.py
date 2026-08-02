from __future__ import annotations

from unittest.mock import Mock

from rich.console import Console

from tests.step_time.factories import (
    live_result_from_window,
    window_from_rank_averages,
)
from traceml_ai.renderers.step_time.renderer import StepCombinedRenderer
from traceml_ai.step_time.model import StepTimeMetric


def _metric(
    name: str,
    value: float,
) -> StepTimeMetric:
    return StepTimeMetric(
        metric=name,
        series=None,
        window_size=1,
        steps_used=1,
        median_total=value,
        worst_total=value,
        worst_rank=0,
        skew_ratio=0.0,
        skew_pct=0.0,
        measured_ranks=(0,),
    )


def _render_text(renderable) -> str:
    console = Console(record=True, width=140, color_system=None)
    console.print(renderable)
    return console.export_text()


def test_step_time_cli_uses_the_precomputed_selected_clock_analysis() -> None:
    diagnosis_metrics = [
        _metric("input_wait", 40.0),
        _metric("step_time", 100.0),
        _metric("residual_proxy", 0.0),
    ]
    payload = live_result_from_window(
        window_from_rank_averages(
            {0: {"input_wait": 40.0, "step_time": 100.0}},
            clock="gpu",
            expected_ranks=(0,),
            metrics=diagnosis_metrics,
            training_strategy="fsdp",
        ),
    )

    renderer = StepCombinedRenderer(session=Mock())
    text = _render_text(renderer.render(payload))

    assert payload.analysis.window.metrics is diagnosis_metrics
    assert payload.analysis.window.clock == "gpu"
    assert payload.analysis.window.training_strategy == "fsdp"
    assert "WARMUP" in text
    assert "IW" in text
    assert "DL" not in text
    assert "Average (1 steps)" in text
    assert "Sum (" not in text
    assert "40.0 ms" in text
    assert "12.0 ms" not in text


def test_step_time_cli_renders_zero_timings_as_zero() -> None:
    diagnosis_metrics = [
        _metric("input_wait", 0.0),
        _metric("h2d", 0.0),
        _metric("forward", 4.5),
        _metric("backward", 8.7),
        _metric("optimizer_step", 12.0),
        _metric("step_time", 31.2),
        _metric("residual_proxy", 6.0),
    ]
    payload = live_result_from_window(
        window_from_rank_averages(
            {
                0: {
                    "input_wait": 0.0,
                    "h2d": 0.0,
                    "forward": 4.5,
                    "backward": 8.7,
                    "optimizer_step": 12.0,
                    "step_time": 31.2,
                    "residual_proxy": 6.0,
                }
            },
            expected_ranks=(0,),
            metrics=diagnosis_metrics,
        ),
    )

    renderer = StepCombinedRenderer(session=Mock())
    text = _render_text(renderer.render(payload))

    assert "IW" in text
    assert "H2D" in text
    assert text.count("0.0 ms") >= 2
    assert "4.5 ms" in text


def test_step_time_cli_refreshes_its_injected_session_once() -> None:
    payload = live_result_from_window(
        window_from_rank_averages(
            {0: {"step_time": 10.0}},
            expected_ranks=(0,),
            metrics=[_metric("step_time", 10.0)],
        )
    )
    session = Mock()
    session.refresh.return_value = payload

    renderer = StepCombinedRenderer(session=session)
    text = _render_text(renderer.get_panel_renderable())

    session.refresh.assert_called_once_with()
    assert "STEP" in text
