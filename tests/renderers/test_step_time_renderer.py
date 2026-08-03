from __future__ import annotations

from unittest.mock import Mock

from rich.console import Console

from tests.step_time.factories import (
    live_result_from_window,
    window_from_rank_averages,
)
from traceml_ai.renderers.step_time.renderer import (
    StepTimeRenderer,
    _table_metrics,
)
from traceml_ai.step_time.model import StepTimeMetric, StepTimeWindow
from traceml_ai.step_time.pipeline import LiveStepTimeResult


def _metric(
    name: str,
    value: float,
) -> StepTimeMetric:
    return StepTimeMetric(
        metric=name,
        series=None,
        median_total=value,
        worst_total=value,
        worst_rank=0,
        skew_pct=0.0,
    )


def _render_text(renderable) -> str:
    console = Console(record=True, width=140, color_system=None)
    console.print(renderable)
    return console.export_text()


def _sparse_payload() -> LiveStepTimeResult:
    """Build a live result whose forward phase was never measured."""
    return live_result_from_window(
        window_from_rank_averages(
            {
                0: {
                    "input_wait": 2.0,
                    "h2d": 0.0,
                    "backward": 60.0,
                    "optimizer_step": 10.0,
                    "step_time": 90.0,
                    "total_step": 92.0,
                }
            },
            expected_ranks=(0,),
            metrics=[
                _metric("input_wait", 2.0),
                _metric("h2d", 0.0),
                _metric("backward", 60.0),
                _metric("optimizer_step", 10.0),
                _metric("step_time", 90.0),
            ],
            steps_used=30,
        )
    )


def test_cli_columns_exclude_summary_only_statistics() -> None:
    metrics = [
        _metric("step_time", 100.0),
        _metric("compute", 80.0),
        _metric("dataloader_fetch", 10.0),
        _metric("total_step_cpu", 120.0),
    ]

    assert [metric.metric for metric in _table_metrics(metrics)] == [
        "step_time"
    ]


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

    renderer = StepTimeRenderer(session=Mock())
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

    renderer = StepTimeRenderer(session=Mock())
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

    renderer = StepTimeRenderer(session=session)
    text = _render_text(renderer.get_panel_renderable())

    session.refresh.assert_called_once_with()
    assert "STEP" in text


def test_cli_omits_missing_phase_and_shows_incomplete_data() -> None:
    renderer = StepTimeRenderer(session=Mock())
    text = _render_text(renderer.render(_sparse_payload()))

    assert "INCOMPLETE DATA" in text
    assert "forward" in text
    header = next(line for line in text.splitlines() if "Metric" in line)
    assert "FWD" not in header
    assert "RESIDUAL" not in header
    assert "BWD" in header
    assert "H2D" in header
    assert "0.0 ms" in text
    assert "60.0 ms" in text


def test_cli_drops_dead_view_when_session_window_expires() -> None:
    renderer = StepTimeRenderer(session=Mock())
    assert "60.0 ms" in _render_text(renderer.render(_sparse_payload()))

    expired = live_result_from_window(
        StepTimeWindow(),
        freshness="expired",
    )
    stale_text = _render_text(renderer.render(expired))

    assert "60.0 ms" not in stale_text
    assert "NO DATA" in stale_text
    assert "the last window expired" in stale_text


def test_cli_startup_shows_calm_waiting_panel() -> None:
    renderer = StepTimeRenderer(session=Mock())
    text = _render_text(
        renderer.render(
            live_result_from_window(
                StepTimeWindow(),
                freshness="cold",
            )
        )
    )

    assert "Waiting for first fully completed step" in text
    assert "NO DATA" not in text


def test_cli_renders_multi_rank_sparse_table() -> None:
    def _multi_metric(name: str, value: float) -> StepTimeMetric:
        return StepTimeMetric(
            metric=name,
            series=None,
            median_total=value,
            worst_total=value * 2,
            worst_rank=1,
            skew_pct=1.0,
        )

    per_rank_timing = {
        0: {
            "input_wait": 2.0,
            "backward": 60.0,
            "optimizer_step": 10.0,
            "step_time": 90.0,
            "total_step": 92.0,
        },
        1: {
            "input_wait": 2.0,
            "backward": 120.0,
            "optimizer_step": 10.0,
            "step_time": 180.0,
            "total_step": 182.0,
        },
    }
    payload = live_result_from_window(
        window_from_rank_averages(
            per_rank_timing,
            expected_ranks=(0, 1),
            metrics=[
                _multi_metric("input_wait", 2.0),
                _multi_metric("backward", 60.0),
                _multi_metric("optimizer_step", 10.0),
                _multi_metric("step_time", 90.0),
            ],
            steps_used=30,
        )
    )
    text = _render_text(StepTimeRenderer(session=Mock()).render(payload))

    header = next(line for line in text.splitlines() if "Metric" in line)
    assert "FWD" not in header
    assert "BWD" in header
    assert "Median avg" in text
    assert "Worst avg" in text
    assert "60.0 ms" in text
    assert "120.0 ms" in text
    assert "Worst Rank" in text
