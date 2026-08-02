from __future__ import annotations

from pathlib import Path

from tests.step_time.factories import (
    live_result_from_window,
    window_from_rank_averages,
)
from tests.step_time.scenarios import (
    StepTimeScenario,
    create_step_time_database,
)
from traceml_ai.renderers.step_time.compute import StepCombinedComputer
from traceml_ai.step_time.model import StepTimeMetric, StepTimeWindow


def test_dashboard_adapter_preserves_selected_gpu_clock(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "telemetry.db"
    create_step_time_database(
        db_path,
        StepTimeScenario(
            name="gpu_adapter",
            profiles={0: {"input_wait": 5.0, "step_time": 30.0}},
            steps=(1, 2),
            clock="gpu",
        ),
    )

    result = StepCombinedComputer(
        db_path=str(db_path),
        window_size=2,
    ).compute_dashboard()

    assert result.window is not None
    metrics = {metric.metric: metric for metric in result.window.metrics}
    assert metrics["input_wait"].worst_total == 5.0
    assert metrics["step_time"].worst_total == 30.0
    assert result.window.clock == "gpu"
    assert result.training_strategy == "ddp"
    assert result.window.per_rank_timing[0]["input_wait"] == 5.0


def test_worst_rank_requires_measured_total_step() -> None:
    assert (
        window_from_rank_averages(
            {
                0: {"step_time": 100.0},
                1: {"step_time": 400.0},
            }
        ).worst_rank
        is None
    )
    assert (
        window_from_rank_averages(
            {
                0: {"step_time": 100.0, "total_step": 105.0},
                1: {"step_time": 400.0},
            }
        ).worst_rank
        == 0
    )


def _live_window() -> StepTimeWindow:
    metric = StepTimeMetric(
        metric="step_time",
        series=None,
        window_size=1,
        steps_used=1,
        median_total=10.0,
        worst_total=10.0,
        worst_rank=0,
        skew_ratio=0.0,
        skew_pct=0.0,
        measured_ranks=(0,),
    )
    return window_from_rank_averages(
        {0: {"step_time": 10.0, "total_step": 10.0}},
        expected_ranks=(0,),
        metrics=(metric,),
    )


def test_dashboard_adapter_projects_live_freshness(monkeypatch) -> None:
    computer = StepCombinedComputer(db_path=":memory:")
    window = _live_window()
    results = iter(
        (
            live_result_from_window(window, freshness="live"),
            live_result_from_window(
                window,
                freshness="bridged",
                status_message="STALE (no metrics this tick)",
            ),
            live_result_from_window(StepTimeWindow(), freshness="cold"),
            live_result_from_window(StepTimeWindow(), freshness="expired"),
        )
    )
    monkeypatch.setattr(computer._session, "refresh", lambda: next(results))

    live = computer.compute_dashboard()
    bridged = computer.compute_dashboard()
    cold = computer.compute_dashboard()
    expired = computer.compute_dashboard()

    assert live.window is window and live.had_ok is True
    assert bridged.window is window and bridged.had_ok is True
    assert bridged.status_message.startswith("STALE")
    assert cold.window is None and cold.had_ok is False
    assert expired.window is None and expired.had_ok is True


def test_dashboard_adapter_has_no_cli_alias() -> None:
    assert not hasattr(StepCombinedComputer, "compute_cli")
