from rich.console import Console
from rich.panel import Panel

from traceml.renderers.step_memory.formatter import StepMemoryRichFormatter
from traceml.renderers.step_memory.schema import (
    StepMemoryCombinedCoverage,
    StepMemoryCombinedMetric,
    StepMemoryCombinedResult,
    StepMemoryCombinedSeries,
    StepMemoryCombinedSummary,
)


def _metric(
    name: str,
    *,
    world_size: int,
    ranks_present: int,
    median_peak: float = 1024.0,
    worst_peak: float = 2048.0,
    worst_rank: int | None = 1,
    skew_pct: float = 1.0,
) -> StepMemoryCombinedMetric:
    return StepMemoryCombinedMetric(
        metric=name,
        device="cuda:0",
        series=StepMemoryCombinedSeries(
            steps=[1, 2, 3, 4, 5, 6],
            median=[512.0, 512.0, 512.0, 1024.0, 1024.0, 1024.0],
            worst=[1024.0, 1024.0, 1024.0, 2048.0, 2048.0, 2048.0],
        ),
        summary=StepMemoryCombinedSummary(
            window_size=6,
            steps_used=6,
            median_peak=median_peak,
            worst_peak=worst_peak,
            worst_rank=worst_rank,
            skew_ratio=skew_pct,
            skew_pct=skew_pct,
        ),
        coverage=StepMemoryCombinedCoverage(
            expected_steps=6,
            steps_used=6,
            completed_step=6,
            world_size=world_size,
            ranks_present=ranks_present,
            incomplete=False,
        ),
    )


def _render_text(panel: Panel) -> str:
    console = Console(
        force_terminal=True,
        color_system=None,
        width=140,
        record=True,
    )
    console.print(panel)
    return console.export_text()


def test_step_memory_formatter_renders_empty_state() -> None:
    formatter = StepMemoryRichFormatter()

    panel = formatter.format(
        StepMemoryCombinedResult(
            metrics=[],
            status_message="Waiting for memory samples",
        )
    )

    assert isinstance(panel, Panel)
    assert "Model Step Memory" in _render_text(panel)
    assert "Waiting for memory samples" in _render_text(panel)


def test_step_memory_formatter_renders_single_rank_shape() -> None:
    formatter = StepMemoryRichFormatter()

    text = _render_text(
        formatter.format(
            StepMemoryCombinedResult(
                metrics=[
                    _metric(
                        "peak_reserved",
                        world_size=1,
                        ranks_present=1,
                    ),
                    _metric(
                        "peak_allocated",
                        world_size=1,
                        ranks_present=1,
                    ),
                ],
                status_message="",
            )
        )
    )

    assert "Peak Allocated" in text
    assert "Peak Reserved" in text
    assert "Peak (max/6)" in text
    assert "Head/Tail Delta" in text
    assert "Median Peak" not in text
    assert "Peaks = max over last K aligned steps for the only rank." in text


def test_step_memory_formatter_renders_multi_rank_shape() -> None:
    formatter = StepMemoryRichFormatter()

    text = _render_text(
        formatter.format(
            StepMemoryCombinedResult(
                metrics=[
                    _metric(
                        "peak_allocated",
                        world_size=4,
                        ranks_present=4,
                    ),
                    _metric(
                        "peak_reserved",
                        world_size=4,
                        ranks_present=4,
                    ),
                ],
                status_message="",
            )
        )
    )

    assert "Median Peak (max/6)" in text
    assert "Worst Peak (max/6)" in text
    assert "Worst Rank" in text
    assert "r1" in text
    assert "Skew (%)" in text
    assert "Head/Tail Delta (worst)" in text
    assert (
        "Peaks = per-rank max over last K; median/worst = across ranks."
        in text
    )
