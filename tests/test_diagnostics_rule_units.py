from __future__ import annotations

from traceml.diagnostics.process.api import build_process_diagnosis_result
from traceml.diagnostics.process.context import build_process_summary_signals
from traceml.diagnostics.process.rules import (
    GPUMemoryReservedOverhangRule,
    HighCPUProcessPressureRule,
    HighGPUMemoryPressureRule,
    HighRSSPressureRule,
    RankGPUMemoryImbalanceRule,
)
from traceml.diagnostics.step_memory import (
    build_step_memory_summary_diagnosis_result,
)
from traceml.diagnostics.step_memory.adapters import (
    StepMemorySummaryMetricSignals,
    StepMemorySummaryTrendSignals,
)
from traceml.diagnostics.step_memory.rules import (
    CreepConfirmedRule,
    CreepEarlyRule,
    HighPressureRule,
    ImbalanceRule,
)
from traceml.diagnostics.step_time.api import (
    DEFAULT_THRESHOLDS,
    build_step_diagnosis_result,
)
from traceml.diagnostics.step_time.context import build_step_time_context
from traceml.diagnostics.step_time.rules import (
    ComputeBoundRule,
    ComputeStragglerRule,
    InputBoundRule,
    InputStragglerRule,
    WaitHeavyRule,
)
from traceml.diagnostics.system.api import build_system_diagnosis_result
from traceml.diagnostics.system.context import build_system_summary_signals
from traceml.diagnostics.system.rules import (
    GPUUtilImbalanceRule,
    HighCPUPressureRule,
    HighRAMPressureRule,
    LowGPUUtilizationRule,
)
from traceml.renderers.step_memory.schema import (
    StepMemoryCombinedCoverage,
    StepMemoryCombinedMetric,
    StepMemoryCombinedSeries,
    StepMemoryCombinedSummary,
)
from traceml.renderers.step_time.schema import (
    StepCombinedTimeCoverage,
    StepCombinedTimeMetric,
    StepCombinedTimeSeries,
    StepCombinedTimeSummary,
)


def _time_metric(
    name: str,
    *,
    median: float,
    worst: float,
    worst_rank: int | None = 1,
    skew: float = 0.0,
    world_size: int = 2,
    steps: int = 64,
) -> StepCombinedTimeMetric:
    return StepCombinedTimeMetric(
        metric=name,
        clock="mixed",
        series=StepCombinedTimeSeries(
            steps=list(range(steps)),
            median=[median] * steps,
            worst=[worst] * steps,
            sum=[median * world_size] * steps,
        ),
        summary=StepCombinedTimeSummary(
            window_size=steps,
            steps_used=steps,
            median_total=median,
            worst_total=worst,
            worst_rank=worst_rank,
            skew_ratio=skew,
            skew_pct=skew,
        ),
        coverage=StepCombinedTimeCoverage(
            expected_steps=steps,
            steps_used=steps,
            completed_step=steps,
            world_size=world_size,
            ranks_present=world_size,
            incomplete=False,
        ),
    )


def _time_context(*metrics: StepCombinedTimeMetric):
    return build_step_time_context(
        metrics=metrics,
        thresholds=DEFAULT_THRESHOLDS,
    )


def _single_rank_step_metrics(
    *,
    step: float = 100.0,
    dataloader: float = 5.0,
    forward: float = 30.0,
    backward: float = 50.0,
    optimizer: float = 10.0,
    wait: float = 5.0,
) -> tuple[StepCombinedTimeMetric, ...]:
    return (
        _time_metric(
            "step_time",
            median=step,
            worst=step,
            worst_rank=0,
            world_size=1,
        ),
        _time_metric(
            "dataloader_fetch",
            median=dataloader,
            worst=dataloader,
            worst_rank=0,
            world_size=1,
        ),
        _time_metric(
            "forward",
            median=forward,
            worst=forward,
            worst_rank=0,
            world_size=1,
        ),
        _time_metric(
            "backward",
            median=backward,
            worst=backward,
            worst_rank=0,
            world_size=1,
        ),
        _time_metric(
            "optimizer_step",
            median=optimizer,
            worst=optimizer,
            worst_rank=0,
            world_size=1,
        ),
        _time_metric(
            "wait_proxy",
            median=wait,
            worst=wait,
            worst_rank=0,
            world_size=1,
        ),
    )


def test_step_time_rules_trigger_and_no_trigger_cases() -> None:
    input_ctx = _time_context(
        _time_metric("step_time", median=220.0, worst=250.0),
        _time_metric("dataloader_fetch", median=10.0, worst=45.0, skew=0.2),
        _time_metric("forward", median=40.0, worst=40.0),
        _time_metric("backward", median=130.0, worst=130.0),
        _time_metric("optimizer_step", median=20.0, worst=20.0),
        _time_metric("wait_proxy", median=20.0, worst=20.0),
    )
    assert InputStragglerRule().evaluate(input_ctx).kind == "INPUT_STRAGGLER"
    assert (
        InputStragglerRule().evaluate(
            _time_context(*_single_rank_step_metrics())
        )
        is None
    )

    compute_ctx = _time_context(
        _time_metric("step_time", median=240.0, worst=310.0),
        _time_metric("dataloader_fetch", median=10.0, worst=10.0),
        _time_metric("forward", median=40.0, worst=90.0, skew=0.2),
        _time_metric("backward", median=130.0, worst=160.0, skew=0.15),
        _time_metric("optimizer_step", median=20.0, worst=20.0),
        _time_metric("wait_proxy", median=40.0, worst=40.0),
    )
    assert (
        ComputeStragglerRule().evaluate(compute_ctx).kind
        == "COMPUTE_STRAGGLER"
    )
    assert (
        ComputeStragglerRule().evaluate(
            _time_context(*_single_rank_step_metrics())
        )
        is None
    )

    assert (
        InputBoundRule()
        .evaluate(
            _time_context(
                *_single_rank_step_metrics(
                    step=100.0,
                    dataloader=35.0,
                    forward=20.0,
                    backward=30.0,
                    optimizer=5.0,
                    wait=10.0,
                )
            )
        )
        .kind
        == "INPUT_BOUND"
    )
    assert (
        InputBoundRule().evaluate(
            _time_context(*_single_rank_step_metrics(dataloader=10.0))
        )
        is None
    )

    assert (
        WaitHeavyRule()
        .evaluate(_time_context(*_single_rank_step_metrics(wait=20.0)))
        .kind
        == "WAIT_HEAVY"
    )
    assert (
        WaitHeavyRule().evaluate(
            _time_context(*_single_rank_step_metrics(wait=5.0))
        )
        is None
    )

    assert (
        ComputeBoundRule()
        .evaluate(
            _time_context(*_single_rank_step_metrics(dataloader=2.0, wait=3.0))
        )
        .kind
        == "COMPUTE_BOUND"
    )
    assert (
        ComputeBoundRule().evaluate(
            _time_context(
                *_single_rank_step_metrics(dataloader=35.0, wait=3.0)
            )
        )
        is None
    )


def test_step_time_primary_selection_combines_input_and_compute_stragglers() -> (
    None
):
    result = build_step_diagnosis_result(
        [
            _time_metric("step_time", median=240.0, worst=330.0),
            _time_metric(
                "dataloader_fetch", median=10.0, worst=45.0, skew=0.2
            ),
            _time_metric("forward", median=40.0, worst=90.0, skew=0.2),
            _time_metric("backward", median=130.0, worst=160.0, skew=0.15),
            _time_metric("optimizer_step", median=20.0, worst=20.0),
            _time_metric("wait_proxy", median=40.0, worst=40.0),
        ]
    )

    assert result.primary.kind == "STRAGGLER"
    assert {issue.kind for issue in result.issues} >= {
        "INPUT_STRAGGLER",
        "COMPUTE_STRAGGLER",
        "STRAGGLER",
    }


def _trend(
    *,
    early: bool = False,
    confirmed: bool = False,
) -> StepMemorySummaryTrendSignals:
    return StepMemorySummaryTrendSignals(
        eligible=True,
        baseline_avg_bytes=100.0,
        mid_avg_bytes=120.0,
        recent_avg_bytes=160.0,
        overall_abs_delta_bytes=60.0,
        overall_worst_growth_pct=0.6,
        overall_median_growth_pct=0.4,
        early=early,
        confirmed=confirmed,
        score=2.5,
    )


def _memory_signal(
    *,
    steps_used: int = 60,
    pressure_frac: float | None = 0.5,
    skew_pct: float = 0.0,
    trend: StepMemorySummaryTrendSignals | None = None,
) -> StepMemorySummaryMetricSignals:
    return StepMemorySummaryMetricSignals(
        metric="peak_reserved",
        device="cuda:0",
        steps_used=steps_used,
        window_size=steps_used,
        completed_step=steps_used,
        ranks_seen=2,
        worst_rank=1,
        worst_peak_bytes=90.0,
        median_peak_bytes=80.0,
        skew_ratio=skew_pct,
        skew_pct=skew_pct,
        pressure_frac=pressure_frac,
        trend=trend or _trend(),
    )


def _memory_metric(
    *,
    worst_peak: float = 90.0,
    median_peak: float = 80.0,
    steps_used: int = 60,
) -> StepMemoryCombinedMetric:
    return StepMemoryCombinedMetric(
        metric="peak_reserved",
        device="cuda:0",
        series=StepMemoryCombinedSeries(
            steps=list(range(steps_used)),
            median=[median_peak] * steps_used,
            worst=[worst_peak] * steps_used,
        ),
        summary=StepMemoryCombinedSummary(
            window_size=steps_used,
            steps_used=steps_used,
            median_peak=median_peak,
            worst_peak=worst_peak,
            worst_rank=1,
            skew_ratio=0.0,
            skew_pct=0.0,
        ),
        coverage=StepMemoryCombinedCoverage(
            expected_steps=steps_used,
            steps_used=steps_used,
            completed_step=steps_used,
            world_size=2,
            ranks_present=2,
            incomplete=False,
        ),
    )


def test_step_memory_rules_trigger_and_no_trigger_cases() -> None:
    assert (
        HighPressureRule().evaluate(_memory_signal(pressure_frac=0.93)).kind
        == "HIGH_PRESSURE"
    )
    assert (
        HighPressureRule().evaluate(_memory_signal(pressure_frac=0.2)) is None
    )

    assert (
        ImbalanceRule().evaluate(_memory_signal(skew_pct=0.3)).kind
        == "IMBALANCE"
    )
    assert ImbalanceRule().evaluate(_memory_signal(skew_pct=0.01)) is None

    assert (
        CreepConfirmedRule()
        .evaluate(_memory_signal(trend=_trend(confirmed=True)))
        .kind
        == "CREEP_CONFIRMED"
    )
    assert (
        CreepConfirmedRule().evaluate(_memory_signal(trend=_trend())) is None
    )

    assert (
        CreepEarlyRule()
        .evaluate(_memory_signal(trend=_trend(early=True)))
        .kind
        == "CREEP_EARLY"
    )
    assert (
        CreepEarlyRule().evaluate(
            _memory_signal(trend=_trend(early=True, confirmed=True))
        )
        is None
    )


def test_step_memory_primary_selection_uses_sorted_summary_issues() -> None:
    result = build_step_memory_summary_diagnosis_result(
        [_memory_metric(worst_peak=96.0, median_peak=80.0)],
        gpu_total_bytes=100.0,
    )

    assert result.primary.kind == "HIGH_PRESSURE"
    assert result.issues[0].kind == "HIGH_PRESSURE"


def _system_signals(**overrides):
    values = dict(
        duration_s=10.0,
        samples=10,
        cpu_avg_percent=20.0,
        cpu_peak_percent=40.0,
        ram_avg_bytes=100.0,
        ram_peak_bytes=200.0,
        ram_total_bytes=1000.0,
        gpu_available=True,
        gpu_count=2,
        gpu_util_avg_percent=70.0,
        gpu_util_peak_percent=90.0,
        gpu_mem_avg_bytes=100.0,
        gpu_mem_peak_bytes=200.0,
        gpu_temp_avg_c=None,
        gpu_temp_peak_c=None,
        gpu_power_avg_w=None,
        gpu_power_peak_w=None,
        per_gpu={
            0: {"util_avg_percent": 70.0, "mem_peak_bytes": 200.0},
            1: {"util_avg_percent": 68.0, "mem_peak_bytes": 210.0},
        },
    )
    values.update(overrides)
    return build_system_summary_signals(**values)


def test_system_rules_trigger_and_no_trigger_cases() -> None:
    assert (
        LowGPUUtilizationRule()
        .evaluate(_system_signals(gpu_util_avg_percent=10.0))
        .kind
        == "LOW_GPU_UTILIZATION"
    )
    assert (
        LowGPUUtilizationRule().evaluate(
            _system_signals(gpu_util_avg_percent=60.0)
        )
        is None
    )

    assert (
        HighCPUPressureRule()
        .evaluate(_system_signals(cpu_avg_percent=95.0))
        .kind
        == "HIGH_CPU_PRESSURE"
    )
    assert (
        HighCPUPressureRule().evaluate(_system_signals(cpu_avg_percent=50.0))
        is None
    )

    assert (
        HighRAMPressureRule()
        .evaluate(_system_signals(ram_peak_bytes=930.0))
        .kind
        == "HIGH_RAM_PRESSURE"
    )
    assert (
        HighRAMPressureRule().evaluate(_system_signals(ram_peak_bytes=500.0))
        is None
    )

    assert (
        GPUUtilImbalanceRule()
        .evaluate(
            _system_signals(
                per_gpu={
                    0: {"util_avg_percent": 90.0, "mem_peak_bytes": 200.0},
                    1: {"util_avg_percent": 40.0, "mem_peak_bytes": 210.0},
                }
            )
        )
        .kind
        == "GPU_UTIL_IMBALANCE"
    )
    assert GPUUtilImbalanceRule().evaluate(_system_signals()) is None


def test_system_primary_selection_uses_highest_severity_issue() -> None:
    result = build_system_diagnosis_result(
        duration_s=10.0,
        system_samples=10,
        cpu_avg_percent=95.0,
        cpu_peak_percent=98.0,
        ram_avg_bytes=100.0,
        ram_peak_bytes=930.0,
        ram_total_bytes=1000.0,
        gpu_available=True,
        gpu_count=1,
        gpu_util_avg_percent=70.0,
        gpu_util_peak_percent=90.0,
        gpu_mem_avg_bytes=100.0,
        gpu_mem_peak_bytes=200.0,
        gpu_temp_avg_c=None,
        gpu_temp_peak_c=None,
        gpu_power_avg_w=None,
        gpu_power_peak_w=None,
        per_gpu={},
    )

    assert result.primary.kind == "HIGH_CPU_PRESSURE"
    assert result.primary.severity == "crit"


def _process_signals(**overrides):
    values = dict(
        duration_s=10.0,
        samples=10,
        distinct_ranks=2,
        distinct_pids=2,
        cpu_avg_percent=120.0,
        cpu_peak_percent=200.0,
        cpu_logical_core_count=8,
        ram_avg_bytes=100.0,
        ram_peak_bytes=200.0,
        ram_total_bytes=1000.0,
        gpu_available=True,
        gpu_count=2,
        gpu_device_index=None,
        gpu_mem_used_avg_bytes=100.0,
        gpu_mem_used_peak_bytes=200.0,
        gpu_mem_reserved_avg_bytes=120.0,
        gpu_mem_reserved_peak_bytes=240.0,
        gpu_mem_total_bytes=1000.0,
        per_rank={
            0: {
                "ram_peak_bytes": 200.0,
                "gpu_mem_used_peak_bytes": 200.0,
                "gpu_mem_reserved_peak_bytes": 240.0,
                "gpu_mem_total_bytes": 1000.0,
            },
            1: {
                "ram_peak_bytes": 190.0,
                "gpu_mem_used_peak_bytes": 190.0,
                "gpu_mem_reserved_peak_bytes": 230.0,
                "gpu_mem_total_bytes": 1000.0,
            },
        },
    )
    values.update(overrides)
    return build_process_summary_signals(**values)


def test_process_rules_trigger_and_no_trigger_cases() -> None:
    assert (
        HighCPUProcessPressureRule()
        .evaluate(_process_signals(cpu_avg_percent=700.0))
        .kind
        == "HIGH_CPU_PROCESS_PRESSURE"
    )
    assert (
        HighCPUProcessPressureRule().evaluate(
            _process_signals(cpu_avg_percent=120.0)
        )
        is None
    )

    assert (
        HighRSSPressureRule()
        .evaluate(_process_signals(ram_peak_bytes=930.0))
        .kind
        == "HIGH_RSS_PRESSURE"
    )
    assert (
        HighRSSPressureRule().evaluate(_process_signals(ram_peak_bytes=500.0))
        is None
    )

    assert (
        HighGPUMemoryPressureRule()
        .evaluate(_process_signals(gpu_mem_reserved_peak_bytes=930.0))
        .kind
        == "HIGH_GPU_MEMORY_PRESSURE"
    )
    assert (
        HighGPUMemoryPressureRule().evaluate(
            _process_signals(gpu_mem_reserved_peak_bytes=500.0)
        )
        is None
    )

    assert (
        GPUMemoryReservedOverhangRule()
        .evaluate(
            _process_signals(
                gpu_mem_used_peak_bytes=400.0,
                gpu_mem_reserved_peak_bytes=700.0,
            )
        )
        .kind
        == "GPU_MEMORY_RESERVED_OVERHANG"
    )
    assert (
        GPUMemoryReservedOverhangRule().evaluate(
            _process_signals(
                gpu_mem_used_peak_bytes=400.0,
                gpu_mem_reserved_peak_bytes=450.0,
            )
        )
        is None
    )

    assert (
        RankGPUMemoryImbalanceRule()
        .evaluate(
            _process_signals(
                per_rank={
                    0: {
                        "gpu_mem_used_peak_bytes": 900.0,
                        "gpu_mem_reserved_peak_bytes": 900.0,
                        "gpu_mem_total_bytes": 1000.0,
                    },
                    1: {
                        "gpu_mem_used_peak_bytes": 400.0,
                        "gpu_mem_reserved_peak_bytes": 400.0,
                        "gpu_mem_total_bytes": 1000.0,
                    },
                }
            )
        )
        .kind
        == "RANK_GPU_MEMORY_IMBALANCE"
    )
    assert RankGPUMemoryImbalanceRule().evaluate(_process_signals()) is None


def test_process_primary_selection_uses_gpu_memory_pressure_first() -> None:
    result = build_process_diagnosis_result(
        duration_s=10.0,
        process_samples=10,
        distinct_ranks=2,
        distinct_pids=2,
        cpu_avg_percent=700.0,
        cpu_peak_percent=800.0,
        cpu_logical_core_count=8,
        ram_avg_bytes=100.0,
        ram_peak_bytes=930.0,
        ram_total_bytes=1000.0,
        gpu_available=True,
        gpu_count=1,
        gpu_device_index=0,
        gpu_mem_used_avg_bytes=600.0,
        gpu_mem_used_peak_bytes=850.0,
        gpu_mem_reserved_avg_bytes=650.0,
        gpu_mem_reserved_peak_bytes=930.0,
        gpu_mem_total_bytes=1000.0,
        per_rank={},
    )

    assert result.primary.kind == "HIGH_GPU_MEMORY_PRESSURE"
    assert result.primary.severity == "crit"
