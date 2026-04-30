from traceml.diagnostics.step_memory import (
    DEFAULT_STEP_MEMORY_THRESHOLDS,
    StepMemoryDiagnosisThresholds,
    build_step_memory_diagnosis,
    build_step_memory_summary_diagnosis_result,
)
from traceml.diagnostics.step_memory.adapters import (
    build_step_memory_summary_signals,
)
from traceml.diagnostics.step_memory.rules import (
    DEFAULT_STEP_MEMORY_SUMMARY_RULES,
    run_step_memory_summary_rules,
)
from traceml.diagnostics.step_memory_summary import (
    build_step_memory_summary_diagnosis_result as legacy_summary_builder,
)
from traceml.diagnostics.step_memory_trend import (
    evaluate_step_memory_creep as legacy_trend_builder,
)
from traceml.diagnostics.step_memory.trend import evaluate_step_memory_creep
from traceml.renderers.step_memory.schema import (
    StepMemoryCombinedCoverage,
    StepMemoryCombinedMetric,
    StepMemoryCombinedSeries,
    StepMemoryCombinedSummary,
)


def _metric(
    name: str = "peak_reserved",
    *,
    steps_used: int = 60,
    worst_peak: float = 100.0,
    median_peak: float = 90.0,
    skew_pct: float = 0.0,
) -> StepMemoryCombinedMetric:
    return StepMemoryCombinedMetric(
        metric=name,
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
            worst_rank=0,
            skew_ratio=skew_pct,
            skew_pct=skew_pct,
        ),
        coverage=StepMemoryCombinedCoverage(
            expected_steps=steps_used,
            steps_used=steps_used,
            completed_step=steps_used,
            world_size=1,
            ranks_present=1,
            incomplete=False,
        ),
    )


def _rising_metric(
    name: str = "peak_reserved",
    *,
    steps_used: int = 50,
    start_bytes: float = 4.0 * 1024.0 * 1024.0 * 1024.0,
    end_bytes: float = 7.4 * 1024.0 * 1024.0 * 1024.0,
) -> StepMemoryCombinedMetric:
    values = [
        start_bytes + (end_bytes - start_bytes) * (idx / float(steps_used - 1))
        for idx in range(steps_used)
    ]
    return StepMemoryCombinedMetric(
        metric=name,
        device="cuda:0",
        series=StepMemoryCombinedSeries(
            steps=list(range(steps_used)),
            median=values,
            worst=values,
        ),
        summary=StepMemoryCombinedSummary(
            window_size=steps_used,
            steps_used=steps_used,
            median_peak=max(values),
            worst_peak=max(values),
            worst_rank=0,
            skew_ratio=0.0,
            skew_pct=0.0,
        ),
        coverage=StepMemoryCombinedCoverage(
            expected_steps=steps_used,
            steps_used=steps_used,
            completed_step=steps_used,
            world_size=1,
            ranks_present=1,
            incomplete=False,
        ),
    )


def test_step_memory_package_exports_live_and_summary_builders():
    metrics = [_metric()]

    live = build_step_memory_diagnosis(metrics)
    summary = build_step_memory_summary_diagnosis_result(metrics)

    assert live.kind == "BALANCED"
    assert summary.primary.kind == "BALANCED"
    assert summary.metric_attribution["peak_reserved"]["steps_used"] == 60


def test_step_memory_fifty_step_window_detects_large_creep():
    diagnosis = build_step_memory_diagnosis([_rising_metric()])

    assert diagnosis.kind == "CREEP_CONFIRMED"
    assert diagnosis.reason == "peak reserved is rising across the window."


def test_step_memory_summary_adapters_and_rules_are_importable():
    metrics = [
        _metric(
            worst_peak=98.0,
            median_peak=90.0,
            skew_pct=0.0,
        )
    ]
    signals = build_step_memory_summary_signals(
        metrics,
        gpu_total_bytes=100.0,
    )

    issues = run_step_memory_summary_rules(signals["peak_reserved"])

    assert DEFAULT_STEP_MEMORY_SUMMARY_RULES
    assert issues
    assert issues[0].kind == "HIGH_PRESSURE"


def test_step_memory_legacy_import_paths_are_thin_shims():
    assert legacy_summary_builder is build_step_memory_summary_diagnosis_result
    assert legacy_trend_builder is evaluate_step_memory_creep
    assert isinstance(
        DEFAULT_STEP_MEMORY_THRESHOLDS,
        StepMemoryDiagnosisThresholds,
    )


def test_step_memory_summary_enrichment_fails_open(monkeypatch):
    import traceml.diagnostics.step_memory.adapters as adapters

    def broken_adapter(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        adapters,
        "build_step_memory_summary_signals",
        broken_adapter,
    )

    result = build_step_memory_summary_diagnosis_result([_metric()])

    assert result.primary.kind == "BALANCED"
    assert result.issues == ()
    assert result.metric_attribution == {}
