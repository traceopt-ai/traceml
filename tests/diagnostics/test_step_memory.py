from __future__ import annotations

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
    sort_step_memory_summary_issues,
)
from traceml.diagnostics.common import DiagnosticIssue
from traceml.renderers.step_memory.schema import (
    StepMemoryCombinedCoverage,
    StepMemoryCombinedMetric,
    StepMemoryCombinedSeries,
    StepMemoryCombinedSummary,
)


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
    skew_pct: float = 0.0,
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
            skew_ratio=skew_pct,
            skew_pct=skew_pct,
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


def _rising_memory_metric(
    *,
    steps_used: int = 60,
    start: float = 4.0 * 1024.0 * 1024.0 * 1024.0,
    end: float = 6.0 * 1024.0 * 1024.0 * 1024.0,
    median_scale: float = 0.5,
    skew_pct: float = 0.0,
) -> StepMemoryCombinedMetric:
    worst = [
        start + (end - start) * (idx / float(steps_used - 1))
        for idx in range(steps_used)
    ]
    median = [value * median_scale for value in worst]
    return StepMemoryCombinedMetric(
        metric="peak_reserved",
        device="cuda:0",
        series=StepMemoryCombinedSeries(
            steps=list(range(steps_used)),
            median=median,
            worst=worst,
        ),
        summary=StepMemoryCombinedSummary(
            window_size=steps_used,
            steps_used=steps_used,
            median_peak=max(median),
            worst_peak=max(worst),
            worst_rank=1,
            skew_ratio=skew_pct,
            skew_pct=skew_pct,
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


def test_step_memory_summary_primary_uses_rule_priority() -> None:
    metric = _rising_memory_metric(skew_pct=0.4)
    result = build_step_memory_summary_diagnosis_result(
        [metric],
        gpu_total_bytes=6.1 * 1024.0 * 1024.0 * 1024.0,
    )

    assert result.primary.kind == "HIGH_PRESSURE"
    assert [issue.kind for issue in result.issues] == [
        "HIGH_PRESSURE",
        "IMBALANCE",
        "CREEP_CONFIRMED",
    ]


def test_step_memory_summary_primary_for_each_non_pressure_issue() -> None:
    imbalance = build_step_memory_summary_diagnosis_result(
        [
            _memory_metric(
                worst_peak=100.0,
                median_peak=70.0,
                skew_pct=0.3,
                steps_used=60,
            )
        ],
        gpu_total_bytes=1000.0,
    )
    assert imbalance.primary.kind == "IMBALANCE"

    confirmed = build_step_memory_summary_diagnosis_result(
        [_rising_memory_metric()],
        gpu_total_bytes=100.0 * 1024.0 * 1024.0 * 1024.0,
    )
    assert confirmed.primary.kind == "CREEP_CONFIRMED"

    early = build_step_memory_summary_diagnosis_result(
        [
            _rising_memory_metric(
                end=4.1 * 1024.0 * 1024.0 * 1024.0,
            )
        ],
        gpu_total_bytes=100.0 * 1024.0 * 1024.0 * 1024.0,
    )
    assert early.primary.kind == "CREEP_EARLY"

    balanced = build_step_memory_summary_diagnosis_result(
        [_memory_metric(worst_peak=90.0, median_peak=88.0, steps_used=60)],
        gpu_total_bytes=1000.0,
    )
    assert balanced.primary.kind == "BALANCED"

    no_data = build_step_memory_summary_diagnosis_result([])
    assert no_data.primary.kind == "NO_DATA"


def test_step_memory_issue_sort_uses_domain_priority() -> None:
    issues = (
        DiagnosticIssue(
            kind="CREEP_CONFIRMED",
            status="MEMORY CREEP",
            severity="warn",
            summary="creep",
            action="check",
            score=100.0,
        ),
        DiagnosticIssue(
            kind="HIGH_PRESSURE",
            status="HIGH PRESSURE",
            severity="warn",
            summary="pressure",
            action="reduce",
            score=0.93,
        ),
        DiagnosticIssue(
            kind="IMBALANCE",
            status="IMBALANCE",
            severity="warn",
            summary="imbalance",
            action="inspect",
            score=0.4,
        ),
    )

    assert [
        issue.kind for issue in sort_step_memory_summary_issues(issues)
    ] == [
        "HIGH_PRESSURE",
        "IMBALANCE",
        "CREEP_CONFIRMED",
    ]
