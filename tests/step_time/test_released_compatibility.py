"""One-release compatibility checks for the completed Step Time migration."""

from dataclasses import asdict
import inspect
import statistics

import pytest

from tests.step_time.factories import window_from_rank_averages
from traceml_ai.diagnostics.step_time import (
    StepDiagnosis,
    build_step_diagnosis,
    build_step_diagnosis_result,
    diagnose_step_time_window,
)
from traceml_ai.diagnostics.step_time.api import (
    build_step_diagnosis as api_build_step_diagnosis,
)
from traceml_ai.diagnostics.step_time.policy import (
    DiagnosisThresholds,
    StepTimeDiagnosisPolicy,
)
from traceml_ai.reporting.summaries.step_time import RankStepSummary
from traceml_ai.step_time.model import StepTimeMetric


def _metrics(
    timing: dict[int, dict[str, float]],
    *,
    steps: int = 20,
) -> tuple[StepTimeMetric, ...]:
    metrics = []
    for key in (
        "input_wait",
        "h2d",
        "forward",
        "backward",
        "optimizer_step",
        "step_time",
        "residual_proxy",
    ):
        values = {rank: row[key] for rank, row in timing.items() if key in row}
        if not values:
            continue
        median = float(statistics.median(values.values()))
        worst_rank = max(values, key=lambda rank: (values[rank], -rank))
        worst = float(values[worst_rank])
        skew = (
            max(0.0, (worst - median) / median)
            if len(values) > 1 and median > 0.0
            else None
        )
        representative = min(
            values,
            key=lambda rank: (
                abs(values[rank] - median),
                values[rank],
                rank,
            ),
        )
        metrics.append(
            StepTimeMetric(
                metric=key,
                series=None,
                window_size=steps,
                steps_used=steps,
                median_total=median,
                worst_total=worst,
                worst_rank=worst_rank,
                skew_ratio=skew,
                skew_pct=skew,
                representative_rank=representative,
                representative_total=float(values[representative]),
                measured_ranks=tuple(sorted(values)),
            )
        )
    return tuple(metrics)


_COMPLETE = {
    0: {
        "input_wait": 40.0,
        "h2d": 5.0,
        "forward": 20.0,
        "backward": 20.0,
        "optimizer_step": 10.0,
        "step_time": 100.0,
        "residual_proxy": 45.0,
        "total_step": 140.0,
    },
    1: {
        "input_wait": 10.0,
        "h2d": 5.0,
        "forward": 20.0,
        "backward": 40.0,
        "optimizer_step": 10.0,
        "step_time": 100.0,
        "residual_proxy": 25.0,
        "total_step": 110.0,
    },
}


@pytest.mark.parametrize(
    ("strategy", "timing"),
    [
        ("ddp", _COMPLETE),
        (
            "ddp",
            {
                0: _COMPLETE[0],
                1: {
                    key: value
                    for key, value in _COMPLETE[1].items()
                    if key != "forward"
                },
            },
        ),
        (
            "ddp",
            {
                0: {**_COMPLETE[0], "forward": 0.0},
                1: _COMPLETE[1],
            },
        ),
        ("fsdp", _COMPLETE),
    ],
)
def test_released_rank_map_api_matches_canonical_diagnosis(
    strategy: str,
    timing: dict[int, dict[str, float]],
) -> None:
    metrics = _metrics(timing)
    policy = StepTimeDiagnosisPolicy(name="compatibility-test")
    window = window_from_rank_averages(
        timing,
        metrics=metrics,
        expected_ranks=tuple(timing),
        training_strategy=strategy,
    )
    canonical = diagnose_step_time_window(
        window,
        policy=policy,
        training_strategy=strategy,
        include_attribution=True,
    )

    with pytest.warns(DeprecationWarning):
        compatibility = build_step_diagnosis_result(
            metrics,
            per_rank_timing=timing,
            expected_ranks=tuple(timing),
            training_strategy=strategy,
        )

    assert compatibility == canonical
    assert compatibility.metric_attribution["step_time"]["top_ranks"]


def test_released_diagnosis_imports_and_signatures_remain_available() -> None:
    assert build_step_diagnosis is api_build_step_diagnosis
    parameters = inspect.signature(build_step_diagnosis).parameters
    assert tuple(parameters)[-4:] == (
        "per_rank_timing",
        "expected_ranks",
        "diagnosis_clock",
        "training_strategy",
    )

    metrics = _metrics(_COMPLETE)
    with pytest.warns(DeprecationWarning):
        primary = build_step_diagnosis(metrics, per_rank_timing=_COMPLETE)
    assert isinstance(primary, StepDiagnosis)


def test_released_rank_map_options_still_reach_canonical_policy() -> None:
    metrics = _metrics(_COMPLETE, steps=5)
    thresholds = DiagnosisThresholds(
        overhead_share_warn=0.95,
        overhead_share_crit=0.99,
    )
    with pytest.warns(DeprecationWarning):
        result = build_step_diagnosis_result(
            metrics,
            thresholds,
            per_rank_timing={0: _COMPLETE[0]},
            expected_ranks=(0, 1),
            diagnosis_clock="gpu",
        )

    assert result.primary.kind == "INCOMPLETE_DATA"
    assert result.primary.steps_used == 5


def test_rank_step_summary_keeps_its_released_dataclass_shape() -> None:
    summary = RankStepSummary(
        steps_analyzed=20,
        avg_dataloader_ms=1.0,
        avg_input_wait_ms=2.0,
        avg_step_time_ms=3.0,
        avg_h2d_ms=None,
        avg_forward_ms=4.0,
        avg_backward_ms=5.0,
        avg_optimizer_ms=0.0,
        avg_traced_step_ms=3.0,
        avg_compute_ms=9.0,
        avg_residual_ms=1.0,
        avg_total_step_ms=4.0,
    )

    assert asdict(summary)["avg_h2d_ms"] is None
    assert asdict(summary)["avg_optimizer_ms"] == 0.0
