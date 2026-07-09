# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Adapters that feed summary step-time data into shared diagnosis rules."""

import math
from dataclasses import dataclass
from typing import Any, Dict

from traceml_ai.diagnostics.common import DiagnosticResult
from traceml_ai.diagnostics.step_time.api import (
    StepDiagnosis,
    build_step_diagnosis_result,
    build_step_warmup_diagnosis,
)
from traceml_ai.diagnostics.step_time.policy import (
    SUMMARY_STEP_TIME_POLICY,
    StepTimeDiagnosisPolicy,
)
from traceml_ai.loggers.error_log import get_error_logger
from traceml_ai.renderers.step_time.schema import StepCombinedTimeCoverage
from traceml_ai.utils.step_time_diagnosis_clock import (
    build_diagnosis_metrics_from_timing,
    build_diagnosis_timing_from_step_metrics,
)
from traceml_ai.utils.step_windows import common_suffix_steps

_LOGGER = get_error_logger("StepTimeDiagnostics")


DEFAULT_SUMMARY_DIAG_CONFIG = SUMMARY_STEP_TIME_POLICY

# Per-rank per-step canonical metrics map:
# rank -> step -> metric_key -> value_ms
RankStepMetricSeries = Dict[int, Dict[int, Dict[str, float]]]


@dataclass(frozen=True)
class StepTimeDiagnosisInput:
    """
    Aligned step-time data consumed by final-summary diagnosis.

    Diagnosis is intentionally based on aligned per-step metrics only.
    """

    per_rank_step_metrics: RankStepMetricSeries
    max_rows: int
    policy: StepTimeDiagnosisPolicy = DEFAULT_SUMMARY_DIAG_CONFIG


def _log_adapter_error(message: str, exc: Exception) -> None:
    """Log adapter failures without blocking final summary generation."""
    try:
        _LOGGER.exception("[TraceML] %s", message)
    except Exception:
        pass


def _finite_float(x: Any) -> float:
    """Convert to float; return 0.0 for non-finite or invalid values."""
    try:
        v = float(x)
    except Exception:
        return 0.0
    return v if math.isfinite(v) else 0.0


def build_summary_step_diagnosis_result(
    per_rank_step_metrics: RankStepMetricSeries,
    *,
    max_rows: int,
    policy: StepTimeDiagnosisPolicy = DEFAULT_SUMMARY_DIAG_CONFIG,
) -> DiagnosticResult[StepDiagnosis]:
    """
    Build rich summary-mode diagnosis from aligned per-step timing signals.

    Notes
    -----
    - Diagnosis uses the same aligned step window as the final summary.
    - No aggregate-only fallback is supported.
    """
    if not per_rank_step_metrics:
        return build_step_diagnosis_result([], thresholds=policy.thresholds)

    ranks = sorted(int(rank) for rank in per_rank_step_metrics.keys())
    try:
        common_steps = common_suffix_steps(
            per_rank_step_metrics,
            max_rows=max_rows,
        )
    except Exception as exc:
        _log_adapter_error("Step-time summary step alignment failed.", exc)
        common_steps = []

    if not common_steps:
        return build_step_diagnosis_result([], thresholds=policy.thresholds)

    steps_used = int(len(common_steps))
    if steps_used < int(policy.min_steps_for_diag):
        return build_step_warmup_diagnosis(
            steps_used=steps_used,
            required_steps=int(policy.min_steps_for_diag),
        )

    diagnosis_timing = build_diagnosis_timing_from_step_metrics(
        per_rank_step_metrics,
        common_steps,
        aggregate="average",
    )
    coverage = StepCombinedTimeCoverage(
        expected_steps=int(max_rows),
        steps_used=steps_used,
        completed_step=int(common_steps[-1]),
        world_size=len(ranks),
        ranks_present=len(ranks),
        incomplete=False,
    )

    overall_rank_scores = {
        r: _finite_float(
            diagnosis_timing.per_rank_timing.get(r, {}).get("total_step")
        )
        for r in ranks
    }
    overall_worst_rank = (
        int(max(ranks, key=lambda r: (overall_rank_scores[r], -r)))
        if ranks
        else None
    )
    metrics = build_diagnosis_metrics_from_timing(
        diagnosis_timing.per_rank_timing,
        coverage=coverage,
        include_series=True,
        series_steps=common_steps,
        per_rank_step_timing=diagnosis_timing.per_rank_step_timing,
        worst_rank_override=overall_worst_rank,
    )

    if not metrics:
        return build_step_diagnosis_result([], thresholds=policy.thresholds)

    return build_step_diagnosis_result(
        metrics,
        thresholds=policy.thresholds,
        per_rank_timing=diagnosis_timing.per_rank_timing,
        diagnosis_clock=diagnosis_timing.clock,
    )


def diagnose_step_time_summary(
    data: StepTimeDiagnosisInput,
) -> DiagnosticResult[StepDiagnosis]:
    """Run final-summary Step Time diagnosis from the typed input contract."""
    return build_summary_step_diagnosis_result(
        data.per_rank_step_metrics,
        max_rows=data.max_rows,
        policy=data.policy,
    )


__all__ = [
    "DEFAULT_SUMMARY_DIAG_CONFIG",
    "RankStepMetricSeries",
    "StepTimeDiagnosisInput",
    "build_summary_step_diagnosis_result",
    "diagnose_step_time_summary",
]
