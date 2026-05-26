# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Aligned step-window helpers for Step Time final reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from traceml_ai.reporting.sections.step_time.model import (
    RankStepSummary,
    finite_float,
)
from traceml_ai.utils.step_windows import common_suffix_steps


@dataclass(frozen=True)
class AlignedStepWindow:
    """Common step window used for fair global-rank comparisons."""

    alignment: str
    steps_analyzed: int
    start_step: Optional[int]
    end_step: Optional[int]
    window_size: int
    global_ranks_used: int
    global_ranks_observed: int

    def to_json(self) -> Dict[str, Any]:
        """Return a stable JSON block for the aligned analysis window."""
        return {
            "alignment": self.alignment,
            "aligned_steps_analyzed": int(self.steps_analyzed),
            "start_step": self.start_step,
            "end_step": self.end_step,
            "window_size": int(self.window_size),
        }


def _summary_from_step_metrics(
    step_metrics: Dict[int, Dict[str, float]],
) -> Optional[RankStepSummary]:
    """Build a rank summary from already-normalized per-step metrics."""
    if not step_metrics:
        return None

    sum_dl = 0.0
    sum_fwd = 0.0
    sum_h2d = 0.0
    sum_bwd = 0.0
    sum_opt = 0.0
    sum_step_cpu = 0.0
    sum_traced_step = 0.0
    sum_total = 0.0
    n = 0

    for metrics in step_metrics.values():
        dataloader = finite_float(metrics.get("dataloader_fetch"))
        h2d = finite_float(metrics.get("h2d"))
        forward = finite_float(metrics.get("forward"))
        backward = finite_float(metrics.get("backward"))
        optimizer = finite_float(metrics.get("optimizer_step"))
        step_time = finite_float(metrics.get("step_time"))
        compute = forward + backward + optimizer
        known_step = h2d + compute

        sum_dl += dataloader
        sum_h2d += h2d
        sum_fwd += forward
        sum_bwd += backward
        sum_opt += optimizer
        sum_step_cpu += max(0.0, step_time)
        traced_step = max(step_time, known_step)
        sum_traced_step += traced_step
        sum_total += dataloader + traced_step
        n += 1

    if n == 0:
        return None

    return RankStepSummary(
        steps_analyzed=n,
        avg_dataloader_ms=sum_dl / n,
        avg_h2d_ms=sum_h2d / n,
        avg_forward_ms=sum_fwd / n,
        avg_backward_ms=sum_bwd / n,
        avg_optimizer_ms=sum_opt / n,
        avg_step_cpu_ms=sum_step_cpu / n,
        avg_traced_step_ms=sum_traced_step / n,
        avg_gpu_compute_ms=(sum_fwd + sum_bwd + sum_opt) / n,
        avg_total_step_ms=sum_total / n,
    )


def build_aligned_step_summary(
    *,
    per_global_rank_step_metrics: Dict[int, Dict[int, Dict[str, float]]],
    max_rows: int,
) -> tuple[
    Dict[int, RankStepSummary],
    Dict[int, Dict[int, Dict[str, float]]],
    AlignedStepWindow,
]:
    """
    Build fair global-rank summaries from the latest common step window.

    Global median/worst/skew compares the same completed steps across all
    observed ranks. Per-rank detail can still use its own latest window.
    """
    observed = len(per_global_rank_step_metrics)
    window_size = max(1, int(max_rows))
    common_steps = common_suffix_steps(per_global_rank_step_metrics, max_rows)
    if not common_steps:
        return (
            {},
            {},
            AlignedStepWindow(
                alignment="common_steps",
                steps_analyzed=0,
                start_step=None,
                end_step=None,
                window_size=window_size,
                global_ranks_used=0,
                global_ranks_observed=observed,
            ),
        )

    aligned_metrics: Dict[int, Dict[int, Dict[str, float]]] = {}
    aligned_summary: Dict[int, RankStepSummary] = {}
    common_step_set = set(common_steps)

    for global_rank, step_map in per_global_rank_step_metrics.items():
        rank_metrics = {
            int(step): metrics
            for step, metrics in step_map.items()
            if int(step) in common_step_set
        }
        summary = _summary_from_step_metrics(rank_metrics)
        if summary is None:
            continue
        aligned_metrics[int(global_rank)] = dict(sorted(rank_metrics.items()))
        aligned_summary[int(global_rank)] = summary

    return (
        aligned_summary,
        aligned_metrics,
        AlignedStepWindow(
            alignment="common_steps",
            steps_analyzed=len(common_steps),
            start_step=int(common_steps[0]),
            end_step=int(common_steps[-1]),
            window_size=window_size,
            global_ranks_used=len(aligned_summary),
            global_ranks_observed=observed,
        ),
    )


__all__ = [
    "AlignedStepWindow",
    "build_aligned_step_summary",
]
