"""
Conservative trend heuristics for step-time diagnosis.

Design goals
------------
- Never break diagnosis flow.
- Trend should annotate, not dominate, the primary diagnosis.
- Work with the live renderer window instead of requiring very long history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from traceml.renderers.step_time.schema import StepCombinedTimeMetric

from .trends import (
    DEFAULT_TREND_CONFIG,
    TrendConfig,
    compute_trend_pct,
    format_trend_pct,
)


@dataclass(frozen=True)
class StepTrendHeuristicConfig:
    """
    Conservative gates for trend-based diagnosis annotations.
    """

    enabled: bool = True
    min_steps: int = 80
    deadband_pct: float = 0.03
    worsening_threshold_pct: float = 0.08
    improving_threshold_pct: float = 0.08
    near_warn_fraction: float = 0.90
    trend: TrendConfig = field(default_factory=lambda: DEFAULT_TREND_CONFIG)


DEFAULT_STEP_TREND_HEURISTICS = StepTrendHeuristicConfig()


def _safe_metric_trend_pct(
    metric: Optional[StepCombinedTimeMetric],
    *,
    single_rank: bool,
    cfg: StepTrendHeuristicConfig,
) -> Optional[float]:
    """Safely compute trend percentage for one metric."""
    try:
        if metric is None or metric.series is None:
            return None
        series = metric.series.worst if single_rank else metric.series.median
        if not series:
            return None
        return compute_trend_pct(series, config=cfg.trend)
    except Exception:
        return None


def _trend_state(
    pct: Optional[float],
    *,
    cfg: StepTrendHeuristicConfig,
) -> Optional[str]:
    """Map trend ratio to a coarse state."""
    if pct is None:
        return None
    if pct >= cfg.worsening_threshold_pct:
        return "worsening"
    if pct <= -cfg.improving_threshold_pct:
        return "improving"
    return None


def build_step_trend_note(
    *,
    diagnosis_kind: str,
    steps_used: int,
    single_rank: bool,
    step_metric: Optional[StepCombinedTimeMetric],
    wait_metric: Optional[StepCombinedTimeMetric],
    dataloader_metric: Optional[StepCombinedTimeMetric],
    wait_share: float,
    dataloader_share: float,
    wait_warn_threshold: float,
    input_warn_threshold: float,
    cfg: StepTrendHeuristicConfig = DEFAULT_STEP_TREND_HEURISTICS,
) -> Optional[str]:
    """
    Build one optional trend note for a step diagnosis.
    """
    try:
        if not cfg.enabled or steps_used < cfg.min_steps:
            return None

        step_tr = _safe_metric_trend_pct(
            step_metric, single_rank=single_rank, cfg=cfg
        )
        wait_tr = _safe_metric_trend_pct(
            wait_metric, single_rank=single_rank, cfg=cfg
        )
        dl_tr = _safe_metric_trend_pct(
            dataloader_metric, single_rank=single_rank, cfg=cfg
        )

        step_state = _trend_state(step_tr, cfg=cfg)
        wait_state = _trend_state(wait_tr, cfg=cfg)
        dl_state = _trend_state(dl_tr, cfg=cfg)

        if diagnosis_kind in {"INPUT_BOUND", "INPUT_STRAGGLER"} and dl_state:
            return (
                "Trend: dataloader is "
                f"{dl_state} ({format_trend_pct(dl_tr, deadband_pct=cfg.deadband_pct)})."
            )

        if (
            diagnosis_kind
            in {"COMPUTE_BOUND", "COMPUTE_STRAGGLER", "STRAGGLER"}
            and step_state
        ):
            return (
                "Trend: step time is "
                f"{step_state} ({format_trend_pct(step_tr, deadband_pct=cfg.deadband_pct)})."
            )

        if diagnosis_kind == "WAIT_HEAVY" and wait_state:
            return (
                "Trend: WAIT* is "
                f"{wait_state} ({format_trend_pct(wait_tr, deadband_pct=cfg.deadband_pct)})."
            )

        near_wait_warn = wait_share >= (
            wait_warn_threshold * cfg.near_warn_fraction
        )
        near_input_warn = dataloader_share >= (
            input_warn_threshold * cfg.near_warn_fraction
        )

        if (
            diagnosis_kind == "BALANCED"
            and step_state == "worsening"
            and (near_wait_warn or near_input_warn)
        ):
            return (
                "Trend: step time is rising "
                f"({format_trend_pct(step_tr, deadband_pct=cfg.deadband_pct)})."
            )

        return None
    except Exception:
        return None
