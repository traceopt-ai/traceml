"""
Backward-compatible trend helpers.

Trend computation now lives under ``traceml.analytics.trends`` so renderers,
diagnostics, summaries, and compare paths can share one canonical definition.
"""

from traceml.analytics.trends import (
    DEFAULT_TREND_CONFIG,
    TrendComputationConfig,
    TrendEvidence,
    compute_trend_evidence,
    compute_trend_pct,
    format_trend_pct,
)

TrendConfig = TrendComputationConfig

__all__ = [
    "TrendConfig",
    "TrendEvidence",
    "DEFAULT_TREND_CONFIG",
    "compute_trend_evidence",
    "compute_trend_pct",
    "format_trend_pct",
]
