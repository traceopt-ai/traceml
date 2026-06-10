# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Canonical Step Time names and legacy aliases."""

from __future__ import annotations

OVERHEAD_HEAVY_KIND = "OVERHEAD_HEAVY"
LEGACY_WAIT_HEAVY_KIND = "WAIT_HEAVY"

STEP_OVERHEAD_METRIC_KEY = "step_overhead_proxy"
LEGACY_WAIT_METRIC_KEY = "wait_proxy"

STEP_OVERHEAD_PUBLIC_METRIC = "step_overhead_ms"
LEGACY_WAIT_PUBLIC_METRIC = "wait_ms"

STEP_OVERHEAD_PHASE = "step_overhead"
LEGACY_WAIT_PHASE = "wait"


def canonical_issue_kind(kind: str) -> str:
    """Return the canonical issue/diagnosis kind for legacy inputs."""
    return OVERHEAD_HEAVY_KIND if kind == LEGACY_WAIT_HEAVY_KIND else kind


def canonical_step_time_metric_key(metric_key: str) -> str:
    """Return the canonical internal Step Time metric key."""
    if metric_key == LEGACY_WAIT_METRIC_KEY:
        return STEP_OVERHEAD_METRIC_KEY
    return metric_key


def canonical_step_time_public_metric(metric_key: str) -> str:
    """Return the canonical final-summary Step Time metric name."""
    if metric_key == LEGACY_WAIT_PUBLIC_METRIC:
        return STEP_OVERHEAD_PUBLIC_METRIC
    return metric_key


def public_metric_aliases(metric_key: str) -> tuple[str, ...]:
    """Return accepted public summary metric names for one canonical metric."""
    if metric_key == STEP_OVERHEAD_PUBLIC_METRIC:
        return (STEP_OVERHEAD_PUBLIC_METRIC, LEGACY_WAIT_PUBLIC_METRIC)
    return (metric_key,)


__all__ = [
    "LEGACY_WAIT_HEAVY_KIND",
    "LEGACY_WAIT_METRIC_KEY",
    "LEGACY_WAIT_PHASE",
    "LEGACY_WAIT_PUBLIC_METRIC",
    "OVERHEAD_HEAVY_KIND",
    "STEP_OVERHEAD_METRIC_KEY",
    "STEP_OVERHEAD_PHASE",
    "STEP_OVERHEAD_PUBLIC_METRIC",
    "canonical_issue_kind",
    "canonical_step_time_metric_key",
    "canonical_step_time_public_metric",
    "public_metric_aliases",
]
