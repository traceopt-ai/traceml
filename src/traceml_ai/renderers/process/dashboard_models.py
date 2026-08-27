# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""The contract between Process compute and the Process card.

Everything the card draws arrives as one of these. They are explicit on
purpose: the previous payload was a list of dictionaries whose GPU keys
were present only when a GPU had reported, so the view had to ask "is this
key here?" to find out whether the run had a GPU at all. Absence is now
a typed ``None`` on a named field, which cannot be confused with a zero
and cannot be answered by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class GpuSnapshot:
    """One step's GPU memory picture, from the rank with least headroom.

    The rank is chosen by headroom rather than by usage: the process at
    risk is the one closest to filling its card, not the one holding the
    most bytes. ``used_imbalance_bytes`` is the spread across every
    reporting rank, so it describes the cohort while the rest of this
    object describes one rank.
    """

    used_bytes: float
    total_bytes: float
    headroom_bytes: float
    rank: Optional[int]
    used_imbalance_bytes: float


@dataclass(frozen=True)
class ProcessHistoryEntry:
    """One globally committed step, aggregated across ranks.

    ``gpu`` is ``None`` for a CPU-only run. That is the single place the
    card should ask whether there is a GPU to talk about.
    """

    seq: int
    ts: Optional[float]
    cpu_percent_max: float
    ram_used_bytes_max: float
    ram_total_bytes: float
    gpu: Optional[GpuSnapshot] = None


@dataclass(frozen=True)
class MetricRollup:
    """Window statistics for one metric.

    ``p50`` is optional because not every metric needs a median to be
    described; a field that is absent says so rather than carrying a zero
    that reads like a measurement.
    """

    now: float
    p95: float
    p50: Optional[float] = None
    total: Optional[float] = None


@dataclass(frozen=True)
class ChartTrace:
    """One line, as values against the timestamps they were sampled at.

    Points carry epoch seconds, not a formatted label: turning a moment
    into an axis tick is the view's job, and a compute layer that emits
    display strings has already crossed the boundary.
    """

    label: str
    timestamps: Tuple[Optional[float], ...]
    values: Tuple[float, ...]


@dataclass(frozen=True)
class ChartSeries:
    """The traces the card draws, already scaled to what they mean.

    Both are percentages of their own denominator, which is a decision
    about the metric rather than about the drawing, so it is made here.
    """

    ram_percent: ChartTrace
    gpu_percent: Optional[ChartTrace] = None


@dataclass(frozen=True)
class ProcessDashboardPayload:
    """Everything the Process card needs, and nothing it has to derive."""

    history: Tuple[ProcessHistoryEntry, ...] = ()
    window_len: int = 0
    cpu: Optional[MetricRollup] = None
    ram: Optional[MetricRollup] = None
    gpu: Optional[MetricRollup] = None
    gpu_used_imbalance_bytes: Optional[float] = None
    chart: Optional[ChartSeries] = None

    @property
    def has_data(self) -> bool:
        """Whether the card has anything to draw."""
        return bool(self.history)

    @property
    def gpu_available(self) -> bool:
        """Whether the newest step reported a GPU.

        Follows the newest entry rather than any entry, so a run that loses
        its GPU reporting stops claiming one.
        """
        return bool(self.history) and self.history[-1].gpu is not None


__all__ = [
    "ChartSeries",
    "ChartTrace",
    "GpuSnapshot",
    "MetricRollup",
    "ProcessDashboardPayload",
    "ProcessHistoryEntry",
]
