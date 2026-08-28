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

from dataclasses import dataclass, field
from typing import Optional, Tuple

from traceml_ai.renderers.shared.freshness import FreshnessState
from traceml_ai.renderers.shared.run_series import SeriesMode


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
class RankSnapshot:
    """One rank's own state, on its own clock.

    Ranks are read independently rather than aligned on a shared step: a
    rank that stops reporting must keep its own history instead of being
    squeezed out by livelier peers, and a rank that never reports must not
    shrink everyone else's window.

    The levels here are deliberately not all "the newest sample". CPU and
    the allocator's live bytes are sampled far slower than they move, so a
    single reading lands wherever the sawtooth happened to be; those carry
    a window median. Reserved memory and RSS carry both, because which rank
    is WORST is a judgement about its typical state while the number SHOWN
    should be what that rank last actually sent.
    """

    global_rank: int
    node_rank: Optional[int] = None
    gpu_index: Optional[int] = None

    cpu_capacity_percent: Optional[float] = None
    ram_used_bytes: Optional[float] = None
    ram_used_p50_bytes: Optional[float] = None
    ram_total_bytes: Optional[float] = None

    gpu_allocated_p50_bytes: Optional[float] = None
    gpu_reserved_bytes: Optional[float] = None
    gpu_reserved_p50_bytes: Optional[float] = None
    gpu_total_bytes: Optional[float] = None

    age_s: Optional[float] = None
    freshness: FreshnessState = "unknown"

    @property
    def gpu_reported(self) -> bool:
        """Whether this rank has ever sent a usable GPU reading."""
        return self.gpu_total_bytes is not None and self.gpu_total_bytes > 0


@dataclass(frozen=True)
class RankCoverage:
    """Who is reporting, and who has gone quiet.

    Stated rather than implied. A block that silently drops a dead rank
    forgets it a few minutes after it died, which is exactly when its
    death starts to matter.

    A rank sits in exactly one of three buckets. ``unknown`` is the one
    worth stating: it is a rank that sent data but no usable timestamp,
    so it is neither proven live nor proven dead. Folding it into either
    of the other two invents a fact the telemetry did not carry.
    """

    total: int = 0
    live: int = 0
    stale: int = 0
    unknown: int = 0

    @property
    def excluding_stale(self) -> bool:
        """Whether aggregates were computed over a subset of the ranks."""
        return self.live > 0 and self.stale > 0


@dataclass(frozen=True)
class MetricRollup:
    """Window statistics for one metric.

    ``p50`` and ``p95`` are optional because not every metric needs them to
    be described; a field that is absent says so rather than carrying a
    number that reads like a measurement.

    The per-rank rollups leave ``p95`` unset. They used to copy ``now``
    into it, which produced a statistic nobody computed: on a finished run
    that reported ``p95`` below ``p50``, a rollup contradicting itself.
    """

    now: float
    p95: Optional[float] = None
    p50: Optional[float] = None
    total: Optional[float] = None
    worst_rank: Optional[int] = None


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
class RankTrace:
    """One rank's own line, over the window or over the whole run."""

    global_rank: int
    timestamps: Tuple[float, ...] = ()
    values: Tuple[float, ...] = ()
    peaks: Tuple[float, ...] = ()


@dataclass(frozen=True)
class RankChart:
    """A per-rank chart, and which history it is made of.

    ``mode`` is stated, never inferred. The previous payload carried a
    window series and a whole-run series in two differently named fields
    and left the view to work out which one to draw from whichever
    happened to be non-empty, which is a rule no reader can see.
    """

    mode: SeriesMode = "recent"
    window_s: Optional[float] = None
    span_s: Optional[float] = None
    traces: Tuple[RankTrace, ...] = ()

    @property
    def is_retained(self) -> bool:
        return self.mode == "retained"


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

    # Per-rank facts. Empty on a payload built before this layer existed,
    # so a consumer of the older shape keeps working unchanged.
    ranks: Tuple[RankSnapshot, ...] = ()
    coverage: RankCoverage = field(default_factory=RankCoverage)

    # The per-rank rollups, as NEW fields. `cpu`, `ram` and `gpu` above
    # keep the meanings the card already reads: `cpu` is raw psutil
    # percent, `gpu` is allocated bytes. Repurposing them here would have
    # changed the card silently, which is PR 4's job to do openly.
    cpu_capacity: Optional[MetricRollup] = None
    rss_worst: Optional[MetricRollup] = None
    gpu_reserved: Optional[MetricRollup] = None
    gpu_allocated: Optional[MetricRollup] = None
    reserved_imbalance_percent: Optional[float] = None
    cpu_capacity_chart: Optional[RankChart] = None
    rss_chart: Optional[RankChart] = None
    rows_open: bool = False

    @property
    def has_data(self) -> bool:
        """Whether the card has anything to draw.

        Ranks count as well as steps. The aggregated history needs a step
        committed across every rank, and the per-rank reads are populated
        before that happens, so asking only the history calls a live run
        empty for its first ticks.
        """
        return bool(self.history) or bool(self.ranks)

    @property
    def gpu_available(self) -> bool:
        """Whether this run has a GPU to talk about.

        Asked of the ranks first, and of the step history only as a
        fallback. An earlier version asked the newest committed step, on
        the reasoning that a run losing its GPU reporting should stop
        claiming one. Teardown makes that unworkable: the last samples of
        every run land after torch has released the device, so the newest
        step carries no GPU snapshot and a finished four-GPU run rendered
        as CPU-only, blanking two tiles while the same card printed a
        reserved-memory spread derived from the CUDA bytes it had just
        denied having.

        A rank's ``gpu_reported`` is read from its newest REPORTING row
        for that same reason, so it stays true through teardown.
        """
        if any(rank.gpu_reported for rank in self.ranks):
            return True
        return any(entry.gpu is not None for entry in self.history)

    @property
    def live_ranks(self) -> Tuple[RankSnapshot, ...]:
        """Non-stale ranks, including unknown ages, used by aggregates."""
        return tuple(rank for rank in self.ranks if rank.freshness != "stale")


__all__ = [
    "ChartSeries",
    "ChartTrace",
    "GpuSnapshot",
    "MetricRollup",
    "ProcessDashboardPayload",
    "ProcessHistoryEntry",
    "RankChart",
    "RankCoverage",
    "RankSnapshot",
    "RankTrace",
]
