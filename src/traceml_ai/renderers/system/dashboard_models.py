# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""The contract between System compute and the System card.

The payload was four keys, two of which were `Dict[str, Any]`. Everything
inside them was reachable only by string, so the card asked questions like
"is `mem_total` missing?" to work out facts the compute layer already knew,
and answered some of them differently. Naming the fields is what stops
that: a question with one answer has one place to ask it.

These types are the whole contract. There is no ``to_dict`` alongside
them: a second rendering of the same payload is a second place for the
shape to drift, and the card, the driver and the tests all read the types.
``from_dict`` remains because the compute layer still assembles mappings
internally, and adapting them in one place is what keeps that private.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from traceml_ai.renderers.shared.run_series import SeriesMode


def _mode(value: Any) -> SeriesMode:
    """A series mode, defaulting to the view that needs no history."""
    return "retained" if value == "retained" else "recent"


@dataclass(frozen=True)
class Stat:
    """A level with the percentiles that say whether it is typical."""

    now: Optional[float] = None
    p50: Optional[float] = None
    p95: Optional[float] = None


@dataclass(frozen=True)
class RamStat:
    """Host memory, carrying the capacity that makes the level readable.

    Every field is optional because every one of them can be genuinely
    unread: a window where no sample carried a level, or a row with no
    capacity. 0.0 is a level a machine can be at, so it cannot also mean
    "not measured".
    """

    now: Optional[float] = None
    p95: Optional[float] = None
    total: Optional[float] = None
    headroom: Optional[float] = None


@dataclass(frozen=True)
class GpuMemStat:
    """Device memory on the GPU holding the most, with its own capacity.

    ``total`` is the capacity of the GPU reported in ``now``, not a sum
    across devices, so the tile can read "used of total" about one card
    rather than about an imaginary merged one.
    """

    now: Optional[float] = None
    p95: Optional[float] = None
    headroom: Optional[float] = None
    total: Optional[float] = None


@dataclass(frozen=True)
class TempStat:
    """Temperature, and the engine's verdict on it.

    ``status`` is the diagnosis engine's word. The card prints it and does
    not restate it in its own vocabulary.
    """

    now: Optional[float] = None
    p95: Optional[float] = None
    status: Optional[str] = None


@dataclass(frozen=True)
class PowerStat:
    """Board power, always the busiest GPU rather than a sum.

    ``floor`` is the lowest reading over the whole run when that history
    was aggregated, and over the recent window when it was not.
    """

    now: Optional[float] = None
    p50: Optional[float] = None
    limit: Optional[float] = None
    floor: Optional[float] = None


@dataclass(frozen=True)
class GpuRow:
    """One GPU's newest values, or its slot with nothing in it.

    A GPU that vanishes from the newest tick keeps its row with ``None``
    values rather than disappearing, so the row count never silently
    drops.

    ``reported`` is the compute layer's answer to "has this device told us
    anything", and it is on the row so the card never has to infer it. The
    card used to infer it, with a different rule, and the two disagreed
    about a GPU that reported a power limit but no memory total.
    """

    gpu_idx: int
    util_now: Optional[float] = None
    util_p50: Optional[float] = None
    mem_used: Optional[float] = None
    mem_total: Optional[float] = None
    temp: Optional[float] = None
    power: Optional[float] = None
    power_limit: Optional[float] = None
    reported: bool = False


@dataclass(frozen=True)
class RunContext:
    """Which machine and which slice of the job this payload describes."""

    world_size: int = 0
    gpu_count: int = 0
    hostname: str = ""
    # The host this payload describes: which node was picked, how many
    # were seen in the window, and its name. A mapping rather than an
    # index because the card needs all three to say "node 0 of 2".
    system_node: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class CpuRunSeries:
    """Host CPU over the whole run, rolled and decimated to fit a chart.

    Whether this series has earned the whole-run view is NOT asked here.
    ``ProcessDashboardComputer``'s policy decides it for both charts and
    the answer travels as ``SystemSeries.cpu_run_mode``. A second copy of
    that rule on this class is how the card came to hold two of them.
    """

    t: Tuple[float, ...] = ()
    avg: Tuple[float, ...] = ()
    max: Tuple[float, ...] = ()
    span_s: float = 0.0
    window_s: float = 0.0


@dataclass(frozen=True)
class SystemRollups:
    """Every level the card shows, already decided."""

    cpu: Optional[Stat] = None
    ram: Optional[RamStat] = None
    gpu_util: Optional[Stat] = None
    gpu_delta: Optional[Stat] = None
    gpu_mem: Optional[GpuMemStat] = None
    temp: Optional[TempStat] = None
    gpu_power: Optional[PowerStat] = None
    gpus: Tuple[GpuRow, ...] = ()
    ctx: Optional[RunContext] = None
    status: Optional[str] = None
    # GPUs whose utilisation puts them in the smaller group. Decided here
    # rather than in the card, which is where a rule that selects entities
    # by a derived threshold belongs.
    odd_gpus: Tuple[int, ...] = ()
    # The lowest and highest representative utilisation, or None. Named so
    # there is one definition of "the spread across GPUs" rather than this
    # one and gpu_delta both living unnamed on the same card.
    util_range: Optional[Tuple[float, float]] = None
    # Whether the spread has earned opening the per-GPU rows. A threshold
    # against a measurement is a severity call, so it is decided here.
    rows_over: bool = False
    # Util readings in the newest tick. This gates the tile's unavailable
    # state; it is not presented as coverage for the window median. A
    # device with a NULL util column is absent from this number even when
    # its other metrics reported. None preserves payload compatibility;
    # zero explicitly means no current util reading.
    util_gpu_count: Optional[int] = None

    @property
    def gpus_unreported(self) -> bool:
        """Whether every GPU present is represented by an unavailable row.

        Current sampling failures contain unavailable values; older traces
        may contain all-zero placeholders. Both represent absence rather
        than a GPU sitting at zero, so the card must not draw either as a
        measurement.

        Reads the computer's ``reported`` flag rather than re-deriving the
        answer. The card used to test whether ``mem_total`` and ``power``
        were both absent, a stricter rule than the computer's (``mem_total``
        or ``power_limit_w`` present), so the two disagreed about a GPU
        carrying a power limit and no memory total. This is the only place
        the question is answered.
        """
        return bool(self.gpus) and not any(g.reported for g in self.gpus)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SystemRollups":
        """Adapt the compute layer's rollup mapping."""
        if not raw:
            return cls()

        def stat(key: str) -> Optional[Stat]:
            block = raw.get(key)
            if not block:
                return None
            return Stat(
                now=block.get("now"),
                p50=block.get("p50"),
                p95=block.get("p95"),
            )

        ram = raw.get("ram") or None
        mem = raw.get("gpu_mem") or None
        temp = raw.get("temp") or None
        power = raw.get("gpu_power") or None
        ctx = raw.get("ctx") or None
        span = raw.get("util_range")
        return cls(
            cpu=stat("cpu"),
            gpu_util=stat("gpu_util"),
            gpu_delta=stat("gpu_delta"),
            ram=(
                RamStat(
                    now=ram.get("now"),
                    p95=ram.get("p95"),
                    total=ram.get("total"),
                    headroom=ram.get("headroom"),
                )
                if ram
                else None
            ),
            gpu_mem=(
                GpuMemStat(
                    now=mem.get("now"),
                    p95=mem.get("p95"),
                    headroom=mem.get("headroom"),
                    total=mem.get("total"),
                )
                if mem
                else None
            ),
            temp=(
                TempStat(
                    now=temp.get("now"),
                    p95=temp.get("p95"),
                    status=temp.get("status"),
                )
                if temp
                else None
            ),
            gpu_power=(
                PowerStat(
                    now=power.get("now"),
                    p50=power.get("p50"),
                    limit=power.get("limit"),
                    floor=power.get("floor"),
                )
                if power
                else None
            ),
            gpus=gpu_rows_from_dicts(raw.get("gpus") or []),
            ctx=(
                RunContext(
                    world_size=int(ctx.get("world_size") or 0),
                    gpu_count=int(ctx.get("gpu_count") or 0),
                    hostname=str(ctx.get("hostname") or ""),
                    system_node=ctx.get("system_node"),
                )
                if ctx
                else None
            ),
            status=raw.get("status"),
            odd_gpus=tuple(raw.get("odd_gpus") or ()),
            util_range=(tuple(span) if span else None),
            rows_over=bool(raw.get("rows_over")),
            util_gpu_count=(
                int(raw["util_gpu_count"])
                if raw.get("util_gpu_count") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class SystemSeries:
    """The lines the card draws, over the window and over the run."""

    x_time: Tuple[str, ...] = ()
    cpu: Tuple[Optional[float], ...] = ()
    gpu_avg: Tuple[Optional[float], ...] = ()
    # Per-GPU power over the window: one entry per device, each
    # {"gpu_idx", "values"}. Not a flat series, which is what the
    # annotation used to claim.
    gpu_power: Tuple[Dict[str, Any], ...] = ()
    cpu_run: CpuRunSeries = field(default_factory=CpuRunSeries)
    gpu_power_run: Tuple[Dict[str, Any], ...] = ()
    # Which view each chart is in. Both answered by one rule in the
    # compute layer; the card used to answer them itself with two
    # different rules and its charts could disagree.
    cpu_run_mode: SeriesMode = "recent"
    power_run_mode: SeriesMode = "recent"

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SystemSeries":
        """Adapt the compute layer's series mapping."""
        run = raw.get("cpu_run") or {}
        return cls(
            x_time=tuple(raw.get("x_time") or ()),
            cpu=tuple(raw.get("cpu") or ()),
            gpu_avg=tuple(raw.get("gpu_avg") or ()),
            gpu_power=tuple(raw.get("gpu_power") or ()),
            cpu_run=CpuRunSeries(
                t=tuple(run.get("t") or ()),
                avg=tuple(run.get("avg") or ()),
                max=tuple(run.get("max") or ()),
                span_s=float(run.get("span_s") or 0.0),
                window_s=float(run.get("window_s") or 0.0),
            ),
            gpu_power_run=tuple(raw.get("gpu_power_run") or ()),
            cpu_run_mode=_mode(raw.get("cpu_run_mode")),
            power_run_mode=_mode(raw.get("power_run_mode")),
        )


@dataclass(frozen=True)
class SystemDashboardPayload:
    """Everything the System card needs, and nothing it has to derive."""

    window_len: int = 0
    gpu_available: bool = False
    rollups: SystemRollups = field(default_factory=SystemRollups)
    series: SystemSeries = field(default_factory=SystemSeries)


__all__ = [
    "CpuRunSeries",
    "GpuMemStat",
    "GpuRow",
    "PowerStat",
    "RamStat",
    "RunContext",
    "Stat",
    "SystemDashboardPayload",
    "SystemRollups",
    "SystemSeries",
    "TempStat",
]


def gpu_rows_from_dicts(rows: List[Dict[str, Any]]) -> Tuple[GpuRow, ...]:
    """Adapt the compute layer's row dicts while it still emits them."""
    return tuple(
        GpuRow(
            gpu_idx=int(r.get("gpu_idx", 0)),
            util_now=r.get("util_now"),
            util_p50=r.get("util_p50"),
            mem_used=r.get("mem_used"),
            mem_total=r.get("mem_total"),
            temp=r.get("temp"),
            power=r.get("power"),
            power_limit=r.get("power_limit"),
            reported=bool(r.get("reported", False)),
        )
        for r in rows
    )
