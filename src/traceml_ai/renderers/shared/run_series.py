# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""How a run-length series is sampled, shared by every telemetry domain.

A live chart shows the last N samples. Once a run outlives that window the
chart has to describe the WHOLE run instead, which means choosing a rolling
window, a stride, and a point budget. That machinery is identical for every
domain, so it lives here once and each repository executes its own SQL
against the plan this module produces.

What this module deliberately does not know: table names, column names,
metric definitions, or anything about a display framework. It plans; the
caller reads.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal, Optional

SeriesMode = Literal["recent", "retained"]
"""Which view a chart is showing.

``recent`` is the last N samples as they were taken. ``retained`` is the
whole run, rolled and decimated to fit a point budget. The payload states
which one it holds; a view must never infer it from which field happens to
be populated.
"""

# SQLite grew numeric RANGE frame offsets in 3.28. A time frame is the
# honest one: unreported samples leave holes in the cadence, so a ROW frame
# silently spans more wall clock than it claims. Older engines fall back to
# the row frame.
HAS_RANGE_FRAME = sqlite3.sqlite_version_info >= (3, 28, 0)


@dataclass(frozen=True)
class RunSeriesPolicy:
    """The knobs that decide how a whole-run series is shaped.

    Defaults are the values both the System and Process blocks arrived at
    independently, which is why they are worth sharing rather than copying
    a third time.
    """

    roll_min_s: float = 30.0
    roll_max_s: float = 300.0
    roll_fraction: float = 50.0  # about a fiftieth of the run
    max_points: int = 120
    # The run must outgrow the live window by this much before the chart
    # switches to the whole-run view, so a chart near the boundary does not
    # flip back and forth every tick.
    retained_factor: float = 1.2

    def window_for(self, span_s: float) -> float:
        """The rolling window for a run of ``span_s`` seconds, in round steps.

        Round steps rather than a raw fraction so the label a card prints
        ("rolling 2 min") stays a number a reader recognises.
        """
        if span_s <= 0:
            return self.roll_min_s
        raw = max(
            self.roll_min_s,
            min(self.roll_max_s, span_s / self.roll_fraction),
        )
        for step in (30.0, 60.0, 120.0, 300.0):
            if raw <= step:
                # The ladder is a display convenience, not an override: a
                # policy that caps the window below a rung must still be
                # honoured, or the chart aggregates over more time than the
                # caller allowed.
                return min(max(step, self.roll_min_s), self.roll_max_s)
        return self.roll_max_s

    def mode_for(self, run_span_s: float, window_span_s: float) -> SeriesMode:
        """Whether the chart should describe the window or the whole run."""
        if window_span_s <= 0:
            return "recent"
        if run_span_s > window_span_s * self.retained_factor:
            return "retained"
        return "recent"


DEFAULT_RUN_SERIES_POLICY = RunSeriesPolicy()


@dataclass(frozen=True)
class RunSeriesPlan:
    """Everything a repository needs to read one whole-run series.

    Carries no SQL of its own beyond the window frame, which is fully
    determined by the numbers here. The alternative, letting each domain
    build the frame itself, is what produced two copies of this logic in
    the first place.
    """

    window_s: float
    cadence_s: float
    stride: int
    max_points: int
    sample_count: int

    @property
    def preceding_rows(self) -> int:
        """Rows of history one rolling aggregate covers, for a ROW frame."""
        return max(
            1, int(round(self.window_s / max(self.cadence_s, 1e-6))) - 1
        )

    @property
    def eligible_count(self) -> int:
        """Samples that can carry a full window.

        A partial window at the head of a run averages fewer samples than it
        claims, so those rows are excluded before the stride is chosen. This
        is what keeps the point budget honest: dividing by the raw count
        overshoots it.
        """
        return max(0, self.sample_count - self.preceding_rows)

    def frame_clause(self) -> str:
        """The window frame for a rolling aggregate over ``window_s``."""
        if HAS_RANGE_FRAME:
            return (
                f"RANGE BETWEEN {float(self.window_s):.6f} "
                "PRECEDING AND CURRENT ROW"
            )
        return f"ROWS BETWEEN {self.preceding_rows} PRECEDING AND CURRENT ROW"


def stride_for(eligible_count: int, max_points: int) -> int:
    """Keep at most ``max_points`` samples, by ceiling division.

    Ceiling, not floor: a floor divides the run into more buckets than the
    budget allows and the chart ships more points than its own label
    promises.
    """
    if eligible_count <= 0 or max_points <= 0:
        return 1
    return max(1, (eligible_count + max_points - 1) // max_points)


def cadence_of(span_s: float, sample_count: int) -> Optional[float]:
    """The observed gap between samples, or ``None`` when unknowable.

    Measured rather than assumed: a configured sampler interval says what
    was asked for, and this says what arrived.
    """
    if sample_count < 2 or span_s <= 0:
        return None
    return span_s / float(sample_count - 1)


def plan_run_series(
    *,
    span_s: float,
    sample_count: int,
    cadence_s: Optional[float] = None,
    policy: RunSeriesPolicy = DEFAULT_RUN_SERIES_POLICY,
) -> Optional[RunSeriesPlan]:
    """Plan one whole-run read, or ``None`` when there is nothing to plan.

    ``cadence_s`` may be supplied by a caller that already knows it; when
    omitted it is measured from the span and the count.
    """
    if sample_count < 2 or span_s <= 0:
        return None
    observed = (
        cadence_s
        if cadence_s is not None
        else cadence_of(span_s, sample_count)
    )
    if not observed or observed <= 0:
        return None

    window_s = policy.window_for(span_s)
    plan = RunSeriesPlan(
        window_s=window_s,
        cadence_s=float(observed),
        stride=1,
        max_points=policy.max_points,
        sample_count=int(sample_count),
    )
    return RunSeriesPlan(
        window_s=window_s,
        cadence_s=float(observed),
        stride=stride_for(plan.eligible_count, policy.max_points),
        max_points=policy.max_points,
        sample_count=int(sample_count),
    )


__all__ = [
    "DEFAULT_RUN_SERIES_POLICY",
    "HAS_RANGE_FRAME",
    "RunSeriesPlan",
    "RunSeriesPolicy",
    "SeriesMode",
    "cadence_of",
    "plan_run_series",
    "stride_for",
]
