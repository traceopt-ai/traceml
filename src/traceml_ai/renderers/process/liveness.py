# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Which ranks are still reporting, read from the process heartbeat.

Every rank's sampler thread writes a ``process_samples`` row on its own
cadence, and keeps doing so while the training thread is blocked in a
collective waiting for a dead peer. That makes this table the per-rank
heartbeat. This module reads it once per tick and applies the shared
:class:`FreshnessPolicy`, so the terminal and the dashboard judge a rank
by one rule and one clock.

Moved out of ``dashboard_compute.py`` (issue #358): the dashboard already
judged rank freshness this way, and the terminal now reuses the same read
rather than a second rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from traceml_ai.renderers.shared.freshness import FreshnessPolicy, RankLiveness
from traceml_ai.renderers.shared.run_series import finite

from .repository import ProcessRepository

# The recent window the tiles describe, as a DURATION. A sample count
# means a different span at every sampling rate, so two runs sampling at
# different cadences could not be compared and the card could not say what
# period it was summarising. The same duration drives both the repository
# read and the recent-to-retained chart transition.
RECENT_WINDOW_S = 60.0


def opt_float(value: Any) -> Optional[float]:
    """A usable number from a database cell, or ``None``."""
    if value is None:
        return None
    try:
        return finite(float(value))
    except (TypeError, ValueError):
        return None


def _as_rank(value: Any) -> Optional[int]:
    """A rank number from a database cell, or ``None``."""
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _rank_id(row: Any) -> Optional[int]:
    """The row's rank, or ``None`` so a row without a usable one is skipped.

    ``global_rank`` first; ``rank`` when that cell is NULL or cannot be
    parsed. One malformed cell must cost its own row at most, never
    every rank's verdict.
    """
    rank_id = _as_rank(row["global_rank"])
    return rank_id if rank_id is not None else _as_rank(row["rank"])


def _arrival(row: Any) -> float:
    """When the aggregator received a row, oldest when unreadable."""
    recv = opt_float(row["recv_ts_ns"])
    return recv if recv is not None else float("-inf")


def _sample_order(row: Any) -> Tuple[float, int]:
    """The windowed read's own order within a rank: seq, then id."""
    seq = opt_float(row["seq"])
    return (seq if seq is not None else float("-inf"), int(row["id"]))


def observed_cadence(by_rank: Dict[int, List[Any]]) -> Optional[float]:
    """The gap the ranks actually sample at, from the busiest rank."""
    best: Optional[float] = None
    for rows in by_rank.values():
        stamps = [
            value
            for value in (opt_float(r["sample_ts_s"]) for r in rows)
            if value is not None
        ]
        if len(stamps) < 2:
            continue
        span = max(stamps) - min(stamps)
        cadence = finite(span / float(len(stamps) - 1))
        if cadence and cadence > 0:
            best = cadence if best is None else min(best, cadence)
    return best


def newest_recv(rows: Sequence[Any]) -> float:
    """The aggregator's newest arrival clock, the reference for age."""
    stamps = [
        value / 1e9
        for value in (opt_float(r["recv_ts_ns"]) for r in rows)
        if value is not None
    ]
    return max(stamps) if stamps else 0.0


@dataclass(frozen=True)
class RankClock:
    """One tick's per-rank rows, the policy, and the reference clock.

    ``by_rank`` is each rank's recent window; ``newest_by_rank`` is the
    newest row of every rank that has EVER reported, so a rank silent for
    longer than the window still appears with its true age.
    """

    policy: FreshnessPolicy
    now_s: float
    by_rank: Dict[int, List[Any]]
    newest_by_rank: Dict[int, Any]

    def liveness_of(self, rank_id: int) -> RankLiveness:
        """The last-seen clock and verdict for one rank."""
        recv = opt_float(self.newest_by_rank[rank_id]["recv_ts_ns"])
        last_seen = recv / 1e9 if recv is not None else None
        age = self.policy.age_of(last_seen, now_s=self.now_s)
        return RankLiveness(
            global_rank=rank_id,
            last_seen_s=last_seen,
            age_s=age,
            freshness=self.policy.state_of(age),
        )

    def liveness(self) -> Tuple[RankLiveness, ...]:
        """Every rank that has reported, in rank order."""
        return tuple(
            self.liveness_of(rank_id)
            for rank_id in sorted(self.newest_by_rank)
        )


def read_rank_clock(
    db: ProcessRepository,
    conn: Any,
    *,
    newest_ts: Optional[float],
    configured_interval_s: Optional[float],
    window_s: float = RECENT_WINDOW_S,
) -> RankClock:
    """Every rank's own state, read on its own clock.

    Two reads, on purpose. The windowed one carries each rank's recent
    history; the latest-row one carries every rank that has EVER
    reported. Age is measured against the newest arrival across ranks,
    so a rank is stale when its peers keep reporting and it does not.
    """
    window_rows = db.fetch_recent_rank_window(
        conn, window_s=window_s, newest_ts=newest_ts
    )
    latest_rows = db.fetch_rank_latest(conn)

    by_rank: Dict[int, List[Any]] = {}
    for row in window_rows:
        rank_id = _rank_id(row)
        if rank_id is None:
            continue
        by_rank.setdefault(rank_id, []).append(row)
    # A row that fell back to its ``rank`` cell arrives in a group of its
    # own, after that rank's rows. Put it where it was sampled; rows
    # already in the read's order stay as they are.
    for rows in by_rank.values():
        rows.sort(key=_sample_order)

    newest_by_rank: Dict[int, Any] = {}
    for row in latest_rows:
        rank_id = _rank_id(row)
        if rank_id is None:
            continue
        # A row that fell back to its ``rank`` cell is the newest of its
        # own group, not necessarily of that rank; the later arrival is
        # the rank's last word.
        held = newest_by_rank.get(rank_id)
        if held is None or _arrival(row) > _arrival(held):
            newest_by_rank[rank_id] = row

    policy = FreshnessPolicy.from_observed_cadence(
        observed_cadence(by_rank),
        configured_s=configured_interval_s,
    )
    return RankClock(
        policy=policy,
        now_s=newest_recv(list(newest_by_rank.values())),
        by_rank=by_rank,
        newest_by_rank=newest_by_rank,
    )


__all__ = [
    "RECENT_WINDOW_S",
    "RankClock",
    "newest_recv",
    "observed_cadence",
    "opt_float",
    "read_rank_clock",
]
