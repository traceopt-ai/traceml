"""Shared SQLite loading for canonical Step Time windows.

Live CLI/dashboard and final-summary reporting both diagnose Step Time from
the same global-rank event rows. They differ only in how many rows they load:
live uses a small recent window with extra lookback, while summary uses a
larger final-report window.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from traceml_ai.utils.step_time_window import (
    StepTimeWindow,
    build_step_time_window_from_events,
)


@dataclass(frozen=True)
class StepTimeSQLiteWindow:
    """A Step Time window plus the global-rank ids used to build it."""

    window: StepTimeWindow
    global_ranks: tuple[int, ...]


def load_step_time_window_from_sqlite(
    conn: sqlite3.Connection,
    *,
    max_rows: int,
    lookback_factor: int = 1,
    table: str = "step_time_samples",
    rank_filter: Optional[Sequence[int]] = None,
) -> StepTimeSQLiteWindow:
    """
    Load one canonical selected-clock Step Time window from SQLite.

    The loader reads `global_rank` ids and returns them as rank-shaped data for
    existing diagnosis, CLI/dashboard, and summary contracts. `max_rows` is the
    final aligned window size; `lookback_factor` only controls how much recent
    per-rank history is read before common-step alignment.
    """
    row_limit = max(1, int(max_rows))
    lookback = max(row_limit * max(1, int(lookback_factor)), row_limit)
    allowed_ranks = (
        {int(rank) for rank in rank_filter}
        if rank_filter is not None
        else None
    )

    rows = conn.execute(
        f"""
        SELECT DISTINCT global_rank
        FROM {table}
        WHERE global_rank IS NOT NULL
        ORDER BY global_rank ASC;
        """
    ).fetchall()
    global_ranks = tuple(
        int(row[0])
        for row in rows
        if row[0] is not None
        and (allowed_ranks is None or int(row[0]) in allowed_ranks)
    )

    per_rank_steps: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for global_rank in global_ranks:
        step_rows = conn.execute(
            f"""
            SELECT step, events_json
            FROM (
                SELECT id, step, events_json
                FROM {table}
                WHERE global_rank = ?
                ORDER BY step DESC, id DESC
                LIMIT ?
            )
            ORDER BY step ASC, id DESC;
            """,
            (int(global_rank), int(lookback)),
        ).fetchall()

        step_map: Dict[int, Dict[str, Any]] = {}
        for step, events_json in step_rows:
            if step is None or not events_json:
                continue

            step_id = int(step)
            if step_id in step_map:
                continue

            try:
                parsed = json.loads(events_json)
            except Exception:
                continue

            if isinstance(parsed, dict):
                step_map[step_id] = parsed

        if step_map:
            per_rank_steps[int(global_rank)] = step_map

    return StepTimeSQLiteWindow(
        window=build_step_time_window_from_events(
            per_rank_steps,
            max_rows=row_limit,
            expected_ranks=global_ranks,
        ),
        global_ranks=global_ranks,
    )


__all__ = [
    "StepTimeSQLiteWindow",
    "load_step_time_window_from_sqlite",
]
