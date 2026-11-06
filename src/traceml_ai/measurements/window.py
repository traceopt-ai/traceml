# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Select the numeric interval analyzed by a final report."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional

from traceml_ai.telemetry.retention import (
    DEFAULT_HISTORY_RETENTION_S,
    parse_history_retention,
)


@dataclass(frozen=True, slots=True)
class AnalysisWindow:
    """Step-derived time bounds shared by measured report sections."""

    retention_s: float
    anchor: Optional[str] = None
    start_time_unix_ns: Optional[int] = None
    end_time_unix_ns: Optional[int] = None
    start_step: Optional[int] = None
    end_step: Optional[int] = None

    @property
    def duration_s(self) -> Optional[float]:
        if self.start_time_unix_ns is None or self.end_time_unix_ns is None:
            return None
        return max(0, self.end_time_unix_ns - self.start_time_unix_ns) / 1e9


def resolve_analysis_window(
    connection: sqlite3.Connection,
    *,
    retention_s: float = DEFAULT_HISTORY_RETENTION_S,
) -> AnalysisWindow:
    """Resolve one completed-step window, with a periodic-data fallback."""
    retention = parse_history_retention(retention_s)
    steps = _common_completed_steps(connection)
    if steps:
        end_step, end_time_unix_ns = steps[-1]
        cutoff = end_time_unix_ns - round(retention * 1_000_000_000)
        selected = [item for item in steps if item[1] >= cutoff]
        start_step, start_time_unix_ns = selected[0]
        return AnalysisWindow(
            retention_s=retention,
            anchor="step_time",
            start_time_unix_ns=start_time_unix_ns,
            end_time_unix_ns=end_time_unix_ns,
            start_step=start_step,
            end_step=end_step,
        )

    bounds = _periodic_bounds(connection, retention_s=retention)
    if bounds is None:
        return AnalysisWindow(retention_s=retention)
    return AnalysisWindow(
        retention_s=retention,
        anchor="periodic_telemetry",
        start_time_unix_ns=bounds[0],
        end_time_unix_ns=bounds[1],
    )


def _common_completed_steps(
    connection: sqlite3.Connection,
) -> list[tuple[int, int]]:
    """Return completed-step source times in Unix nanoseconds."""
    try:
        rows = connection.execute("""
            WITH deduplicated AS (
                SELECT
                    COALESCE(global_rank, rank) AS rank_id,
                    world_size,
                    step,
                    timestamp_unix_ns,
                    ROW_NUMBER() OVER (
                        PARTITION BY COALESCE(global_rank, rank), step
                        ORDER BY id DESC
                    ) AS duplicate_row
                FROM step_time_samples
                WHERE COALESCE(global_rank, rank) IS NOT NULL
                  AND step IS NOT NULL
                  AND timestamp_unix_ns IS NOT NULL
                  AND events_json IS NOT NULL
                  AND events_json != ''
            ),
            rank_count AS (
                SELECT COALESCE(
                    MAX(CASE WHEN world_size > 0 THEN world_size END),
                    COUNT(DISTINCT rank_id)
                ) AS value
                FROM deduplicated
                WHERE duplicate_row = 1
            )
            SELECT step, MAX(timestamp_unix_ns)
            FROM deduplicated, rank_count
            WHERE duplicate_row = 1
            GROUP BY step
            HAVING COUNT(DISTINCT rank_id) = rank_count.value
            ORDER BY step ASC;
            """).fetchall()
    except sqlite3.Error:
        return []
    return [(int(row[0]), int(row[1])) for row in rows]


def _periodic_bounds(
    connection: sqlite3.Connection,
    *,
    retention_s: float,
) -> Optional[tuple[int, int]]:
    """Return up-to-retention bounds from available System/Process samples."""
    values: list[int] = []
    for table in ("system_samples", "process_samples"):
        try:
            row = connection.execute(
                f"SELECT MIN(timestamp_unix_ns), MAX(timestamp_unix_ns) FROM {table};"
            ).fetchone()
        except sqlite3.Error:
            continue
        if row and row[0] is not None and row[1] is not None:
            values.extend(row)
    if not values:
        return None
    end_time_unix_ns = max(values)
    return (
        max(
            min(values), end_time_unix_ns - round(retention_s * 1_000_000_000)
        ),
        end_time_unix_ns,
    )


__all__ = ["AnalysisWindow", "resolve_analysis_window"]
