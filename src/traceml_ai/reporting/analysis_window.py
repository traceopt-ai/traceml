# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Resolve and describe one shared final-report analysis interval."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from typing import Any, Optional

from traceml_ai.telemetry.legacy_source_time import legacy_source_time_storage
from traceml_ai.telemetry.retention import (
    DEFAULT_HISTORY_RETENTION_S,
    parse_history_retention,
)

_SECTION_TABLES = {
    "system": "system_samples",
    "process": "process_samples",
    "step_time": "step_time_samples",
    "step_memory": "step_memory_samples",
}
_STEP_SECTIONS = frozenset({"step_time", "step_memory"})


@dataclass(frozen=True, slots=True)
class AnalysisWindow:
    """Step-derived time bounds shared by every final-summary section."""

    retention_s: float
    anchor: Optional[str] = None
    start_ts_s: Optional[float] = None
    end_ts_s: Optional[float] = None
    start_step: Optional[int] = None
    end_step: Optional[int] = None

    @property
    def duration_s(self) -> Optional[float]:
        if self.start_ts_s is None or self.end_ts_s is None:
            return None
        return max(0.0, self.end_ts_s - self.start_ts_s)

    def metadata(self) -> dict[str, Any]:
        """Return common fields embedded in every section metadata block."""
        return {
            "analysis_start_ts_s": self.start_ts_s,
            "analysis_end_ts_s": self.end_ts_s,
            "analysis_start_step": self.start_step,
            "analysis_end_step": self.end_step,
        }


def resolve_analysis_window(
    db_path: str,
    *,
    retention_s: float = DEFAULT_HISTORY_RETENTION_S,
) -> AnalysisWindow:
    """Resolve one completed-step window, with a periodic-data fallback."""
    retention = parse_history_retention(retention_s)
    with closing(sqlite3.connect(db_path)) as conn:
        steps = _common_completed_steps(conn)
        if steps:
            end_step, end_ts = steps[-1]
            cutoff = end_ts - retention
            selected = [item for item in steps if item[1] >= cutoff]
            start_step, start_ts = selected[0]
            return AnalysisWindow(
                retention_s=retention,
                anchor="step_time",
                start_ts_s=start_ts,
                end_ts_s=end_ts,
                start_step=start_step,
                end_step=end_step,
            )

        bounds = _periodic_bounds(conn, retention_s=retention)
        if bounds is None:
            return AnalysisWindow(retention_s=retention)
        return AnalysisWindow(
            retention_s=retention,
            anchor="periodic_telemetry",
            start_ts_s=bounds[0],
            end_ts_s=bounds[1],
        )


def _common_completed_steps(
    conn: sqlite3.Connection,
) -> list[tuple[int, float]]:
    """Return completed-step source times in seconds for the public window."""
    try:
        rows = conn.execute("""
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
    # LEGACY SOURCE TIME: public seconds bounds until RFC-0011 PR 5.
    return [(int(row[0]), row[1] / 1_000_000_000) for row in rows]


def _periodic_bounds(
    conn: sqlite3.Connection,
    *,
    retention_s: float,
) -> Optional[tuple[float, float]]:
    """Return up-to-retention bounds from available System/Process samples."""
    values: list[float] = []
    for table in ("system_samples", "process_samples"):
        column, units_per_s = legacy_source_time_storage(table)
        try:
            row = conn.execute(
                f"SELECT MIN({column}), MAX({column}) FROM {table};"
            ).fetchone()
        except sqlite3.Error:
            continue
        if row and row[0] is not None and row[1] is not None:
            values.extend((row[0] / units_per_s, row[1] / units_per_s))
    if not values:
        return None
    end_ts = max(values)
    return max(min(values), end_ts - retention_s), end_ts


def build_analysis_window_payload(
    db_path: str,
    window: AnalysisWindow,
) -> dict[str, Any]:
    """Build public window metadata and per-section observations."""
    with closing(sqlite3.connect(db_path)) as conn:
        sections = {
            section: _section_observation(conn, section, table, window)
            for section, table in _SECTION_TABLES.items()
        }
    return {
        "anchor": window.anchor,
        "retention_s": window.retention_s,
        "start_ts_s": window.start_ts_s,
        "end_ts_s": window.end_ts_s,
        "duration_s": window.duration_s,
        "start_step": window.start_step,
        "end_step": window.end_step,
        "sections": sections,
    }


def _section_observation(
    conn: sqlite3.Connection,
    section: str,
    table: str,
    window: AnalysisWindow,
) -> dict[str, Any]:
    if window.start_ts_s is None or window.end_ts_s is None:
        return _empty_observation()

    column, units_per_s = legacy_source_time_storage(table)
    if section in _STEP_SECTIONS and window.start_step is not None:
        where = "step BETWEEN ? AND ?"
        params: tuple[Any, ...] = (window.start_step, window.end_step)
    else:
        where = f"{column} BETWEEN ? AND ?"
        # LEGACY SOURCE TIME: public seconds bounds until RFC-0011 PR 5.
        params = tuple(
            round(value * units_per_s) if units_per_s != 1 else value
            for value in (window.start_ts_s, window.end_ts_s)
        )

    try:
        row = conn.execute(
            f"""
            SELECT COUNT(*), MIN({column}), MAX({column})
            FROM {table}
            WHERE {where};
            """,
            params,
        ).fetchone()
    except sqlite3.Error:
        return _empty_observation()

    samples = int(row[0] or 0) if row else 0
    return {
        "samples": samples,
        "observed_start_ts_s": (
            row[1] / units_per_s if row and row[1] is not None else None
        ),
        "observed_end_ts_s": (
            row[2] / units_per_s if row and row[2] is not None else None
        ),
    }


def _empty_observation() -> dict[str, Any]:
    return {
        "samples": 0,
        "observed_start_ts_s": None,
        "observed_end_ts_s": None,
    }


__all__ = [
    "AnalysisWindow",
    "build_analysis_window_payload",
    "resolve_analysis_window",
]
