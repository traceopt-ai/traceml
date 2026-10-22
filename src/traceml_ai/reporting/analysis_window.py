# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Project the shared measurement interval into the public report."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from typing import Any

from traceml_ai.measurements.window import AnalysisWindow

_SECTION_TABLES = {
    "system": "system_samples",
    "process": "process_samples",
    "step_time": "step_time_samples",
    "step_memory": "step_memory_samples",
}
_STEP_SECTIONS = frozenset({"step_time", "step_memory"})


def analysis_window_metadata(window: AnalysisWindow) -> dict[str, Any]:
    """Return common fields embedded in every section metadata block."""
    return {
        "analysis_start_time_unix_ns": window.start_time_unix_ns,
        "analysis_end_time_unix_ns": window.end_time_unix_ns,
        "analysis_start_step": window.start_step,
        "analysis_end_step": window.end_step,
    }


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
        "start_time_unix_ns": window.start_time_unix_ns,
        "end_time_unix_ns": window.end_time_unix_ns,
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
    if window.start_time_unix_ns is None or window.end_time_unix_ns is None:
        return _empty_observation()

    if section in _STEP_SECTIONS and window.start_step is not None:
        where = "step BETWEEN ? AND ?"
        params: tuple[Any, ...] = (window.start_step, window.end_step)
    else:
        where = "timestamp_unix_ns BETWEEN ? AND ?"
        params = (window.start_time_unix_ns, window.end_time_unix_ns)

    try:
        row = conn.execute(
            f"""
            SELECT COUNT(*), MIN(timestamp_unix_ns), MAX(timestamp_unix_ns)
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
        "first_sample_time_unix_ns": (
            int(row[1]) if row and row[1] is not None else None
        ),
        "last_sample_time_unix_ns": (
            int(row[2]) if row and row[2] is not None else None
        ),
    }


def _empty_observation() -> dict[str, Any]:
    return {
        "samples": 0,
        "first_sample_time_unix_ns": None,
        "last_sample_time_unix_ns": None,
    }


__all__ = [
    "analysis_window_metadata",
    "build_analysis_window_payload",
]
