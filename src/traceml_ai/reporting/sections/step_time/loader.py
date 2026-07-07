# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""SQLite loader for the final-report step-time section."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from traceml_ai.reporting.config import normalize_summary_window_rows
from traceml_ai.reporting.sections.step_time.model import (
    MAX_SUMMARY_WINDOW_ROWS,
    GlobalRankIdentity,
    RankStepSummary,
    rank_summaries_from_window,
)
from traceml_ai.utils.step_time_window import (
    StepTimeWindow,
    build_step_time_window_from_events,
)


@dataclass(frozen=True)
class StepTimeSectionData:
    """Loaded inputs for the step-time final-report section."""

    training_steps: int
    latest_step_observed: Optional[int]
    step_time_window: StepTimeWindow
    aligned_summary: Dict[int, RankStepSummary]
    aligned_step_metrics: Dict[int, Dict[int, Dict[str, float]]]
    aligned_window: StepTimeWindow
    per_global_rank_summary: Dict[int, RankStepSummary]
    per_global_rank_step_metrics: Dict[int, Dict[int, Dict[str, float]]]
    identities: Dict[int, GlobalRankIdentity]
    max_rows: int


def load_global_rank_step_rows(
    conn: sqlite3.Connection,
    *,
    global_rank: int,
    max_rows: int,
) -> list[Dict[str, Any]]:
    """Load the latest bounded step-time rows for one global rank."""
    cur = conn.execute(
        """
        SELECT step, events_json
        FROM step_time_samples
        WHERE global_rank = ?
        ORDER BY step DESC, id DESC
        LIMIT ?;
        """,
        (int(global_rank), int(max_rows)),
    )

    rows: list[Dict[str, Any]] = []
    for step, events_json in cur:
        if step is None or not events_json:
            continue
        try:
            events = json.loads(events_json)
        except Exception:
            continue
        if isinstance(events, dict):
            rows.append({"step": int(step), "events": events})
    return rows


def load_global_rank_identities(
    conn: sqlite3.Connection,
    global_ranks: Iterable[int],
) -> Dict[int, GlobalRankIdentity]:
    """Load the latest runtime identity metadata for each global rank."""
    identities: Dict[int, GlobalRankIdentity] = {}
    for global_rank in global_ranks:
        row = conn.execute(
            """
            SELECT local_rank, node_rank, hostname, local_world_size,
                   world_size
            FROM step_time_samples
            WHERE global_rank = ?
            ORDER BY sample_ts_s DESC, id DESC
            LIMIT 1;
            """,
            (int(global_rank),),
        ).fetchone()
        local_rank = int(row[0]) if row and row[0] is not None else None
        node_rank = int(row[1]) if row and row[1] is not None else None
        hostname = str(row[2]) if row and row[2] is not None else None
        local_world_size = int(row[3]) if row and row[3] is not None else None
        world_size = int(row[4]) if row and row[4] is not None else None
        identities[int(global_rank)] = GlobalRankIdentity(
            global_rank=int(global_rank),
            local_rank=local_rank,
            node_rank=node_rank,
            hostname=hostname,
            local_world_size=local_world_size,
            world_size=world_size,
        )
    return identities


def load_step_time_section_data(
    db_path: str,
    *,
    max_rows: int = MAX_SUMMARY_WINDOW_ROWS,
) -> StepTimeSectionData:
    """
    Load bounded step-time section data from the SQLite history database.
    """
    row_limit = normalize_summary_window_rows(max_rows)
    conn = sqlite3.connect(db_path)

    try:
        latest_step_observed_row = conn.execute(
            "SELECT MAX(step) FROM step_time_samples;"
        ).fetchone()
        latest_step_observed = (
            int(latest_step_observed_row[0])
            if latest_step_observed_row[0] is not None
            else None
        )
        training_steps = (
            latest_step_observed + 1 if latest_step_observed is not None else 0
        )

        global_rank_rows = conn.execute(
            """
            SELECT DISTINCT global_rank
            FROM step_time_samples
            WHERE global_rank IS NOT NULL
            ORDER BY global_rank ASC;
            """
        ).fetchall()
        global_ranks_present = [
            int(row[0]) for row in global_rank_rows if row[0] is not None
        ]
        identities = load_global_rank_identities(conn, global_ranks_present)

        per_global_rank_steps: Dict[int, Dict[int, Dict[str, Any]]] = {}

        for global_rank in global_ranks_present:
            step_rows = load_global_rank_step_rows(
                conn,
                global_rank=global_rank,
                max_rows=row_limit,
            )
            step_map = {
                int(row["step"]): row["events"]
                for row in step_rows
                if row.get("step") is not None
            }
            if step_map:
                per_global_rank_steps[int(global_rank)] = step_map

        step_time_window = build_step_time_window_from_events(
            per_global_rank_steps,
            max_rows=row_limit,
            expected_ranks=global_ranks_present,
        )
        selected_summary = rank_summaries_from_window(step_time_window)
    finally:
        conn.close()

    return StepTimeSectionData(
        training_steps=training_steps,
        latest_step_observed=latest_step_observed,
        step_time_window=step_time_window,
        aligned_summary=selected_summary,
        aligned_step_metrics=step_time_window.per_rank_step_timing,
        aligned_window=step_time_window,
        per_global_rank_summary=selected_summary,
        per_global_rank_step_metrics=step_time_window.per_rank_step_timing,
        identities=identities,
        max_rows=row_limit,
    )


__all__ = [
    "StepTimeSectionData",
    "load_global_rank_identities",
    "load_global_rank_step_rows",
    "load_step_time_section_data",
]
