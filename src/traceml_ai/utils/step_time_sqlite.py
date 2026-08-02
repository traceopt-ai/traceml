"""Compatibility adapters for canonical Step Time SQLite windows.

SQL selection and JSON decoding live in :mod:`traceml_ai.step_time.sqlite`.
This module preserves the existing analyzed-window API while callers migrate
to the repository boundary.
"""

# TODO(PR9): Remove this compatibility adapter after all consumers call the
# repository and analyzer boundaries directly.

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from traceml_ai.step_time.model import (
    StepTimeLoadRequest,
    StepTimeRankIdentity,
    StepTimeRepositorySnapshot,
    StepTimeSourceCursor,
    StepTimeSourceRow,
    StepTimeWindow,
)
from traceml_ai.step_time.sqlite import (
    SQLiteStepTimeRepository,
    load_training_strategy_from_sqlite,
)
from traceml_ai.utils.step_time_window import (
    build_step_time_window_from_events,
)
from traceml_ai.utils.training_strategy import (
    DEFAULT_TRAINING_STRATEGY,
    KNOWN_TRAINING_STRATEGIES,
)


@dataclass(frozen=True)
class StepTimeSQLiteWindow:
    """Analyzed Step Time window plus repository source context."""

    window: StepTimeWindow
    global_ranks: tuple[int, ...]
    training_strategy: str = DEFAULT_TRAINING_STRATEGY
    identities: Mapping[int, StepTimeRankIdentity] = field(
        default_factory=dict
    )
    latest_step_observed: Optional[int] = None
    cursor: StepTimeSourceCursor = field(default_factory=StepTimeSourceCursor)


# TODO(PR4): Remove regrouping when StepTimeAnalyzer consumes flat source rows.
def _group_source_rows(
    rows: Sequence[StepTimeSourceRow],
) -> dict[int, dict[int, StepTimeSourceRow]]:
    """Group flat source rows for the unchanged window-analysis contract."""
    grouped: dict[int, dict[int, StepTimeSourceRow]] = {}
    for row in rows:
        grouped.setdefault(row.global_rank, {})[row.step] = row
    return grouped


def _analyze_snapshot(
    snapshot: StepTimeRepositorySnapshot,
    *,
    max_rows: int,
) -> StepTimeSQLiteWindow:
    """Apply the shared analyzer to either repository data profile."""
    window = build_step_time_window_from_events(
        _group_source_rows(snapshot.rows),
        max_rows=max_rows,
        expected_ranks=snapshot.global_ranks,
    )
    return StepTimeSQLiteWindow(
        window=window,
        global_ranks=snapshot.global_ranks,
        training_strategy=snapshot.training_strategy,
        identities=snapshot.identities,
        latest_step_observed=snapshot.latest_step_observed,
        cursor=snapshot.cursor,
    )


def load_step_time_window_from_sqlite(
    conn: sqlite3.Connection,
    *,
    max_rows: int,
    lookback_factor: int = 1,
    table: str = "step_time_samples",
    rank_filter: Optional[Sequence[int]] = None,
) -> StepTimeSQLiteWindow:
    """Load and analyze the bounded live Step Time source profile.

    This historical signature now names the terminal/dashboard data profile.
    Final summary uses :func:`load_step_time_summary_from_sqlite`. Both retain
    the same analysis and return contract.
    """
    row_limit = max(1, int(max_rows))
    request = StepTimeLoadRequest(
        window_size=row_limit,
        lookback_factor=max(1, int(lookback_factor)),
        rank_filter=(
            tuple(int(rank) for rank in rank_filter)
            if rank_filter is not None
            else None
        ),
    )
    snapshot = SQLiteStepTimeRepository(conn, table=table).load_live(request)
    return _analyze_snapshot(snapshot, max_rows=row_limit)


def load_step_time_summary_from_sqlite(
    conn: sqlite3.Connection,
    *,
    max_rows: int,
    table: str = "step_time_samples",
    rank_filter: Optional[Sequence[int]] = None,
) -> StepTimeSQLiteWindow:
    """Load the metadata-complete source profile used by final summary.

    The returned analysis is identical to the live adapter for an equivalent
    request. Only the SQLite selection strategy and metadata completeness
    differ.
    """
    row_limit = max(1, int(max_rows))
    request = StepTimeLoadRequest(
        window_size=row_limit,
        rank_filter=(
            tuple(int(rank) for rank in rank_filter)
            if rank_filter is not None
            else None
        ),
    )
    repository = SQLiteStepTimeRepository(conn, table=table)
    snapshot = repository.load_summary(request)
    return _analyze_snapshot(snapshot, max_rows=row_limit)


__all__ = [
    "DEFAULT_TRAINING_STRATEGY",
    "KNOWN_TRAINING_STRATEGIES",
    "StepTimeSQLiteWindow",
    "load_training_strategy_from_sqlite",
    "load_step_time_summary_from_sqlite",
    "load_step_time_window_from_sqlite",
]
