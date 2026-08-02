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

from traceml_ai.step_time.analysis import StepTimeAnalyzer
from traceml_ai.step_time.model import (
    StepTimeLoadRequest,
    StepTimeRankIdentity,
    StepTimeRepositorySnapshot,
    StepTimeSourceCursor,
    StepTimeWindow,
)
from traceml_ai.step_time.sqlite import (
    SQLiteStepTimeRepository,
    load_training_strategy_from_sqlite,
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
    cursor: StepTimeSourceCursor = field(default_factory=StepTimeSourceCursor)


def _analyze_snapshot(
    snapshot: StepTimeRepositorySnapshot,
    *,
    max_rows: int,
) -> StepTimeSQLiteWindow:
    """Apply the shared analyzer to either repository data profile."""
    window = StepTimeAnalyzer().analyze(
        snapshot,
        window_size=max_rows,
    )
    return StepTimeSQLiteWindow(
        window=window,
        global_ranks=snapshot.global_ranks,
        training_strategy=snapshot.training_strategy,
        identities=snapshot.identities,
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
