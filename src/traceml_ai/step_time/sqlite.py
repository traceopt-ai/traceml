"""SQLite repository for bounded, consistent Step Time source snapshots.

The repository owns SQL selection and JSON normalization only. Live consumers
use an index-bounded tail query; final summary uses a metadata-complete query.
Both return the same source contract and deliberately leave alignment, clock
selection, derivation, diagnosis, and formatting to later pipeline layers.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from traceml_ai.step_time.model import (
    STEP_TIME_EVENT_NAMES,
    StepTimeClockValues,
    StepTimeLoadRequest,
    StepTimeRankIdentity,
    StepTimeRepositorySnapshot,
    StepTimeSourceCursor,
    StepTimeSourceRow,
)
from traceml_ai.utils.training_strategy import (
    DEFAULT_TRAINING_STRATEGY,
    normalize_training_strategy,
)

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_IDENTITY_FIELDS = (
    "local_rank",
    "node_rank",
    "hostname",
    "local_world_size",
    "world_size",
)


def _optional_non_negative_float(value: Any) -> Optional[float]:
    """Return one finite non-negative duration, preserving missing values."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = value
    else:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
    if not math.isfinite(result):
        return None
    return max(0.0, float(result))


def _sum_clocks(by_device: Any) -> StepTimeClockValues:
    """Sum both clocks across devices in one pass."""
    if not isinstance(by_device, dict):
        return StepTimeClockValues()

    cpu_total = 0.0
    gpu_total = 0.0
    cpu_measured = False
    gpu_measured = False
    for stats in by_device.values():
        if not isinstance(stats, dict):
            continue
        cpu_ms = _optional_non_negative_float(stats.get("cpu_ms"))
        gpu_ms = _optional_non_negative_float(stats.get("gpu_ms"))
        if cpu_ms is not None:
            cpu_total += cpu_ms
            cpu_measured = True
        if gpu_ms is not None:
            gpu_total += gpu_ms
            gpu_measured = True
    return StepTimeClockValues(
        cpu_ms=cpu_total if cpu_measured else None,
        gpu_ms=gpu_total if gpu_measured else None,
    )


def normalize_step_time_events(
    events: Any,
) -> Optional[dict[str, StepTimeClockValues]]:
    """Normalize one persisted event mapping into dual-clock source values.

    The repository decoder and the temporary raw-fixture adapter share this
    boundary so device summation and invalid-duration handling have one owner.
    """
    if not isinstance(events, dict):
        return None

    metrics: dict[str, StepTimeClockValues] = {}
    for metric, event_name in STEP_TIME_EVENT_NAMES.items():
        payload = events.get(event_name)
        if not isinstance(payload, dict) or not payload:
            continue
        metrics[metric] = _sum_clocks(payload)
    return metrics


def _decode_metrics(
    events_json: str,
) -> Optional[dict[str, StepTimeClockValues]]:
    """Decode one JSON payload into semantic dual-clock metric values."""
    try:
        events = json.loads(events_json)
    except (TypeError, ValueError):
        return None
    return normalize_step_time_events(events)


def _optional_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _optional_str(value: Any) -> Optional[str]:
    return str(value) if value is not None else None


def load_training_strategy_from_sqlite(conn: sqlite3.Connection) -> str:
    """Load the latest recognized strategy, defaulting safely to ``ddp``."""
    try:
        row = conn.execute(
            """
            SELECT training_strategy
            FROM runtime_environment
            WHERE training_strategy IS NOT NULL
              AND TRIM(training_strategy) != ''
            ORDER BY id DESC
            LIMIT 1;
            """
        ).fetchone()
    except Exception:
        return DEFAULT_TRAINING_STRATEGY

    if not row:
        return DEFAULT_TRAINING_STRATEGY
    return normalize_training_strategy(row[0])


class SQLiteStepTimeRepository:
    """Read decoded Step Time source facts from one caller-owned connection.

    Args:
        conn: SQLite connection. The repository never closes it.
        table: Step Time projection table. Only a plain SQL identifier is
            accepted because SQLite cannot parameterize identifiers.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        table: str = "step_time_samples",
    ) -> None:
        table_name = str(table)
        if _IDENTIFIER.fullmatch(table_name) is None:
            raise ValueError(f"Invalid Step Time table name: {table_name!r}")
        self._conn = conn
        self._table = table_name

    def load_live(
        self,
        request: StepTimeLoadRequest,
    ) -> StepTimeRepositorySnapshot:
        """Load the bounded tail needed by terminal and dashboard refreshes.

        The query performs an index-ordered, early-stopping scan for each rank.
        It intentionally omits rank identities and run progress, which live
        diagnosis does not read. Cost therefore follows the requested lookback
        instead of the total number of persisted training steps.
        """
        with self._read_snapshot():
            rows = self._load_live_rows(request)
            strategy = load_training_strategy_from_sqlite(self._conn)
            return self._snapshot_from_rows(rows, strategy=strategy)

    def load_summary(
        self,
        request: StepTimeLoadRequest,
    ) -> StepTimeRepositorySnapshot:
        """Load timing rows and final-summary metadata."""
        with self._read_snapshot():
            rows = self._load_summary_rows(request)
            identities, latest_step, last_row_id = self._summary_context(rows)
            strategy = load_training_strategy_from_sqlite(self._conn)
            return self._snapshot_from_rows(
                rows,
                identities=identities,
                latest_step=latest_step,
                last_row_id=last_row_id,
                strategy=strategy,
            )

    @contextmanager
    def _read_snapshot(self) -> Iterator[None]:
        """Share transaction ownership without coupling the two SQL paths."""
        owns_snapshot = not self._conn.in_transaction
        if owns_snapshot:
            self._conn.execute("BEGIN")

        try:
            yield
        finally:
            if owns_snapshot and self._conn.in_transaction:
                self._conn.rollback()

    def _table_columns(self) -> set[str]:
        """Return available projection columns for legacy-schema support."""
        columns = {
            str(row[1])
            for row in self._conn.execute(
                f'PRAGMA table_info("{self._table}");'
            ).fetchall()
        }
        required = {"id", "global_rank", "step", "events_json"}
        missing = sorted(required - columns)
        if missing:
            names = ", ".join(missing)
            raise ValueError(
                f"Step Time table {self._table!r} is missing: {names}"
            )
        return columns

    @staticmethod
    def _selection_parameters(
        request: StepTimeLoadRequest,
    ) -> tuple[int, Optional[tuple[int, ...]]]:
        """Normalize a request into a lookback and optional rank set."""
        window_size = max(1, int(request.window_size))
        lookback = window_size * max(1, int(request.lookback_factor))
        ranks = (
            tuple(sorted({int(rank) for rank in request.rank_filter}))
            if request.rank_filter is not None
            else None
        )
        return lookback, ranks

    def _rank_universe_cte(
        self,
        ranks: Optional[tuple[int, ...]],
    ) -> tuple[str, list[Any]]:
        """Build an index-seeking rank CTE shared by both data profiles."""
        if ranks is None:
            return (
                f"""rank_universe(global_rank) AS (
                    SELECT MIN(global_rank)
                    FROM "{self._table}"
                    WHERE global_rank IS NOT NULL
                    UNION ALL
                    SELECT (
                        SELECT MIN(sample.global_rank)
                        FROM "{self._table}" AS sample
                        WHERE sample.global_rank > ranks.global_rank
                    )
                    FROM rank_universe AS ranks
                    WHERE ranks.global_rank IS NOT NULL
                )""",
                [],
            )
        if not ranks:
            return "rank_universe(global_rank) AS (SELECT NULL WHERE 0)", []

        values = ", ".join("(?)" for _ in ranks)
        return (
            f"""requested_ranks(global_rank) AS (VALUES {values}),
                rank_universe(global_rank) AS (
                    SELECT requested.global_rank
                    FROM requested_ranks AS requested
                    WHERE EXISTS (
                        SELECT 1
                        FROM "{self._table}" AS sample
                        WHERE sample.global_rank = requested.global_rank
                    )
                )""",
            list(ranks),
        )

    def _load_live_rows(
        self,
        request: StepTimeLoadRequest,
    ) -> list[sqlite3.Row | tuple[Any, ...]]:
        """Select a deduplicated tail without scanning the full run."""
        self._table_columns()
        lookback, ranks = self._selection_parameters(request)
        rank_cte, parameters = self._rank_universe_cte(ranks)
        query = f"""
            WITH RECURSIVE {rank_cte}
            SELECT
                ranks.global_rank,
                sample.id,
                sample.step,
                sample.events_json
            FROM rank_universe AS ranks
            LEFT JOIN "{self._table}" AS sample
              ON sample.id IN (
                    SELECT MAX(candidate.id)
                    FROM "{self._table}" AS candidate
                    WHERE candidate.global_rank = ranks.global_rank
                      AND candidate.step IS NOT NULL
                      AND candidate.events_json IS NOT NULL
                      AND candidate.events_json != ''
                    GROUP BY candidate.step
                    ORDER BY candidate.step DESC
                    LIMIT ?
              )
            WHERE ranks.global_rank IS NOT NULL;
        """
        parameters.append(lookback)
        return self._conn.execute(query, parameters).fetchall()

    def _load_summary_rows(
        self,
        request: StepTimeLoadRequest,
    ) -> list[sqlite3.Row | tuple[Any, ...]]:
        """Select bounded rows plus the metadata required by final summary."""
        columns = self._table_columns()
        lookback, ranks = self._selection_parameters(request)
        rank_cte, parameters = self._rank_universe_cte(ranks)
        parameters.append(lookback)

        projected = [
            f"identity.{name}" if name in columns else f"NULL AS {name}"
            for name in _IDENTITY_FIELDS
        ]
        identity_projection = ",\n                    ".join(projected)
        identity_order = (
            "latest.sample_ts_s DESC, latest.id DESC"
            if "sample_ts_s" in columns
            else "latest.id DESC"
        )
        query = f"""
            WITH RECURSIVE {rank_cte},
            bounded AS (
                SELECT
                    ranks.global_rank,
                    sample.id,
                    sample.step,
                    sample.events_json
                FROM rank_universe AS ranks
                LEFT JOIN "{self._table}" AS sample
                  ON sample.id IN (
                        SELECT MAX(candidate.id)
                        FROM "{self._table}" AS candidate
                        WHERE candidate.global_rank = ranks.global_rank
                          AND candidate.step IS NOT NULL
                          AND candidate.events_json IS NOT NULL
                          AND candidate.events_json != ''
                        GROUP BY candidate.step
                        ORDER BY candidate.step DESC
                        LIMIT ?
                  )
                WHERE ranks.global_rank IS NOT NULL
            ),
            identity_ids AS (
                SELECT
                    ranks.global_rank,
                    MAX((
                        SELECT latest.id
                        FROM "{self._table}" AS latest
                        WHERE latest.global_rank = ranks.global_rank
                        ORDER BY {identity_order}
                        LIMIT 1
                    )) AS id
                FROM rank_universe AS ranks
                WHERE ranks.global_rank IS NOT NULL
                GROUP BY ranks.global_rank
            ),
            progress AS (
                SELECT
                    MAX(step) AS latest_step,
                    MAX(id) AS last_row_id
                FROM "{self._table}"
            )
            SELECT
                bounded.global_rank,
                bounded.id,
                bounded.step,
                bounded.events_json,
                {identity_projection},
                progress.latest_step,
                progress.last_row_id
            FROM bounded
            LEFT JOIN identity_ids
              ON identity_ids.global_rank = bounded.global_rank
            LEFT JOIN "{self._table}" AS identity
              ON identity.id = identity_ids.id
            CROSS JOIN progress
            ;
        """
        return self._conn.execute(query, parameters).fetchall()

    @staticmethod
    def _summary_context(
        rows: list[sqlite3.Row | tuple[Any, ...]],
    ) -> tuple[
        dict[int, StepTimeRankIdentity],
        Optional[int],
        Optional[int],
    ]:
        """Project summary-only identity and progress columns."""
        identities: dict[int, StepTimeRankIdentity] = {}
        latest_step: Optional[int] = None
        last_row_id: Optional[int] = None
        for row in rows:
            global_rank = int(row[0])
            identities.setdefault(
                global_rank,
                StepTimeRankIdentity(
                    global_rank=global_rank,
                    local_rank=_optional_int(row[4]),
                    node_rank=_optional_int(row[5]),
                    hostname=_optional_str(row[6]),
                    local_world_size=_optional_int(row[7]),
                    world_size=_optional_int(row[8]),
                ),
            )
            if latest_step is None:
                latest_step = _optional_int(row[9])
            if last_row_id is None:
                last_row_id = _optional_int(row[10])
        return identities, latest_step, last_row_id

    @staticmethod
    def _snapshot_from_rows(
        rows: list[sqlite3.Row | tuple[Any, ...]],
        *,
        strategy: str,
        identities: Optional[dict[int, StepTimeRankIdentity]] = None,
        latest_step: Optional[int] = None,
        last_row_id: Optional[int] = None,
    ) -> StepTimeRepositorySnapshot:
        global_ranks: list[int] = []
        seen_ranks: set[int] = set()
        source_rows: list[StepTimeSourceRow] = []

        for row in rows:
            global_rank = int(row[0])
            if global_rank not in seen_ranks:
                global_ranks.append(global_rank)
                seen_ranks.add(global_rank)

            source_id = _optional_int(row[1])
            step = _optional_int(row[2])
            if source_id is None or step is None or not row[3]:
                continue
            metrics = _decode_metrics(str(row[3]))
            if metrics is None:
                continue
            source_rows.append(
                StepTimeSourceRow(
                    source_id=source_id,
                    global_rank=global_rank,
                    step=step,
                    metrics=metrics,
                )
            )

        if latest_step is None:
            latest_step = max(
                (row.step for row in source_rows),
                default=None,
            )
        if last_row_id is None:
            last_row_id = max(
                (row.source_id for row in source_rows),
                default=None,
            )

        return StepTimeRepositorySnapshot(
            rows=tuple(source_rows),
            global_ranks=tuple(global_ranks),
            identities=identities or {},
            cursor=StepTimeSourceCursor(
                last_row_id=last_row_id,
                latest_step=latest_step,
            ),
            training_strategy=strategy,
        )


__all__ = [
    "SQLiteStepTimeRepository",
    "load_training_strategy_from_sqlite",
    "normalize_step_time_events",
]
