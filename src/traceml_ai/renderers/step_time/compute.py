import json
import sqlite3
import time
from typing import Any, Dict, List, Optional, Sequence

from traceml_ai.loggers.error_log import get_error_logger
from traceml_ai.renderers.step_time.schema import StepCombinedTimeResult
from traceml_ai.utils.step_time_window import (
    build_step_time_window_from_events,
)

STEP_TIME_TABLE = "step_time_samples"


class StepCombinedComputer:
    """Compute selected-clock Step Time diagnosis payloads from SQLite."""

    def __init__(
        self,
        db_path: str,
        window_size: int = 100,
        stale_ttl_s: Optional[float] = 30.0,
        table: str = STEP_TIME_TABLE,
        lookback_factor: int = 4,
        rank_filter: Optional[Sequence[int]] = None,
    ) -> None:
        self.db_path = str(db_path)
        self.window_size = max(1, int(window_size))
        self.table = str(table)
        self.lookback_factor = max(1, int(lookback_factor))
        self.rank_filter = (
            {int(r) for r in rank_filter} if rank_filter is not None else None
        )

        self.logger = get_error_logger("StepCombinedComputer")
        self._last_ok: Optional[StepCombinedTimeResult] = None
        self._last_ok_ts = 0.0
        self._stale_ttl_s = (
            float(stale_ttl_s) if stale_ttl_s is not None else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_cli(self) -> StepCombinedTimeResult:
        return self._compute()

    def compute_dashboard(self) -> StepCombinedTimeResult:
        return self._compute()

    def _compute(self) -> StepCombinedTimeResult:
        try:
            with self._connect() as conn:
                result = self._compute_impl(conn)
        except Exception as exc:
            self.logger.exception("StepCombined compute failed")
            return self._stale_or_empty(
                f"STALE (exception: {type(exc).__name__})"
            )

        if not result.diagnosis_metrics:
            return self._stale_or_empty("STALE (no metrics this tick)")

        self._last_ok = result
        self._last_ok_ts = time.time()
        return result

    # ------------------------------------------------------------------
    # Core compute
    # ------------------------------------------------------------------

    def _compute_impl(
        self,
        conn: sqlite3.Connection,
    ) -> StepCombinedTimeResult:
        ranks = self._load_ranks(conn)
        if not ranks:
            return StepCombinedTimeResult(
                status_message="No ranks available",
            )

        per_rank_steps = self._load_last_steps(conn, ranks)
        if not per_rank_steps:
            return StepCombinedTimeResult(
                status_message="No StepTime data available",
            )

        window = build_step_time_window_from_events(
            per_rank_steps,
            max_rows=self.window_size,
            expected_ranks=ranks,
        )
        if not window.metrics:
            return StepCombinedTimeResult(
                status_message="No common step window yet",
            )

        status = "OK"
        diagnosis_worst_rank = _worst_rank_from_window(window.per_rank_timing)
        if diagnosis_worst_rank is not None:
            status += f" | diagnosis_worst_rank=r{diagnosis_worst_rank}"

        return StepCombinedTimeResult(
            status_message=status,
            per_rank_timing=window.per_rank_timing,
            diagnosis_clock=window.clock,
            diagnosis_metrics=window.metrics,
        )

    # ------------------------------------------------------------------
    # SQLite loading
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """
        Open a short-lived read connection.

        One connection per compute call keeps the implementation simple and
        avoids cross-thread SQLite issues.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_ranks(self, conn: sqlite3.Connection) -> List[int]:
        """
        Load available ranks from the step-time table.

        Returns only non-null ranks, optionally filtered by `rank_filter`.
        """
        rows = conn.execute(
            f"""
            SELECT DISTINCT rank
            FROM {self.table}
            WHERE rank IS NOT NULL
            ORDER BY rank ASC;
            """
        ).fetchall()

        ranks = [int(r["rank"]) for r in rows if r["rank"] is not None]
        if self.rank_filter is not None:
            ranks = [r for r in ranks if r in self.rank_filter]
        return ranks

    def _load_last_steps(
        self,
        conn: sqlite3.Connection,
        ranks: List[int],
    ) -> Dict[int, Dict[int, Dict[str, Any]]]:
        """
        Load a bounded recent step map per rank.

        To avoid scanning the full table, this method only reads a recent
        suffix per rank. The lookback is intentionally larger than the final
        window size so the common-step suffix has room to form even when some
        ranks are slightly ahead/behind.

        Returns
        -------
        dict[int, dict[int, dict]]
            rank -> step -> events_payload
        """
        out: Dict[int, Dict[int, Dict[str, Any]]] = {}
        lookback = max(
            self.window_size * self.lookback_factor, self.window_size
        )

        for rank in ranks:
            try:
                rows = conn.execute(
                    f"""
                    SELECT step, events_json
                    FROM (
                        SELECT step, events_json
                        FROM {self.table}
                        WHERE rank = ?
                        ORDER BY step DESC, id DESC
                        LIMIT ?
                    )
                    ORDER BY step ASC;
                    """,
                    (int(rank), int(lookback)),
                ).fetchall()

                if not rows:
                    continue

                step_map: Dict[int, Dict[str, Any]] = {}
                for row in rows:
                    step_raw = row["step"]
                    if step_raw is None:
                        continue

                    step = int(step_raw)
                    # If duplicates somehow exist, keep the latest from the DESC/limit
                    # selection; the final ORDER BY ASC preserves chronological order.
                    if step in step_map:
                        continue

                    events_json = row["events_json"]
                    try:
                        parsed = json.loads(events_json) if events_json else {}
                    except Exception:
                        parsed = {}

                    step_map[step] = parsed if isinstance(parsed, dict) else {}

                if step_map:
                    out[int(rank)] = step_map

            except Exception:
                self.logger.exception(
                    "Failed loading step data for rank=%s", rank
                )

        return out

    # ------------------------------------------------------------------
    # Stale handling
    # ------------------------------------------------------------------

    def _stale_or_empty(self, msg: str) -> StepCombinedTimeResult:
        now = time.time()
        if self._last_ok is not None:
            if (
                self._stale_ttl_s is None
                or (now - self._last_ok_ts) <= self._stale_ttl_s
            ):
                return StepCombinedTimeResult(
                    status_message=msg,
                    per_rank_timing=self._last_ok.per_rank_timing,
                    diagnosis_clock=self._last_ok.diagnosis_clock,
                    diagnosis_metrics=self._last_ok.diagnosis_metrics,
                )
        return StepCombinedTimeResult(
            status_message="No fresh step-combined data",
            per_rank_timing={},
            diagnosis_clock="cpu",
            diagnosis_metrics=[],
        )


def _worst_rank_from_window(
    per_rank_timing: Dict[int, Dict[str, float]],
) -> Optional[int]:
    """Return the slowest rank by selected average total step."""
    if not per_rank_timing:
        return None
    return max(
        per_rank_timing,
        key=lambda rank: (
            float(per_rank_timing.get(rank, {}).get("total_step", 0.0)),
            -int(rank),
        ),
    )
