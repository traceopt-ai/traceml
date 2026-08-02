import sqlite3
import time
from dataclasses import replace
from typing import Dict, Optional, Sequence

from traceml_ai.loggers.error_log import get_error_logger
from traceml_ai.step_time.model import StepTimeResult
from traceml_ai.utils.step_time_sqlite import load_step_time_window_from_sqlite

STEP_TIME_TABLE = "step_time_samples"


class StepCombinedComputer:
    """
    Compute selected-clock Step Time diagnosis payloads from SQLite.

    Live delegates global-rank SQLite loading to the shared Step Time window
    loader, then adds CLI/dashboard-specific empty-read bridging and status
    text. Consumers receive that canonical window without copied metrics.
    """

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
        self._last_ok: Optional[StepTimeResult] = None
        self._last_ok_ts = 0.0
        self._stale_ttl_s = (
            float(stale_ttl_s) if stale_ttl_s is not None else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_cli(self) -> StepTimeResult:
        return self._compute()

    def compute_dashboard(self) -> StepTimeResult:
        return self._compute()

    @property
    def had_ok(self) -> bool:
        """Whether any compute ever produced diagnosis metrics.

        Lets renderers distinguish "no data yet" from an expired last-good
        bridge after repeated empty computes. It does not claim process
        liveness.
        """
        return self._last_ok is not None

    def _compute(self) -> StepTimeResult:
        try:
            with self._connect() as conn:
                result = self._compute_impl(conn)
        except Exception as exc:
            self.logger.exception("StepCombined compute failed")
            return self._stale_or_empty(
                f"STALE (exception: {type(exc).__name__})"
            )

        if result.window is None or not result.window.metrics:
            return self._stale_or_empty("STALE (no metrics this tick)")

        now = time.time()
        self._last_ok = result
        self._last_ok_ts = now
        return replace(result, had_ok=True)

    # ------------------------------------------------------------------
    # Core compute
    # ------------------------------------------------------------------

    def _compute_impl(
        self,
        conn: sqlite3.Connection,
    ) -> StepTimeResult:
        loaded = load_step_time_window_from_sqlite(
            conn,
            max_rows=self.window_size,
            lookback_factor=self.lookback_factor,
            table=self.table,
            rank_filter=(
                tuple(self.rank_filter)
                if self.rank_filter is not None
                else None
            ),
        )
        ranks = loaded.global_ranks
        if not ranks:
            return StepTimeResult(
                status_message="No ranks available",
            )

        window = loaded.window
        if window.coverage.ranks_present <= 0:
            return StepTimeResult(
                status_message="No StepTime data available",
            )
        if not window.metrics:
            return StepTimeResult(
                status_message="No common step window yet",
            )

        status = "OK"
        diagnosis_worst_rank = _worst_rank_from_window(window.per_rank_timing)
        if diagnosis_worst_rank is not None:
            status += f" | diagnosis_worst_rank=r{diagnosis_worst_rank}"

        return StepTimeResult(
            status_message=status,
            window=window,
            training_strategy=loaded.training_strategy,
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

    # ------------------------------------------------------------------
    # Stale handling
    # ------------------------------------------------------------------

    def _stale_or_empty(self, msg: str) -> StepTimeResult:
        """Bridge transient empty reads without inferring producer liveness."""
        now = time.time()
        had_ok = self._last_ok is not None
        if self._last_ok is not None:
            if (
                self._stale_ttl_s is None
                or (now - self._last_ok_ts) <= self._stale_ttl_s
            ):
                return StepTimeResult(
                    status_message=msg,
                    window=self._last_ok.window,
                    training_strategy=self._last_ok.training_strategy,
                    had_ok=True,
                )
        return StepTimeResult(
            status_message="No fresh step-combined data",
            had_ok=had_ok,
        )


def _worst_rank_from_window(
    per_rank_timing: Dict[int, Dict[str, float]],
) -> Optional[int]:
    """Return the slowest rank by selected average total step.

    Only ranks whose total step is measured compete; when none is (the
    input wait was never measured anywhere), no rank is named rather
    than defaulting every rank to a fake zero.
    """
    candidates = {
        int(rank): float(values["total_step"])
        for rank, values in per_rank_timing.items()
        if "total_step" in values
    }
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda rank: (candidates[rank], -rank),
    )
