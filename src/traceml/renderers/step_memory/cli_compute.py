"""
CLI compute for step-memory telemetry.

This wrapper computes a terminal-oriented payload from SQLite and supports
stale fallback to avoid panel flicker on transient DB/read issues.
"""

import time
from typing import Optional

from traceml.loggers.error_log import get_error_logger

from .common import StepMemoryMetricsDB, build_step_memory_combined_result
from .schema import StepMemoryCombinedResult


class StepMemoryCLIComputer:
    """Compute step-memory payload for CLI rendering."""

    def __init__(
        self,
        db_path: str,
        *,
        window_size: int = 100,
        stale_ttl_s: Optional[float] = 30.0,
    ) -> None:
        self._db = StepMemoryMetricsDB(db_path=db_path)
        self._window_size = int(window_size)
        self._logger = get_error_logger("StepMemoryCLIComputer")

        self._last_ok: Optional[StepMemoryCombinedResult] = None
        self._last_ok_ts: float = 0.0
        self._stale_ttl_s: Optional[float] = (
            float(stale_ttl_s) if stale_ttl_s is not None else None
        )

    def compute(self) -> StepMemoryCombinedResult:
        """Return latest CLI payload (with stale fallback on transient failures)."""
        try:
            with self._db.connect() as conn:
                out = build_step_memory_combined_result(
                    conn,
                    db=self._db,
                    window_size=self._window_size,
                )
        except Exception:
            self._logger.exception("Step memory CLI compute failed")
            return self._return_stale_or_empty("STALE (exception)")

        if not out.metrics:
            if "No GPU detected" in str(out.status_message):
                self._last_ok = None
                self._last_ok_ts = 0.0
                return out
            return self._return_stale_or_empty("STALE (no metrics this tick)")

        self._last_ok = out
        self._last_ok_ts = time.time()
        return out

    def _return_stale_or_empty(self, msg: str) -> StepMemoryCombinedResult:
        now = time.time()
        if self._last_ok is not None:
            if (
                self._stale_ttl_s is None
                or (now - self._last_ok_ts) <= self._stale_ttl_s
            ):
                return StepMemoryCombinedResult(
                    metrics=self._last_ok.metrics,
                    status_message=msg,
                )

        return StepMemoryCombinedResult(
            metrics=[],
            status_message="No complete memory metrics available",
        )
