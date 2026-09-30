"""
CLI compute for step-memory telemetry.

This wrapper computes a terminal-oriented payload from SQLite and supports
stale fallback to avoid panel flicker on transient DB/read issues.
"""

import time
from dataclasses import replace
from typing import Optional, Tuple

from traceml_ai.loggers.error_log import get_error_logger
from traceml_ai.renderers.shared.freshness import RankReporting

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
        sampler_interval_s: Optional[float] = None,
    ) -> None:
        self._db = StepMemoryMetricsDB(db_path=db_path)
        self._window_size = int(window_size)
        self._sampler_interval_s = sampler_interval_s
        self._logger = get_error_logger("StepMemoryCLIComputer")

        self._last_ok: Optional[StepMemoryCombinedResult] = None
        self._last_ok_ts: float = 0.0
        self._stale_ttl_s: Optional[float] = (
            float(stale_ttl_s) if stale_ttl_s is not None else None
        )

    def compute(self) -> StepMemoryCombinedResult:
        """Return latest CLI payload (with stale fallback on transient failures).

        The rank reporting status is always this tick's own read, never
        the one held figures were computed with; ``None`` when it failed.
        """
        reporting: Optional[Tuple[RankReporting, ...]] = None
        try:
            with self._db.connect() as conn:
                reporting = self._db.fetch_rank_reporting(
                    conn, configured_interval_s=self._sampler_interval_s
                )
                out = build_step_memory_combined_result(
                    conn,
                    db=self._db,
                    window_size=self._window_size,
                )
        except Exception:
            self._logger.exception("Step memory CLI compute failed")
            return self._return_stale_or_empty(
                "STALE (exception)", rank_reporting=reporting
            )

        out = replace(out, rank_reporting=reporting)
        if not out.metrics:
            if "No GPU detected" in str(out.status_message):
                self._last_ok = None
                self._last_ok_ts = 0.0
                return out
            return self._return_stale_or_empty(
                "STALE (no metrics this tick)",
                rank_reporting=out.rank_reporting,
            )

        self._last_ok = out
        self._last_ok_ts = time.time()
        return out

    def _return_stale_or_empty(
        self,
        msg: str,
        *,
        rank_reporting: Optional[Tuple[RankReporting, ...]] = None,
    ) -> StepMemoryCombinedResult:
        """Reuse the last good metrics, with this tick's reporting status.

        ``rank_reporting`` is ``None`` when this tick could not read it.
        ``()`` means it was read and no rank has reported.
        """
        now = time.time()
        if self._last_ok is not None:
            if (
                self._stale_ttl_s is None
                or (now - self._last_ok_ts) <= self._stale_ttl_s
            ):
                return StepMemoryCombinedResult(
                    metrics=self._last_ok.metrics,
                    status_message=msg,
                    gpu_total_bytes=self._last_ok.gpu_total_bytes,
                    rank_reporting=rank_reporting,
                )

        return StepMemoryCombinedResult(
            metrics=[],
            status_message="No complete memory metrics available",
            rank_reporting=rank_reporting,
        )
