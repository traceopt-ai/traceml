"""
Dashboard compute for step-memory telemetry.

Uses the same core aggregation as CLI, with independent window sizing and stale
fallback policy for dashboard consumers.
"""

import time
from dataclasses import replace
from typing import Optional, Tuple

from traceml_ai.loggers.error_log import get_error_logger
from traceml_ai.renderers.shared.freshness import (
    CachedPayloadTTL,
    LastGoodVerdict,
    RankLiveness,
)

from .common import StepMemoryMetricsDB, build_step_memory_combined_result
from .schema import StepMemoryCombinedResult


class StepMemoryDashboardComputer:
    """Compute step-memory payload for dashboard rendering."""

    def __init__(
        self,
        db_path: str,
        *,
        window_size: int = 200,
        stale_ttl_s: Optional[float] = 30.0,
        sampler_interval_s: Optional[float] = None,
    ) -> None:
        self._db = StepMemoryMetricsDB(db_path=db_path)
        self._window_size = int(window_size)
        self._sampler_interval_s = sampler_interval_s
        self._logger = get_error_logger("StepMemoryDashboardComputer")

        self._last_ok: Optional[StepMemoryCombinedResult] = None
        self._last_ok_ts: float = 0.0
        self._stale_ttl_s: Optional[float] = (
            float(stale_ttl_s) if stale_ttl_s is not None else None
        )
        self._liveness: LastGoodVerdict[Tuple[RankLiveness, ...]] = (
            LastGoodVerdict(CachedPayloadTTL(ttl_s=self._stale_ttl_s))
        )

    def compute(self) -> StepMemoryCombinedResult:
        """Return latest dashboard payload (with stale fallback)."""
        now = time.time()
        try:
            with self._db.connect() as conn:
                out = build_step_memory_combined_result(
                    conn,
                    db=self._db,
                    window_size=self._window_size,
                    configured_interval_s=self._sampler_interval_s,
                )
        except Exception:
            self._logger.exception("Step memory dashboard compute failed")
            return self._return_stale_or_empty(
                "STALE (exception)",
                rank_liveness=self._liveness.carry(None, now_s=now),
            )

        # A heartbeat read that failed, even beside fresh metrics, is
        # answered by the last good verdict rather than by none.
        out = replace(
            out,
            rank_liveness=self._liveness.carry(out.rank_liveness, now_s=now),
        )
        if not out.metrics:
            if "No GPU detected" in str(out.status_message):
                self._last_ok = None
                self._last_ok_ts = 0.0
                return out
            return self._return_stale_or_empty(
                "STALE (no metrics this tick)",
                rank_liveness=out.rank_liveness,
            )

        self._last_ok = out
        self._last_ok_ts = time.time()
        return out

    def _return_stale_or_empty(
        self,
        msg: str,
        *,
        rank_liveness: Optional[Tuple[RankLiveness, ...]] = None,
    ) -> StepMemoryCombinedResult:
        """Reuse the last good metrics, with this tick's rank liveness.

        ``rank_liveness`` is ``None`` when there is no verdict (unread,
        and no last good one inside the TTL). ``()`` means it was read and
        no rank has reported.
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
                    rank_liveness=rank_liveness,
                )

        return StepMemoryCombinedResult(
            metrics=[],
            status_message="No complete memory metrics available",
            rank_liveness=rank_liveness,
        )
