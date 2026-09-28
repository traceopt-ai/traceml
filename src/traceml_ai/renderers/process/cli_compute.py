"""
CLI compute for process telemetry.

This module computes a terminal-oriented live snapshot from SQLite.

Semantics
---------
- latest globally committed seq only
- cross-rank aggregation
- output keys match the current terminal renderer expectations
- per-rank Process reporting status, judged as the dashboard judges it
- the whole run's newest arrival, for the terminal's run-wide line
"""

import time
from typing import Any, Callable, Dict, Optional, Tuple

from traceml_ai.renderers.shared.freshness import RankReporting, RunReporting

from .common import (
    ProcessCLISnapshot,
    rank_reporting_dicts,
    run_reporting_dict,
)
from .reporting import read_rank_clock
from .repository import ProcessRepository

# Every rank's status and the run-wide one, from one read.
_Statuses = Tuple[Optional[Tuple[RankReporting, ...]], Optional[RunReporting]]
_UNREAD: _Statuses = (None, None)


class ProcessCLIComputer:
    """
    Compute the terminal/live snapshot for process telemetry.

    Parameters
    ----------
    db_path:
        Path to the SQLite database.
    stale_ttl_s:
        Maximum age in seconds for stale fallback reuse. When None, stale
        snapshots may be reused indefinitely.
    sampler_interval_s:
        Configured process-sampling cadence used until an observed cadence
        is available for judging rank freshness.
    now_fn:
        The aggregator's clock, the one that stamps every arrival. It ages
        the run's newest arrival for the run-wide status.
    """

    def __init__(
        self,
        db_path: str,
        stale_ttl_s: Optional[float] = 30.0,
        sampler_interval_s: Optional[float] = None,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        self._db = ProcessRepository(db_path=db_path)
        self._configured_interval_s = sampler_interval_s
        # Injectable so the run-wide status can be tested at a chosen
        # moment rather than by sleeping.
        self._now_fn = now_fn
        self._last_ok: Optional[Dict[str, Any]] = None
        self._last_ok_ts: float = 0.0
        self._stale_ttl_s: Optional[float] = (
            float(stale_ttl_s) if stale_ttl_s is not None else None
        )

    def compute(self) -> Dict[str, Any]:
        """
        Compute the latest live snapshot.

        Returns
        -------
        dict[str, Any]
            Terminal-facing snapshot. On transient failure, returns the previous
            good snapshot if still within stale TTL, with this tick's
            reporting statuses rather than the ones it was computed with.
        """
        statuses = _UNREAD
        try:
            with self._db.connect() as conn:
                statuses = self._read_statuses(conn)
                out = self._compute_impl(conn, statuses)
        except Exception:
            return self._return_stale(statuses)

        self._last_ok = out
        self._last_ok_ts = time.time()
        return out

    def _read_statuses(self, conn) -> _Statuses:
        """Every rank's reporting status and the run-wide one.

        Best-effort: statuses that cannot be read are unavailable for this
        tick, ``(None, None)``. That never costs the panel its figures,
        and no earlier status stands in.
        """
        try:
            clock = read_rank_clock(
                self._db,
                conn,
                newest_ts=self._db.newest_sample_ts(conn),
                configured_interval_s=self._configured_interval_s,
            )
            return clock.reporting(), clock.run_reporting(self._now_fn())
        except Exception:
            return _UNREAD

    def _compute_impl(self, conn, statuses: _Statuses) -> Dict[str, Any]:
        committed_seq = self._db.fetch_committed_seq(conn)
        if committed_seq is None or committed_seq < 0:
            return self._empty_snapshot(statuses)

        rows = self._db.fetch_rows_for_seq_all_ranks(conn, committed_seq)
        if not rows:
            return self._empty_snapshot(statuses)

        cpu_used = max(float(r["cpu_percent"] or 0.0) for r in rows)

        gpu_rows = [
            r
            for r in rows
            if r["gpu_available"] == 1
            and r["gpu_mem_used_bytes"] is not None
            and r["gpu_mem_reserved_bytes"] is not None
            and r["gpu_mem_total_bytes"] is not None
        ]

        gpu_used = None
        gpu_reserved = None
        gpu_total = None
        gpu_rank = None
        gpu_used_imbalance = None

        if gpu_rows:

            def headroom(row) -> float:
                total = float(row["gpu_mem_total_bytes"] or 0.0)
                reserved = float(row["gpu_mem_reserved_bytes"] or 0.0)
                return total - reserved

            chosen = min(
                gpu_rows,
                key=lambda row: (
                    headroom(row),
                    int(row["rank"]) if row["rank"] is not None else 10**9,
                    int(row["id"]),
                ),
            )

            gpu_used = float(chosen["gpu_mem_used_bytes"] or 0.0)
            gpu_reserved = float(chosen["gpu_mem_reserved_bytes"] or 0.0)
            gpu_total = float(chosen["gpu_mem_total_bytes"] or 0.0)
            gpu_rank = (
                int(chosen["rank"]) if chosen["rank"] is not None else None
            )

            used_vals = [float(r["gpu_mem_used_bytes"]) for r in gpu_rows]
            gpu_used_imbalance = (
                float(max(used_vals) - min(used_vals))
                if len(used_vals) > 1
                else 0.0
            )

        return ProcessCLISnapshot(
            seq=int(committed_seq),
            cpu_used=float(cpu_used),
            gpu_used=gpu_used,
            gpu_reserved=gpu_reserved,
            gpu_total=gpu_total,
            gpu_rank=gpu_rank,
            gpu_used_imbalance=gpu_used_imbalance,
            rank_reporting=statuses[0],
            run_reporting=statuses[1],
        ).to_dict()

    def _return_stale(self, statuses: _Statuses) -> Dict[str, Any]:
        """The last good figures within the TTL, else an empty snapshot.

        Either way the reporting statuses are this tick's own read, or
        ``None`` when it failed. The held figures never bring back the
        statuses they were computed with.
        """
        now = time.time()
        if self._last_ok is not None:
            if (
                self._stale_ttl_s is None
                or (now - self._last_ok_ts) <= self._stale_ttl_s
            ):
                return {
                    **self._last_ok,
                    "rank_reporting": rank_reporting_dicts(statuses[0]),
                    "run_reporting": run_reporting_dict(statuses[1]),
                }
        return self._empty_snapshot(statuses)

    def _empty_snapshot(self, statuses: _Statuses = _UNREAD) -> Dict[str, Any]:
        return ProcessCLISnapshot(
            seq=None,
            cpu_used=0.0,
            gpu_used=None,
            gpu_reserved=None,
            gpu_total=None,
            gpu_rank=None,
            gpu_used_imbalance=None,
            rank_reporting=statuses[0],
            run_reporting=statuses[1],
        ).to_dict()
