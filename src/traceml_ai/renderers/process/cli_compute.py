"""
CLI compute for process telemetry.

This module computes a terminal-oriented live snapshot from SQLite.

Semantics
---------
- latest globally committed seq only
- cross-rank aggregation
- output keys match the current terminal renderer expectations
- per-rank last-seen and freshness, judged as the dashboard judges them
"""

import time
from typing import Any, Callable, Dict, Optional, Tuple

from traceml_ai.renderers.shared.freshness import (
    CachedPayloadTTL,
    LastGoodVerdict,
    RankLiveness,
)

from .common import ProcessCLISnapshot
from .liveness import read_rank_clock
from .repository import ProcessRepository


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
        The clock that times how long the last good rank verdicts may
        answer for a heartbeat read that failed.
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
        # Injectable so the last good verdicts' expiry can be tested at a
        # chosen moment rather than by sleeping.
        self._now_fn = now_fn
        self._last_ok: Optional[Dict[str, Any]] = None
        self._last_ok_ts: float = 0.0
        self._stale_ttl_s: Optional[float] = (
            float(stale_ttl_s) if stale_ttl_s is not None else None
        )
        self._liveness: LastGoodVerdict[Tuple[RankLiveness, ...]] = (
            LastGoodVerdict(CachedPayloadTTL(ttl_s=self._stale_ttl_s))
        )

    def compute(self) -> Dict[str, Any]:
        """
        Compute the latest live snapshot.

        Returns
        -------
        dict[str, Any]
            Terminal-facing snapshot. On transient failure, returns the previous
            good snapshot if still within stale TTL.
        """
        try:
            with self._db.connect() as conn:
                out = self._compute_impl(conn)
        except Exception:
            return self._return_stale()

        self._last_ok = out
        self._last_ok_ts = time.time()
        return out

    def _read_liveness(self, conn) -> Optional[Tuple[RankLiveness, ...]]:
        """Every rank's last-seen clock, or ``None`` when there is none.

        Best-effort: a heartbeat that cannot be read costs the verdicts,
        never the panel's figures. The last good verdicts answer for it
        within the stale TTL; after that, ``None``.
        """
        now_s = self._now_fn()
        try:
            read = read_rank_clock(
                self._db,
                conn,
                newest_ts=self._db.newest_sample_ts(conn),
                configured_interval_s=self._configured_interval_s,
            ).liveness()
        except Exception:
            read = None
        return self._liveness.carry(read, now_s=now_s)

    def _compute_impl(self, conn) -> Dict[str, Any]:
        liveness = self._read_liveness(conn)

        committed_seq = self._db.fetch_committed_seq(conn)
        if committed_seq is None or committed_seq < 0:
            return self._empty_snapshot(liveness)

        rows = self._db.fetch_rows_for_seq_all_ranks(conn, committed_seq)
        if not rows:
            return self._empty_snapshot(liveness)

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
            rank_liveness=liveness,
        ).to_dict()

    def _return_stale(self) -> Dict[str, Any]:
        now = time.time()
        if self._last_ok is not None:
            if (
                self._stale_ttl_s is None
                or (now - self._last_ok_ts) <= self._stale_ttl_s
            ):
                return self._last_ok
        return self._empty_snapshot()

    def _empty_snapshot(
        self, liveness: Optional[Tuple[RankLiveness, ...]] = None
    ) -> Dict[str, Any]:
        return ProcessCLISnapshot(
            seq=None,
            cpu_used=0.0,
            gpu_used=None,
            gpu_reserved=None,
            gpu_total=None,
            gpu_rank=None,
            gpu_used_imbalance=None,
            rank_liveness=liveness,
        ).to_dict()
