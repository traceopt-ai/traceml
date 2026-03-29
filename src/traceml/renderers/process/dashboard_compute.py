"""
Dashboard compute for process telemetry.

This module computes the rolling UI history from SQLite.

Semantics
---------
- seq-aligned across ranks
- one dashboard entry per globally committed seq
- same history entry keys as the existing NiceGUI frontend expects
- top-level `gpu_used_imbalance` mirrors the latest history row for convenience
"""

import time
from collections import deque
from typing import Any, Deque, Dict, Optional

from .common import ProcessDashboardPayload, ProcessMetricsDB


class ProcessDashboardComputer:
    """
    Compute dashboard history for process telemetry UI.

    Parameters
    ----------
    db_path:
        Path to the SQLite database.
    dashboard_max_rows:
        Maximum retained rolling rows for UI history.
    stale_ttl_s:
        Maximum age in seconds for stale fallback reuse.
    """

    def __init__(
        self,
        db_path: str,
        dashboard_max_rows: int = 200,
        stale_ttl_s: Optional[float] = 30.0,
    ) -> None:
        self._db = ProcessMetricsDB(db_path=db_path)
        self._dashboard_rollup: Deque[Dict[str, Any]] = deque(
            maxlen=max(1, int(dashboard_max_rows))
        )
        self._last_completed_seq: int = -1

        self._last_ok: Optional[Dict[str, Any]] = None
        self._last_ok_ts: float = 0.0
        self._stale_ttl_s: Optional[float] = (
            float(stale_ttl_s) if stale_ttl_s is not None else None
        )

    def compute(self) -> Dict[str, Any]:
        """
        Advance and return dashboard history.

        Returns
        -------
        dict[str, Any]
            UI payload with:
            - `history`
            - `gpu_used_imbalance`
            - `series`
        """
        try:
            with self._db.connect() as conn:
                self._update_impl(conn)
                out = self._build_payload()
        except Exception:
            return self._return_stale()

        self._last_ok = out
        self._last_ok_ts = time.time()
        return out

    def _update_impl(self, conn) -> None:
        committed_upto = self._db.fetch_committed_seq(conn)
        if (
            committed_upto is None
            or committed_upto <= self._last_completed_seq
        ):
            return

        start_seq = self._last_completed_seq + 1
        end_seq = int(committed_upto)

        rows = self._db.fetch_seq_range_aggregates(
            conn,
            start_seq=start_seq,
            end_seq=end_seq,
        )

        for row in rows:
            entry: Dict[str, Any] = {
                "seq": int(row["seq"]),
                "ts": (
                    float(row["sample_ts_s"])
                    if row["sample_ts_s"] is not None
                    else None
                ),
                "cpu_max": float(row["cpu_max"] or 0.0),
                "ram_used_max": float(row["ram_used_max"] or 0.0),
                "ram_total": float(row["ram_total"] or 0.0),
            }

            if row["sample_ts_s"] is not None:
                entry["ts"] = float(row["sample_ts_s"])

            if row["gpu_used"] is not None:
                entry.update(
                    {
                        "gpu_used": float(row["gpu_used"] or 0.0),
                        "gpu_total": float(row["gpu_total"] or 0.0),
                        "gpu_headroom": float(row["gpu_headroom"] or 0.0),
                        "gpu_rank": (
                            int(row["gpu_rank"])
                            if row["gpu_rank"] is not None
                            else None
                        ),
                        "gpu_used_imbalance": (
                            float(row["gpu_used_imbalance"])
                            if row["gpu_used_imbalance"] is not None
                            else 0.0
                        ),
                    }
                )

            self._dashboard_rollup.append(entry)

        self._last_completed_seq = end_seq

    def _build_payload(self) -> Dict[str, Any]:
        history = list(self._dashboard_rollup)
        latest_imbalance = None
        if history:
            latest_imbalance = history[-1].get("gpu_used_imbalance")

        series = {
            "time_s": [
                float(r["ts"])
                for r in history
                if isinstance(r, dict) and r.get("ts") is not None
            ],
            "cpu_max": [
                float(r["cpu_max"])
                for r in history
                if isinstance(r, dict) and r.get("cpu_max") is not None
            ],
            "ram_used_max": [
                float(r["ram_used_max"])
                for r in history
                if isinstance(r, dict) and r.get("ram_used_max") is not None
            ],
            "gpu_used": [
                float(r["gpu_used"])
                for r in history
                if isinstance(r, dict) and r.get("gpu_used") is not None
            ],
        }

        return ProcessDashboardPayload(
            history=history,
            gpu_used_imbalance=latest_imbalance,
            series=series,
        ).to_dict()

    def _return_stale(self) -> Dict[str, Any]:
        now = time.time()
        if self._last_ok is not None:
            if (
                self._stale_ttl_s is None
                or (now - self._last_ok_ts) <= self._stale_ttl_s
            ):
                return self._last_ok

        return ProcessDashboardPayload(
            history=[],
            gpu_used_imbalance=None,
            series={
                "time_s": [],
                "cpu_max": [],
                "ram_used_max": [],
                "gpu_used": [],
            },
        ).to_dict()
