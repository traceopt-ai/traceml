"""
Process metrics computation layer.

Pure aggregation + synchronization logic for process-level telemetry.

Upgrades vs previous version (minimal behavioral change, big perf + stability):
- Faster seq lookup in deque tables (build per-rank seq->row map for the needed range)
  instead of scanning reversed(table) for every seq (major speedup).
- Avoids mutating DB on read path (uses get_table where possible).
- Adds last-good caching for:
    - live snapshot (so UI doesn't go blank on transient failure)
    - dashboard history (update failures do not wipe/stop the UI; history remains)
- Keeps output schema/keys the same as your current implementation.

NOTE: Caching returns the previous snapshot/history as-is (no new fields), to avoid
breaking any UI contract that expects specific keys.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from traceml.database.database import Database
from traceml.database.remote_database_store import RemoteDBStore


class ProcessMetricsComputer:
    """
    Compute process-level metrics across one or more ranks.

    Semantics
    ---------
    1. Live snapshot
       - Computed at the latest sequence completed by *all* active ranks
       - CPU: worst (max) across ranks
       - GPU: rank with least memory headroom

    2. Dashboard history
       - Incremental aggregation
       - Strictly seq-aligned
       - Bounded rolling window

    3. Summary
       - Computed from rank-0 local samples only
       - Intended for end-of-run reporting
    """

    TABLE_NAME = "ProcessTable"
    SAMPLER_NAME = "ProcessSampler"
    DASHBOARD_MAX_ROWS = 200

    def __init__(self, remote_store: Optional[RemoteDBStore]):
        self._remote_store = remote_store

        # Rolling dashboard aggregation (seq-aligned)
        self._dashboard_rollup: deque = deque(maxlen=self.DASHBOARD_MAX_ROWS)

        # Highest sequence number fully processed into dashboard
        self._last_completed_seq: int = -1

        # ---- stability cache ----
        self._last_ok_snapshot: Dict[str, Any] = {}
        self._last_ok_history: List[Dict[str, Any]] = []

    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default

    def _collect_rank_dbs(self) -> Dict[int, Database]:
        """
        Collect process databases for all active ranks.

        Returns
        -------
        Dict[int, Database]
            Mapping from rank -> Database instance.
        """
        rank_dbs: Dict[int, Database] = {}

        rs = self._remote_store
        if not rs:
            return rank_dbs

        # localize
        get_db = rs.get_db
        for rank in rs.ranks():
            db = get_db(rank, self.SAMPLER_NAME)
            if db:
                rank_dbs[rank] = db

        return rank_dbs

    # ------------------------------------------------------------------
    # Fast helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_table_readonly(db: Database, table_name: str):
        """
        Prefer read-only access to avoid mutating DB on compute paths.
        Falls back to create_or_get_table to preserve prior behavior if needed.
        """
        tbl = db.get_table(table_name)
        if tbl is not None:
            return tbl
        # preserve old behavior: ensures table exists, then returns it
        return db.create_or_get_table(table_name)

    @staticmethod
    def _last_seq_in_table(tbl) -> int:
        """
        tbl is a deque of dicts; last element is most recent.
        """
        try:
            if not tbl:
                return -1
            v = tbl[-1].get("seq", -1)
            return int(v) if v is not None else -1
        except Exception:
            return -1

    @staticmethod
    def _build_seq_row_map_for_range(tbl, start_seq: int, end_seq: int) -> Dict[int, Dict[str, Any]]:
        """
        Build seq->row map for a contiguous seq range [start_seq, end_seq] from a deque.

        Efficient approach:
        - scan from newest backwards
        - stop once seq < start_seq
        - record only seq <= end_seq
        """
        if not tbl or end_seq < start_seq:
            return {}

        out: Dict[int, Dict[str, Any]] = {}
        # scan from right end (newest first); tbl is a deque -> indexing near ends is cheap enough,
        # but reversed(tbl) is also fine. We'll use reversed() for clarity.
        for row in reversed(tbl):
            try:
                seq = row.get("seq", -1)
                if seq is None:
                    continue
                seq_i = int(seq)
            except Exception:
                continue

            if seq_i < start_seq:
                break
            if seq_i > end_seq:
                continue
            # keep first seen from the right (newest). For a given seq, there should be only one.
            if seq_i not in out:
                out[seq_i] = row

            # small micro-opt: if we already have all seqs, stop early
            if len(out) >= (end_seq - start_seq + 1):
                break

        return out

    @staticmethod
    def _find_row_by_seq(tbl, target_seq: int) -> Optional[Dict[str, Any]]:
        """
        Find row with seq == target_seq scanning from newest to oldest.
        Used for single-seq lookup (live snapshot).
        """
        if not tbl:
            return None
        for row in reversed(tbl):
            try:
                seq = row.get("seq", -1)
                if seq is None:
                    continue
                if int(seq) == target_seq:
                    return row
                if int(seq) < target_seq:
                    # since seq is increasing over time, once we pass below, stop
                    break
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    # Live snapshot
    # ------------------------------------------------------------------
    def compute_live_snapshot(self) -> Dict[str, Any]:
        """
        Compute a step-synchronized live snapshot.

        Returns
        -------
        Dict[str, Any]
            Snapshot at the latest seq completed by all ranks.
            Returns last-good snapshot on transient failure.
        """
        try:
            rank_dbs = self._collect_rank_dbs()
            if not rank_dbs:
                return self._last_ok_snapshot or {}

            # Determine last completed seq per rank
            last_seq_per_rank: Dict[int, int] = {}
            for rank, db in rank_dbs.items():
                table = self._get_table_readonly(db, self.TABLE_NAME)
                if not table:
                    return self._last_ok_snapshot or {}
                last_seq_per_rank[rank] = self._last_seq_in_table(table)

            committed_seq = min(last_seq_per_rank.values()) if last_seq_per_rank else -1
            if committed_seq < 0:
                return self._last_ok_snapshot or {}

            # Fetch rows at committed seq (one scan per rank)
            rows_per_rank: Dict[int, Dict[str, Any]] = {}
            for rank, db in rank_dbs.items():
                table = self._get_table_readonly(db, self.TABLE_NAME)
                row = self._find_row_by_seq(table, committed_seq)
                if row is None:
                    return self._last_ok_snapshot or {}
                rows_per_rank[rank] = row

            # Cross-rank aggregation
            cpu_max = 0.0
            gpu_used: List[float] = []
            gpu_reserved: List[float] = []
            gpu_total: List[float] = []
            gpu_rank: List[int] = []

            sf = self._safe_float
            for rank, row in rows_per_rank.items():
                c = sf(row.get("cpu"))
                if c > cpu_max:
                    cpu_max = c

                gpu = row.get("gpu")
                if row.get("gpu_available", False) and gpu:
                    gpu_used.append(sf(gpu.get("mem_used")))
                    gpu_reserved.append(sf(gpu.get("mem_reserved")))
                    gpu_total.append(sf(gpu.get("mem_total")))
                    gpu_rank.append(rank)

            snapshot: Dict[str, Any] = {
                "seq": committed_seq,
                "cpu_used": cpu_max,
            }

            if gpu_total:
                # headroom = total - reserved ; pick least headroom
                # numpy argmin is fine; list sizes are small
                headrooms = np.subtract(np.asarray(gpu_total, dtype=np.float64), np.asarray(gpu_reserved, dtype=np.float64))
                idx = int(np.argmin(headrooms))

                snapshot.update(
                    {
                        "gpu_used": float(gpu_used[idx]),
                        "gpu_reserved": float(gpu_reserved[idx]),
                        "gpu_total": float(gpu_total[idx]),
                        "gpu_rank": int(gpu_rank[idx]),
                        "gpu_used_imbalance": (
                            float(max(gpu_used) - min(gpu_used)) if len(gpu_used) > 1 else 0.0
                        ),
                    }
                )

            # success: cache
            self._last_ok_snapshot = snapshot
            return snapshot

        except Exception:
            # Never blank UI on exception
            return self._last_ok_snapshot or {}

    # ------------------------------------------------------------------
    # Dashboard aggregation
    # ------------------------------------------------------------------
    def update_dashboard(self) -> None:
        """
        Advance dashboard history using globally synchronized sequence numbers.

        Invariant
        ---------
        One dashboard entry corresponds to one sequence that *all* active
        ranks have completed.

        Stability
        ---------
        On transient failures, keep existing history (do not wipe).
        """
        try:
            rank_dbs = self._collect_rank_dbs()
            if not rank_dbs:
                return

            # Determine committed_upto (min last seq across ranks)
            last_seq_per_rank: Dict[int, int] = {}
            for rank, db in rank_dbs.items():
                table = self._get_table_readonly(db, self.TABLE_NAME)
                if not table:
                    return
                last_seq_per_rank[rank] = self._last_seq_in_table(table)

            committed_upto = min(last_seq_per_rank.values()) if last_seq_per_rank else -1
            if committed_upto <= self._last_completed_seq:
                return

            start_seq = self._last_completed_seq + 1
            end_seq = committed_upto

            # Build per-rank seq->row maps ONCE for the needed range (major speedup)
            per_rank_maps: Dict[int, Dict[int, Dict[str, Any]]] = {}
            for rank, db in rank_dbs.items():
                table = self._get_table_readonly(db, self.TABLE_NAME)
                per_rank_maps[rank] = self._build_seq_row_map_for_range(table, start_seq, end_seq)

            sf = self._safe_float

            for seq in range(start_seq, end_seq + 1):
                rows_per_rank: Dict[int, Dict[str, Any]] = {}
                ok = True
                for rank in rank_dbs.keys():
                    row = per_rank_maps.get(rank, {}).get(seq)
                    if row is None:
                        ok = False
                        break
                    rows_per_rank[rank] = row
                if not ok:
                    continue

                cpu_max = 0.0
                ram_used_max = 0.0
                ram_total = 0.0

                gpu_used_vals: List[float] = []
                gpu_candidates: List[Tuple[float, int, float, float]] = []  # (headroom, rank, used, total)

                for rank, row in rows_per_rank.items():
                    c = sf(row.get("cpu"))
                    if c > cpu_max:
                        cpu_max = c

                    ru = sf(row.get("ram_used"))
                    if ru > ram_used_max:
                        ram_used_max = ru

                    rt = sf(row.get("ram_total"))
                    if rt > ram_total:
                        ram_total = rt

                    gpu = row.get("gpu")
                    if row.get("gpu_available", False) and gpu:
                        used = sf(gpu.get("mem_used"))
                        reserved = sf(gpu.get("mem_reserved"))
                        total = sf(gpu.get("mem_total"))
                        gpu_used_vals.append(used)
                        gpu_candidates.append((total - reserved, rank, used, total))

                entry: Dict[str, Any] = {
                    "seq": seq,
                    "cpu_max": cpu_max,
                    "ram_used_max": ram_used_max,
                    "ram_total": ram_total,
                }

                if gpu_candidates:
                    headroom, rank_min, used_min, total_min = min(gpu_candidates, key=lambda x: x[0])
                    entry.update(
                        {
                            "gpu_used": float(used_min),
                            "gpu_total": float(total_min),
                            "gpu_headroom": float(headroom),
                            "gpu_rank": int(rank_min),
                            "gpu_used_imbalance": (
                                float(max(gpu_used_vals) - min(gpu_used_vals)) if len(gpu_used_vals) > 1 else 0.0
                            ),
                        }
                    )

                self._dashboard_rollup.append(entry)

            self._last_completed_seq = committed_upto
            # cache last-good history for UI fallback
            self._last_ok_history = list(self._dashboard_rollup)

        except Exception:
            # Keep existing dashboard state on failure
            return

    def get_dashboard_history(self) -> List[Dict[str, Any]]:
        """
        Return the current dashboard history buffer.
        On rare issues, return last-good history rather than blank.
        """
        try:
            hist = list(self._dashboard_rollup)
            if hist:
                self._last_ok_history = hist
                return hist
            return self._last_ok_history
        except Exception:
            return self._last_ok_history

    # ------------------------------------------------------------------
    # Summary (rank-0 local only)
    # ------------------------------------------------------------------
    def compute_summary(self, table: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute summary statistics from rank-0 local samples.

        Notes
        -----
        - This is intentionally *not* cross-rank.
        """
        if not table:
            return {"total_samples": 0}

        sf = self._safe_float

        # numpy-friendly arrays (fast percentiles)
        cpu_vals = np.fromiter((sf(r.get("cpu")) / 100.0 for r in table), dtype=np.float64)
        ram_vals = np.fromiter((sf(r.get("ram_used")) for r in table), dtype=np.float64)

        gpu_used: List[float] = []
        gpu_reserved: List[float] = []
        gpu_total: List[float] = []

        for r in table:
            gpu = r.get("gpu")
            if r.get("gpu_available", False) and gpu:
                gpu_used.append(sf(gpu.get("mem_used")))
                gpu_reserved.append(sf(gpu.get("mem_reserved")))
                gpu_total.append(sf(gpu.get("mem_total")))

        summary: Dict[str, Any] = {
            "total_samples": int(len(table)),
            "cpu_cores_p50": float(np.median(cpu_vals)) if cpu_vals.size else 0.0,
            "cpu_cores_p95": float(np.percentile(cpu_vals, 95)) if cpu_vals.size else 0.0,
            "ram_used_p95": float(np.percentile(ram_vals, 95)) if ram_vals.size else 0.0,
            "ram_used_peak": float(np.max(ram_vals)) if ram_vals.size else 0.0,
            "ram_total": float(max(sf(r.get("ram_total")) for r in table)),
            "is_GPU_available": bool(gpu_used),
        }

        if gpu_used:
            gu = np.asarray(gpu_used, dtype=np.float64)
            gr = np.asarray(gpu_reserved, dtype=np.float64)
            gt = np.asarray(gpu_total, dtype=np.float64)
            summary.update(
                {
                    "gpu_mem_used_p95_single": float(np.percentile(gu, 95)),
                    "gpu_mem_used_peak_single": float(np.max(gu)),
                    "gpu_mem_reserved_peak_single": float(np.max(gr)),
                    "gpu_mem_total_capacity": float(np.max(gt)),
                }
            )

        return summary