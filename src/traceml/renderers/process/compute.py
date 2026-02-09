"""
Process metrics computation layer.

This module contains pure aggregation and synchronization logic for
process-level telemetry collected during training.

Responsibilities
----------------
- Cross-rank synchronization using sequence numbers
- Pessimistic live snapshot computation (worst-rank semantics)
- Rolling, seq-aligned dashboard aggregation
- Summary statistics computation (rank-0 local view)

"""

from collections import deque
from typing import Any, Dict, List, Optional

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

        if not self._remote_store:
            return rank_dbs

        for rank in self._remote_store.ranks():
            db = self._remote_store.get_db(rank, self.SAMPLER_NAME)
            if db:
                rank_dbs[rank] = db

        return rank_dbs

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
            Empty dict if synchronization is not possible.
        """
        rank_dbs = self._collect_rank_dbs()
        if not rank_dbs:
            return {}

        # Determine last completed seq per rank
        last_seq_per_rank: Dict[int, int] = {}
        for rank, db in rank_dbs.items():
            table = db.create_or_get_table(self.TABLE_NAME)
            if not table:
                return {}
            last_seq_per_rank[rank] = table[-1].get("seq", -1)

        committed_seq = min(last_seq_per_rank.values())
        if committed_seq < 0:
            return {}

        # Fetch rows at committed seq
        rows_per_rank: Dict[int, Dict[str, Any]] = {}
        for rank, db in rank_dbs.items():
            table = db.create_or_get_table(self.TABLE_NAME)
            row = next(
                (r for r in reversed(table) if r.get("seq") == committed_seq),
                None,
            )
            if row is None:
                return {}
            rows_per_rank[rank] = row

        # Cross-rank aggregation
        cpu_vals: List[float] = []
        gpu_used: List[float] = []
        gpu_reserved: List[float] = []
        gpu_total: List[float] = []
        gpu_rank: List[int] = []

        for rank, row in rows_per_rank.items():
            cpu_vals.append(self._safe_float(row.get("cpu")))

            gpu = row.get("gpu")
            if row.get("gpu_available", False) and gpu:
                gpu_used.append(self._safe_float(gpu.get("mem_used")))
                gpu_reserved.append(self._safe_float(gpu.get("mem_reserved")))
                gpu_total.append(self._safe_float(gpu.get("mem_total")))
                gpu_rank.append(rank)

        snapshot = {
            "seq": committed_seq,
            "cpu_used": max(cpu_vals) if cpu_vals else 0.0,
        }

        if gpu_total:
            headrooms = [t - r for t, r in zip(gpu_total, gpu_reserved)]
            idx = int(np.argmin(headrooms))

            snapshot.update(
                {
                    "gpu_used": gpu_used[idx],
                    "gpu_reserved": gpu_reserved[idx],
                    "gpu_total": gpu_total[idx],
                    "gpu_rank": gpu_rank[idx],
                    "gpu_used_imbalance": (
                        max(gpu_used) - min(gpu_used)
                        if len(gpu_used) > 1
                        else 0.0
                    ),
                }
            )

        return snapshot

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
        """
        rank_dbs = self._collect_rank_dbs()
        if not rank_dbs:
            return

        last_seq_per_rank: Dict[int, int] = {}
        for rank, db in rank_dbs.items():
            table = db.create_or_get_table(self.TABLE_NAME)
            if not table:
                return
            last_seq_per_rank[rank] = table[-1].get("seq", -1)

        committed_upto = min(last_seq_per_rank.values())
        if committed_upto <= self._last_completed_seq:
            return

        for seq in range(self._last_completed_seq + 1, committed_upto + 1):
            rows_per_rank: Dict[int, Dict[str, Any]] = {}

            for rank, db in rank_dbs.items():
                table = db.create_or_get_table(self.TABLE_NAME)
                row = next(
                    (r for r in reversed(table) if r.get("seq") == seq),
                    None,
                )
                if row is None:
                    rows_per_rank = {}
                    break
                rows_per_rank[rank] = row

            if not rows_per_rank:
                continue

            cpu_vals = []
            ram_used_vals = []
            gpu_used_vals = []
            gpu_candidates = []
            ram_total = 0.0

            for rank, row in rows_per_rank.items():
                cpu_vals.append(self._safe_float(row.get("cpu")))

                ram_used = self._safe_float(row.get("ram_used"))
                ram_used_vals.append(ram_used)
                ram_total = max(
                    ram_total, self._safe_float(row.get("ram_total"))
                )

                gpu = row.get("gpu")
                if row.get("gpu_available", False) and gpu:
                    used = self._safe_float(gpu.get("mem_used"))
                    reserved = self._safe_float(gpu.get("mem_reserved"))
                    total = self._safe_float(gpu.get("mem_total"))
                    gpu_used_vals.append(used)
                    gpu_candidates.append(
                        (total - reserved, rank, used, total)
                    )

            entry = {
                "seq": seq,
                "cpu_max": max(cpu_vals) if cpu_vals else 0.0,
                "ram_used_max": max(ram_used_vals) if ram_used_vals else 0.0,
                "ram_total": ram_total,
            }

            if gpu_candidates:
                headroom, rank, used, total = min(gpu_candidates)
                entry.update(
                    {
                        "gpu_used": used,
                        "gpu_total": total,
                        "gpu_headroom": headroom,
                        "gpu_rank": rank,
                        "gpu_used_imbalance": (
                            max(gpu_used_vals) - min(gpu_used_vals)
                            if len(gpu_used_vals) > 1
                            else 0.0
                        ),
                    }
                )

            self._dashboard_rollup.append(entry)

        self._last_completed_seq = committed_upto

    def get_dashboard_history(self) -> List[Dict[str, Any]]:
        """
        Return the current dashboard history buffer.
        """
        return list(self._dashboard_rollup)

    def compute_summary(self, table: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute summary statistics from rank-0 local samples.

        Notes
        -----
        - This is intentionally *not* cross-rank.
        - Summary semantics will be revisited later.
        """
        if not table:
            return {"total_samples": 0}

        cpu_vals = [self._safe_float(r.get("cpu")) / 100.0 for r in table]
        ram_vals = [r.get("ram_used", 0.0) for r in table]

        gpu_used = []
        gpu_reserved = []
        gpu_total = []

        for r in table:
            gpu = r.get("gpu")
            if r.get("gpu_available", False) and gpu:
                gpu_used.append(self._safe_float(gpu.get("mem_used")))
                gpu_reserved.append(self._safe_float(gpu.get("mem_reserved")))
                gpu_total.append(self._safe_float(gpu.get("mem_total")))

        summary = {
            "total_samples": len(table),
            "cpu_cores_p50": float(np.median(cpu_vals)),
            "cpu_cores_p95": float(np.percentile(cpu_vals, 95)),
            "ram_used_p95": float(np.percentile(ram_vals, 95)),
            "ram_used_peak": float(np.max(ram_vals)),
            "ram_total": float(max(r.get("ram_total", 0.0) for r in table)),
            "is_GPU_available": bool(gpu_used),
        }

        if gpu_used:
            summary.update(
                {
                    "gpu_mem_used_p95_single": float(
                        np.percentile(gpu_used, 95)
                    ),
                    "gpu_mem_used_peak_single": float(np.max(gpu_used)),
                    "gpu_mem_reserved_peak_single": float(
                        np.max(gpu_reserved)
                    ),
                    "gpu_mem_total_capacity": float(np.max(gpu_total)),
                }
            )

        return summary
