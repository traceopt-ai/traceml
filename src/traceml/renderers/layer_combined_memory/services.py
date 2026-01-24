from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Deque

from traceml.database.database import Database
from traceml.database.remote_database_store import RemoteDBStore
from traceml.distributed import get_ddp_info
from traceml.loggers.error_log import get_error_logger


@dataclass
class DDPJoinStatus:
    """
    Metadata describing how well layer memory data is aligned across ranks.

    Attributes
    ----------
    safe_step : Optional[int]
        Step index used for the current snapshot.
    incomplete : bool
        True if at least one rank was missing data for `safe_step`.
    missing_ranks : List[int]
        Ranks that did not report data for `safe_step`.
    world_size : int
        Inferred number of ranks.
    """
    safe_step: Optional[int]
    incomplete: bool
    missing_ranks: List[int]
    world_size: int


class LayerCombinedMemoryData:
    """
    Computes **layer-wise combined memory**:
        param_memory + forward_activations + backward_activations

    This service is responsible for:
      - Aligning forward & backward memory across DDP ranks
      - Selecting a *safe step* completed by all ranks
      - Computing:
          * current memory (worst rank at safe step)
          * peak memory (worst rank across time up to safe step)
      - Producing a renderer-ready data structure

    IMPORTANT SEMANTICS
    -------------------
    - "current" memory is **layer-wise worst across ranks**
    - "peak" memory is **max of those worst values over time**
    - Rows do NOT correspond to a single rank
    - This is a *capacity / safety* view, not an execution trace

    Storage assumptions
    -------------------
    - layer_table contains static parameter memory snapshots
    - forward/backward DBs contain rows of the form:
        {
          "step": int,
          "layers": List[(layer_name, memory_bytes)],
          ...
        }
    - One table per sampler (no per-layer tables)
    """

    def __init__(
        self,
        layer_table: Deque[dict],
        layer_forward_db: Database,
        layer_backward_db: Database,
        top_n_layers: int = 20,
        remote_store: Optional[RemoteDBStore] = None,
    ):
        self._layer_table = layer_table
        self._layer_forward_db = layer_forward_db
        self._layer_backward_db = layer_backward_db
        self._top_n = int(top_n_layers)
        self._remote_store = remote_store

        # Cache structure:
        #   layer -> {"current": float, "global": float}
        self._forward_cache: Dict[str, Dict[str, float]] = {}
        self._backward_cache: Dict[str, Dict[str, float]] = {}

        self._last_safe_step: int = -1
        self._join_status: Optional[DDPJoinStatus] = None

        self.logger = get_error_logger("LayerCombinedMemoryData")


    def compute_display_data(self) -> Dict[str, Any]:
        """
        Compute all layer-wise memory metrics required by renderers.

        Returns
        -------
        Dict[str, Any]
            Renderer-ready structure containing:
              - per-layer current & peak memory
              - top-N layers
              - aggregated "other layers"
              - DDP join metadata
        """
        layer_snapshot = self._get_latest_layer_snapshot()
        param_layers = layer_snapshot["layer_memory"]
        model_index = layer_snapshot["model_index"]

        safe_step_candidate = self._compute_candidate_safe_step()

        fwd_snapshot, fwd_ok, fwd_missing = self._compute_step_snapshot(
            self._layer_forward_db, safe_step_candidate
        )
        bwd_snapshot, bwd_ok, bwd_missing = self._compute_step_snapshot(
            self._layer_backward_db, safe_step_candidate
        )

        if safe_step_candidate >= 0 and fwd_ok and bwd_ok:
            self._last_safe_step = safe_step_candidate
            missing = sorted(set(fwd_missing) | set(bwd_missing))
            self._join_status = self._build_join_status(
                safe_step_candidate, missing
            )
        elif self._last_safe_step >= 0:
            fwd_snapshot, _, fwd_missing = self._compute_step_snapshot(
                self._layer_forward_db, self._last_safe_step
            )
            bwd_snapshot, _, bwd_missing = self._compute_step_snapshot(
                self._layer_backward_db, self._last_safe_step
            )
            missing = sorted(set(fwd_missing) | set(bwd_missing))
            self._join_status = self._build_join_status(
                self._last_safe_step, missing
            )
        else:
            self._join_status = self._build_join_status(None, [])

        self._merge_cache(self._forward_cache, fwd_snapshot)
        self._merge_cache(self._backward_cache, bwd_snapshot)

        rows = self._build_rows(param_layers)
        rows_sorted = sorted(
            rows, key=lambda r: r["total_peak_memory"], reverse=True
        )

        top_items = rows_sorted[: self._top_n]
        other_items = rows_sorted[self._top_n :]

        total_current_sum = sum(r["total_current_memory"] for r in rows_sorted)

        other = self._aggregate_other(other_items, total_current_sum)

        join = self._join_status

        return {
            "model_index": model_index,
            "top_items": top_items,
            "other": other,
            "all_items": rows_sorted,
            "total_current_sum": total_current_sum,
            "total_peak_sum": sum(r["total_peak_memory"] for r in rows_sorted),
            "safe_step": join.safe_step if join else None,
            "incomplete": join.incomplete if join else False,
            "missing_ranks": join.missing_ranks if join else [],
            "world_size": join.world_size if join else 1,
        }


    def _get_latest_layer_snapshot(self) -> Dict[str, Any]:
        if not self._layer_table:
            return {"layer_memory": {}, "model_index": "—"}

        last = self._layer_table[-1]
        return {
            "layer_memory": last.get("layer_memory", {}),
            "model_index": last.get("model_index", "—"),
        }

    def _compute_candidate_safe_step(self) -> int:
        """
        Compute the largest step index that *should* be present on all ranks.

        This is a *candidate* only — availability is verified later.
        """
        _, _, world_size = get_ddp_info()
        world_size = max(int(world_size), 1)

        def last_step(db: Database) -> int:

            return max(
                (row.get("step", -1) for row in db.all_tables().values() for row in row),
                default=-1,
            )

        if world_size == 1:
            return min(
                last_step(self._layer_forward_db),
                last_step(self._layer_backward_db),
            )

        steps = []
        for rank in range(world_size):
            db_f = self._get_rank_db(self._layer_forward_db, rank)
            db_b = self._get_rank_db(self._layer_backward_db, rank)
            if not db_f or not db_b:
                return -1
            steps.append(min(last_step(db_f), last_step(db_b)))

        return min(steps)

    def _compute_step_snapshot(
        self,
        db: Database,
        step: int,
    ) -> Tuple[Dict[str, Dict[str, float]], bool, List[int]]:
        """
        Compute per-layer current & peak memory up to `step`.

        Returns
        -------
        snapshot : Dict[layer, {"current_peak", "global_peak"}]
        ok : bool
            True if all expected ranks had data for `step`
        missing_ranks : List[int]
        """
        if step < 0:
            return {}, False, []

        _, _, world_size = get_ddp_info()
        world_size = max(int(world_size), 1)

        layer_current: Dict[str, float] = {}
        layer_peak: Dict[str, float] = {}
        missing: List[int] = []

        for rank in range(world_size):
            rdb = self._get_rank_db(db, rank)
            if rdb is None:
                missing.append(rank)
                continue

            rows = next(iter(rdb.all_tables().values()), [])
            cur_row = self._row_at_step(rows, step)

            if cur_row:
                for layer, mem in cur_row.get("layers", []):
                    layer_current[layer] = max(layer_current.get(layer, 0.0), mem)

            for r in rows:
                if r.get("step", -1) > step:
                    continue
                for layer, mem in r.get("layers", []):
                    layer_peak[layer] = max(layer_peak.get(layer, 0.0), mem)

            if not cur_row:
                missing.append(rank)

        snapshot = {
            layer: {
                "current_peak": layer_current.get(layer, 0.0),
                "global_peak": layer_peak.get(layer, 0.0),
            }
            for layer in set(layer_current) | set(layer_peak)
        }

        return snapshot, not missing, missing


    def _get_rank_db(self, local_db: Database, rank: int) -> Optional[Database]:
        if rank == 0:
            return local_db
        if self._remote_store:
            return self._remote_store.get_db(rank, local_db.sampler_name)
        return None

    @staticmethod
    def _row_at_step(rows: Deque, step: int) -> Optional[dict]:
        for r in reversed(rows):
            if r.get("step") == step:
                return r
            if r.get("step", -1) < step:
                break
        return None

    @staticmethod
    def _merge_cache(cache: Dict[str, Dict[str, float]], snapshot: Dict[str, Dict[str, float]]):
        for layer, v in snapshot.items():
            if layer not in cache:
                cache[layer] = {"current": v["current_peak"], "global": v["global_peak"]}
            else:
                cache[layer]["current"] = v["current_peak"]
                cache[layer]["global"] = max(cache[layer]["global"], v["global_peak"])

    def _build_rows(self, param_layers: Dict[str, float]) -> List[Dict[str, Any]]:
        rows = []
        total_current_sum = 0.0

        for layer, param_mem in param_layers.items():
            fwd = self._forward_cache.get(layer, {})
            bwd = self._backward_cache.get(layer, {})

            current = param_mem + fwd.get("current", 0.0) + bwd.get("current", 0.0)
            peak = param_mem + fwd.get("global", 0.0) + bwd.get("global", 0.0)

            rows.append({
                "layer": layer,
                "param_memory": float(param_mem),
                "forward_current": float(fwd.get("current", 0.0)),
                "forward_peak": float(fwd.get("global", 0.0)),
                "backward_current": float(bwd.get("current", 0.0)),
                "backward_peak": float(bwd.get("global", 0.0)),
                "total_current_memory": float(current),
                "total_peak_memory": float(peak),
                "pct": 0.0,  # filled later
            })

            total_current_sum += current

        for r in rows:
            r["pct"] = (r["total_current_memory"] / total_current_sum * 100.0) if total_current_sum else 0.0

        return rows

    def _aggregate_other(self, rows: List[Dict[str, Any]], total_current_sum: float) -> Dict[str, Any]:
        cur = sum(r["total_current_memory"] for r in rows)
        return {
            "param_memory": sum(r["param_memory"] for r in rows),
            "forward_current": sum(r["forward_current"] for r in rows),
            "forward_peak": sum(r["forward_peak"] for r in rows),
            "backward_current": sum(r["backward_current"] for r in rows),
            "backward_peak": sum(r["backward_peak"] for r in rows),
            "total_current_memory": cur,
            "pct": (cur / total_current_sum * 100.0) if total_current_sum else 0.0,
        }

    def _build_join_status(self, step: Optional[int], missing: List[int]) -> DDPJoinStatus:
        _, _, world_size = get_ddp_info()
        return DDPJoinStatus(
            safe_step=step,
            incomplete=bool(missing),
            missing_ranks=missing,
            world_size=max(int(world_size), 1),
        )


class LayerCombinedMemorySummary:
    """
    Computes coarse, global statistics for logging and reports.

    This class intentionally avoids step alignment and DDP joins.
    It answers *historical* questions, not real-time ones.
    """

    def __init__(
        self,
        layer_table: Deque[dict],
        layer_forward_db: Database,
        layer_backward_db: Database,
    ):
        self._layer_table = layer_table
        self._layer_forward_db = layer_forward_db
        self._layer_backward_db = layer_backward_db

    def compute_layer_memory_summary(self) -> Dict[str, Any]:
        """
        Summarize static model memory usage.
        """
        if not self._layer_table:
            return {"total_models_seen": 0, "model_memory": 0.0}

        signatures = {r.get("model_signature") for r in self._layer_table}
        totals = [float(r.get("total_memory", 0.0)) for r in self._layer_table]

        return {
            "total_models_seen": len(signatures),
            "model_memory": sum(totals) / len(totals) if totals else 0.0,
        }

    def compute_global_peaks(self, is_forward: bool) -> Dict[str, float]:
        """
        Compute global (time-unbounded) peak memory per layer.
        """
        db = self._layer_forward_db if is_forward else self._layer_backward_db
        peaks: Dict[str, float] = {}

        for rows in db.all_tables().values():
            for r in rows:
                for layer, mem in r.get("layers", []):
                    peaks[layer] = max(peaks.get(layer, 0.0), mem)

        return peaks

    @staticmethod
    def top_n_from_dict(d: Dict[str, float], n: int = 3):
        return sorted(d.items(), key=lambda x: x[1], reverse=True)[:n] if d else []
