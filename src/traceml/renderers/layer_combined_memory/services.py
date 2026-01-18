from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Iterable, Deque
from traceml.database.database import Database
from traceml.distributed import get_ddp_info
from traceml.database.remote_database_store import RemoteDBStore
from traceml.loggers.error_log import get_error_logger

@dataclass
class RankStepStatus:
    """Per-rank step visibility for a given sampler."""
    last_step_seen: int = -1           # last step observed in DB (best-effort)
    has_step: bool = False             # whether the rank DB contains the chosen safe step


@dataclass
class DDPJoinStatus:
    """
    Metadata for DDP join / stall reporting.

    - safe_step: step at which we computed "current" across ranks (best-effort).
    - incomplete: True if not all ranks had data for safe_step.
    - missing_ranks: ranks that did not have data for safe_step.
    - world_size: inferred world size (best-effort).
    """
    safe_step: Optional[int]
    incomplete: bool
    missing_ranks: List[int]
    world_size: int


class LayerCombinedMemoryData:
    """
    Layer-wise combined memory service: params + forward (act) + backward (act).

    Supports both single GPU and DDP (rank-0 aggregation) with a RemoteDBStore.

    What we compute per layer:
      - param_memory: from LayerMemorySampler (rank-0 / local)
      - forward_current: "current" activation memory at a chosen safe step S
      - forward_peak: max activation memory over steps <= S (per rank), joined across ranks (max)
      - backward_current: same as forward_current but backward
      - backward_peak: same as forward_peak but backward
      - total_current_memory: param + forward_current + backward_current
      - total_peak_memory: param + forward_peak + backward_peak

    Step alignment and correctness:
      - RemoteDBStore is bounded, and remote ingest is asynchronous.
      - We align on a "safe step" S:
          S = min( min_r last_step_seen_fwd[r], min_r last_step_seen_bwd[r] )
        then we additionally verify that each rank actually contains step S
        in its bounded DB. If not, we do NOT advance; we keep rendering the
        last safe step that was successfully computed.

    Stall reporting:
      - We return join metadata (safe_step/incomplete/missing_ranks) so UI/CLI
        can show "waiting for ranks" instead of silently masking lag.
    """

    def __init__(
        self,
        layer_table,
        layer_forward_db: Database,
        layer_backward_db: Database,
        top_n_layers: Optional[int] = 20,
        remote_store: Optional[RemoteDBStore] = None,
    ):
        self._layer_table = layer_table
        self._layer_forward_db = layer_forward_db
        self._layer_backward_db = layer_backward_db
        self._top_n = top_n_layers if top_n_layers is not None else 20
        self._remote_store = remote_store

        # Cache: per-layer {current, global_peak}
        # - current: overwritten every refresh using safe-step snapshot
        # - global: monotonic max across safe-step snapshots (i.e., peak up to safe-step)
        self._forward_cache: Dict[str, Dict[str, float]] = {}
        self._backward_cache: Dict[str, Dict[str, float]] = {}

        # Service-local step caches
        self._rank_last_step_fwd: Dict[int, int] = {}
        self._rank_last_step_bwd: Dict[int, int] = {}
        self._last_safe_step: int = -1

        # Join status from the last compute()
        self._join_status: Optional[DDPJoinStatus] = None
        self.logger = get_error_logger("LayerCombinedMemoryData")


    def _infer_world_size(self) -> int:
        try:
            _, _, world_size = get_ddp_info()
            return max(int(world_size), 1)
        except Exception:
            return 1


    def compute_display_data(self) -> Dict[str, Any]:
        """
        Returns ALL per-layer data needed by dashboard, CLI, notebook.

        Sorting = total_peak_memory (param + forward_peak + backward_peak)
        Percent (%) = total_current_memory / sum(total_current_memory)
        """

        # Load table snapshot
        layer_snapshot = self._compute_layer_snapshot()
        layers = layer_snapshot.get("layer_memory", {}) or {}
        model_index = layer_snapshot.get("model_index", "—")

        # Compute the best next safe step and attempt to build safe-step snapshots.
        safe_step_candidate = self._compute_candidate_safe_step()

        fwd_snapshot, fwd_ok, fwd_missing = self._compute_step_aligned_snapshot(
            local_db=self._layer_forward_db,
            is_forward=True,
            step=safe_step_candidate,
        )
        bwd_snapshot, bwd_ok, bwd_missing = self._compute_step_aligned_snapshot(
            local_db=self._layer_backward_db,
            is_forward=False,
            step=safe_step_candidate,
        )


        # If candidate safe step is not fully available (bounded DB / ingest lag),
        # fall back to last known safe step.
        if safe_step_candidate >= 0 and fwd_ok and bwd_ok:
            self._last_safe_step = safe_step_candidate
            missing_ranks = sorted(set(fwd_missing) | set(bwd_missing))
            incomplete = len(missing_ranks) > 0
            self._join_status = DDPJoinStatus(
                safe_step=self._last_safe_step,
                incomplete=incomplete,
                missing_ranks=missing_ranks,
                world_size=self._infer_world_size(),
            )
        else:
            # fall back (render stable)
            if self._last_safe_step >= 0:
                fwd_snapshot, _, fwd_missing = self._compute_step_aligned_snapshot(
                    local_db=self._layer_forward_db,
                    is_forward=True,
                    step=self._last_safe_step,
                )
                bwd_snapshot, _, bwd_missing = self._compute_step_aligned_snapshot(
                    local_db=self._layer_backward_db,
                    is_forward=False,
                    step=self._last_safe_step,
                )
                missing_ranks = sorted(set(fwd_missing) | set(bwd_missing))
                incomplete = len(missing_ranks) > 0
                self._join_status = DDPJoinStatus(
                    safe_step=self._last_safe_step,
                    incomplete=incomplete,
                    missing_ranks=missing_ranks,
                    world_size=self._infer_world_size(),
                )
            else:
                # no data yet
                self._join_status = DDPJoinStatus(
                    safe_step=None,
                    incomplete=False,
                    missing_ranks=[],
                    world_size=self._infer_world_size(),
                )

        # Update global caches
        self._merge_cache(self._forward_cache, fwd_snapshot)
        self._merge_cache(self._backward_cache, bwd_snapshot)

        peak_map = {}
        current_map = {}

        # Compute peak & current totals (single pass)
        for layer, param_mem in layers.items():
            fwd_cur = self._forward_cache.get(layer, {}).get("current", 0.0)
            fwd_peak = self._forward_cache.get(layer, {}).get("global", 0.0)

            bwd_cur = self._backward_cache.get(layer, {}).get("current", 0.0)
            bwd_peak = self._backward_cache.get(layer, {}).get("global", 0.0)

            peak_map[layer] = float(param_mem) + float(fwd_peak) + float(bwd_peak)
            current_map[layer] = float(param_mem) + float(fwd_cur) + float(bwd_cur)

        total_current_sum = sum(current_map.values()) if current_map else 0.0

        all_rows = [
            self._build_layer_row(
                layer=layer,
                param_mem=layers.get(layer, 0.0),
                current_map=current_map,
                peak_map=peak_map,
                total_current_sum=total_current_sum,
            )
            for layer in layers.keys()
        ]

        # Sort by total peak memory
        all_rows_sorted = sorted(
            all_rows,
            key=lambda r: r["total_peak_memory"],
            reverse=True,
        )

        #  Split top / other
        top_items = all_rows_sorted[: self._top_n]
        other_items = all_rows_sorted[self._top_n :]

        #  Aggregate "other"
        other_current_total = (
            sum(r["total_current_memory"] for r in other_items) if other_items else 0.0
        )

        other = {
            "param_memory": sum(r["param_memory"] for r in other_items),
            "forward_current": sum(r["forward_current"] for r in other_items),
            "forward_peak": sum(r["forward_peak"] for r in other_items),
            "backward_current": sum(r["backward_current"] for r in other_items),
            "backward_peak": sum(r["backward_peak"] for r in other_items),
            "total_current_memory": other_current_total,
            "pct": (
                other_current_total / total_current_sum * 100.0
                if total_current_sum
                else 0.0
            ),
        }

        join = self._join_status or DDPJoinStatus(
            safe_step=None, incomplete=False, missing_ranks=[], world_size=1
        )

        return {
            "model_index": model_index,
            "top_items": top_items,
            "other": other,
            "all_items": all_rows_sorted,
            "total_current_sum": total_current_sum,
            "total_peak_sum": sum(peak_map.values()),
            "safe_step": join.safe_step,
            "incomplete": join.incomplete,
            "missing_ranks": join.missing_ranks,
            "world_size": join.world_size,
        }

    # Snapshot sources
    def _compute_layer_snapshot(self) -> Dict[str, Any]:
        if not self._layer_table:
            return {"layer_memory": {}, "model_index": "—"}

        last = self._layer_table[-1]
        return {
            "layer_memory": last.get("layer_memory", {}) or {},
            "model_index": last.get("model_index", "—"),
        }


    def _compute_candidate_safe_step(self) -> int:
        """
        Candidate safe step:
          S = min( min_r last_step_seen_fwd[r], min_r last_step_seen_bwd[r] )

        This step is only a *candidate*. We still verify that step S exists in each
        rank DB (bounded buffers might have dropped it).
        """
        self._update_rank_last_step_cache(is_forward=True)
        self._update_rank_last_step_cache(is_forward=False)

        ws = self._infer_world_size()

        # Single GPU
        if ws <= 1:
            return min(self._rank_last_step_fwd.get(0, -1), self._rank_last_step_bwd.get(0, -1))

        f_min = min(self._rank_last_step_fwd.get(r, -1) for r in range(ws))
        b_min = min(self._rank_last_step_bwd.get(r, -1) for r in range(ws))

        if f_min < 0 or b_min < 0:
            return -1

        return min(f_min, b_min)


    def _update_rank_last_step_cache(self, is_forward: bool) -> None:
        """
        Update last-step-seen caches using only last rows (cheap).
        """
        local_db = self._layer_forward_db if is_forward else self._layer_backward_db
        sampler_name = local_db.sampler_name
        cache = self._rank_last_step_fwd if is_forward else self._rank_last_step_bwd

        for rank, db in self._iter_rank_dbs(local_db, sampler_name):
            s = self._db_last_step(db)
            if s is None:
                continue
            cache[rank] = max(cache.get(rank, -1), int(s))

        # Ensure rank0 key exists for stable min() behavior in DDP
        cache.setdefault(0, -1)


    def _iter_rank_dbs(self, local_db: Database, sampler_name: str) -> Iterable[Tuple[int, Database]]:
        """
        Yield (rank, db) for local first, then remotes.
        """
        yield 0, local_db


        if not self._remote_store:
            return

        ws = self._infer_world_size()
        for rank in range(1, ws):
            db = self._remote_store.get_db(rank, sampler_name)
            if db is not None:
                yield rank, db



    def _compute_step_aligned_snapshot(
        self,
        local_db: Database,
        is_forward: bool,
        step: int,
    ) -> Tuple[Dict[str, Dict[str, float]], bool, List[int]]:
        """
        Build a snapshot aligned on a single step `step`.

        Returns:
          snapshot: {layer: {"current_peak": x, "global_peak": y}}
          ok: True if all expected ranks had data for this step
          missing_ranks: ranks lacking step `step` (best-effort)

        Semantics (per layer):
          - current_peak: memory at exactly step `step` (max over devices)
          - global_peak: max memory over steps <= `step` (max over devices)
        Join across ranks:
          - worst-rank wins (max)
        """
        if step < 0:
            return {}, False, []

        sampler_name = local_db.sampler_name
        ws = self._infer_world_size()

        layer_current: Dict[str, float] = {}
        layer_peak: Dict[str, float] = {}
        missing_ranks: List[int] = []

        expected_ranks = list(range(ws)) if ws > 1 else [0]

        for rank in expected_ranks:
            db = local_db if rank == 0 else (self._remote_store.get_db(rank, sampler_name) if self._remote_store else None)
            if db is None:
                missing_ranks.append(rank)
                continue

            # verify rank contains step `step` somewhere (any table) and compute aggregates
            rank_has_step = False

            for layer, rows in db.all_tables().items():
                if not rows:
                    continue

                row = self._row_at_step(rows, step)
                if row is not None:
                    rank_has_step = True
                    cur = self._row_max_device_memory(row)
                    layer_current[layer] = max(layer_current.get(layer, 0.0), cur)

                # peak up to step
                p = self._peak_upto_step(rows, step)
                layer_peak[layer] = max(layer_peak.get(layer, 0.0), p)

            if ws > 1 and not rank_has_step:
                missing_ranks.append(rank)

        ok = (len(missing_ranks) == 0)
        snapshot = {
            layer: {
                "current_peak": layer_current.get(layer, 0.0),
                "global_peak": layer_peak.get(layer, 0.0),
            }
            for layer in (set(layer_current) | set(layer_peak))
        }
        return snapshot, ok, missing_ranks


    @staticmethod
    def _db_last_step(db: Database) -> Optional[int]:
        last_step: Optional[int] = None

        for table_name in db.all_tables().keys():
            last = db.get_last_record(table_name)
            if last is None:
                continue
            s = last.get("step", None)
            if s is None:
                continue
            try:
                s_i = int(s)
            except Exception:
                continue
            last_step = s_i if last_step is None else max(last_step, s_i)

        return last_step


    @staticmethod
    def _row_at_step(rows: Deque, step: int) -> Optional[dict]:
        """
        Find the row with row['step'] == step by scanning from the end.
        Assumes rows are usually appended in increasing step order.
        """
        for r in reversed(rows):
            s = r.get("step", None)
            if s is None:
                continue
            try:
                s_i = int(s)
            except Exception:
                continue
            if s_i == step:
                return r
            if s_i < step:
                # Monotonic-ish: once below target, stop early.
                return None
        return None


    @staticmethod
    def _row_max_device_memory(row: dict) -> float:
        mem = row.get("memory", {}) or {}
        cur = 0.0
        for _, v in mem.items():
            try:
                cur = max(cur, float(v))
            except Exception:
                continue
        return float(cur)


    @staticmethod
    def _peak_upto_step(rows: Deque, step: int) -> float:
        peak = 0.0
        for r in rows:
            s = r.get("step", None)
            if s is None:
                continue
            try:
                s_i = int(s)
            except Exception:
                continue
            if s_i > step:
                continue
            mem = r.get("memory", {}) or {}
            for _, v in mem.items():
                try:
                    peak = max(peak, float(v))
                except Exception:
                    continue
        return float(peak)


    @staticmethod
    def _merge_cache(cache, new_data):
        """
        Merge snapshot into cache:
          - current: overwritten (latest safe-step)
          - global: max with previous (monotonic peak up to safe-step)
        """
        if not new_data:
            return
        for layer, entry in new_data.items():
            cur = entry.get("current_peak", 0.0)
            gbl = entry.get("global_peak", 0.0)
            if layer not in cache:
                cache[layer] = {"current": cur, "global": gbl}
            else:
                cache[layer]["current"] = cur
                cache[layer]["global"] = max(cache[layer]["global"], gbl)


    def _build_layer_row(
        self,
        layer: str,
        param_mem: float,
        current_map: Dict[str, float],
        peak_map: Dict[str, float],
        total_current_sum: float,
    ) -> Dict[str, Any]:

        fwd_cur = self._forward_cache.get(layer, {}).get("current", 0.0)
        fwd_peak = self._forward_cache.get(layer, {}).get("global", 0.0)
        bwd_cur = self._backward_cache.get(layer, {}).get("current", 0.0)
        bwd_peak = self._backward_cache.get(layer, {}).get("global", 0.0)

        current_total = current_map[layer]
        pct = (current_total / total_current_sum * 100.0) if total_current_sum else 0.0

        return {
            "layer": layer,
            "param_memory": float(param_mem),
            "forward_current": float(fwd_cur),
            "forward_peak": float(fwd_peak),
            "backward_current": float(bwd_cur),
            "backward_peak": float(bwd_peak),
            "total_peak_memory": float(peak_map[layer]),
            "total_current_memory": float(current_total),
            "pct": pct,
        }



class LayerCombinedMemorySummary:
    """
    Computes global statistics for log_summary():
      - total samples
      - #models
      - avg & peak memory
      - global forward / backward peaks per layer
    """

    def __init__(
        self,
        layer_table,
        layer_forward_db: Database,
        layer_backward_db: Database,
    ):
        self._layer_table = layer_table
        self._layer_forward_db = layer_forward_db
        self._layer_backward_db = layer_backward_db

    def compute_layer_memory_summary(self) -> Dict[str, Any]:
        if not self._layer_table:
            return {
                "total_models_seen": 0,
                "model_memory": 0.0,
            }

        model_signatures = {entry.get("model_signature") for entry in self._layer_table}

        totals = [float(entry.get("total_memory", 0.0)) for entry in self._layer_table]
        avg_memory = sum(totals) / len(totals) if totals else 0.0

        return {
            "total_models_seen": len(model_signatures),
            "model_memory": avg_memory,
        }

    # ------------------------------------------------------------------
    # Global peaks (for top-k lists)
    # ------------------------------------------------------------------

    def compute_global_peaks(self, is_forward: bool) -> Dict[str, float]:
        """
        Compute global peak per layer from forward/backward_db.
        Equivalent to previous _compute_peaks.
        """
        db = self._layer_forward_db if is_forward else self._layer_backward_db

        peaks: Dict[str, float] = {}
        for layer_name, rows in db.all_tables().items():
            peak = 0.0
            for r in rows:
                mem = r.get("memory", {}) or {}
                if mem:
                    peak = max(peak, max(float(v) for v in mem.values()))
            peaks[layer_name] = peak
        return peaks

    # ------------------------------------------------------------------
    # Helper: top-n from dict
    # ------------------------------------------------------------------

    @staticmethod
    def top_n_from_dict(d: Dict[str, float], n: int = 3):
        if not d:
            return []
        return sorted(d.items(), key=lambda kv: float(kv[1]), reverse=True)[:n]
