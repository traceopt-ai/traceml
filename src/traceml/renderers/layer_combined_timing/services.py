from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from traceml.database.database import Database
from traceml.database.remote_database_store import RemoteDBStore
from traceml.distributed import get_ddp_info
from traceml.loggers.error_log import get_error_logger


@dataclass
class DDPJoinStatus:
    """
    DDP join / stall metadata for step-aligned rendering.

    Attributes:
        safe_step: Step index used for 'current' join. None if no safe step yet.
        incomplete: True if some ranks are missing data for safe_step.
        missing_ranks: Ranks missing safe_step data (best-effort).
        world_size: Inferred world size (best-effort).
    """

    safe_step: Optional[int]
    incomplete: bool
    missing_ranks: List[int]
    world_size: int


class LayerCombinedTimerData:
    """
    Layer-wise combined timing service: forward + backward.

    Works in:
      - single process (local DBs only)
      - DDP (rank-0 aggregation) when provided with RemoteDBStore

    Key timing nuance (vs memory):
      - Timing rows for step S may arrive in multiple flushes while step S is running.
      - Therefore we treat "presence of any rows for step S" as meaning step S-1 is complete.
      - We join on a *completed safe step*:
            safe_step = min_r(last_step_seen[r]) - 1
        and we verify that step exists in every rank's bounded DB.

    Per-layer outputs (joined across ranks by worst-rank max at safe_step):
      - forward_current, forward_avg, forward_peak
      - backward_current, backward_avg, backward_peak
      - pct: layer share of total current (forward_current + backward_current)
      - worst_rank: rank id that maximizes (forward_current + backward_current) at safe_step
        (best-effort; None when not available)

    Notes:
      - Peak here is "peak duration up to safe_step", not necessarily absolute run peak if DB drops history.
    """

    def __init__(
        self,
        forward_db: Optional[Database],
        backward_db: Optional[Database],
        top_n_layers: Optional[int] = 20,
        remote_store: Optional[RemoteDBStore] = None,
        ema_alpha: float = 0.10,
    ) -> None:
        self._forward_db = forward_db
        self._backward_db = backward_db
        self._remote_store = remote_store
        self._top_n = int(top_n_layers) if top_n_layers is not None else 20
        self._ema_alpha = float(ema_alpha)

        # Per-layer caches: {"current": float, "avg": float, "peak": float, "on_gpu": bool}
        self._forward_cache: Dict[str, Dict[str, Any]] = {}
        self._backward_cache: Dict[str, Dict[str, Any]] = {}

        # Per-rank last-step watermarks
        self._rank_last_step_fwd: Dict[int, int] = {}
        self._rank_last_step_bwd: Dict[int, int] = {}
        self._last_safe_step: int = -1
        self._join_status: Optional[DDPJoinStatus] = None

        # Latest "worst rank" computed for each layer at the safe step
        self._worst_rank_by_layer: Dict[str, int] = {}
        self.logger = get_error_logger("LayerCombinedTimerData")

    def compute_display_data(self) -> Dict[str, Any]:
        """
        Compute layer timing rows for renderers (CLI/notebook/dashboard).
        """
        if self._forward_db is None and self._backward_db is None:
            return self._empty_payload(safe_step=None)

        safe_step_candidate = self._compute_candidate_safe_step_completed()

        fwd_snapshot, fwd_ok, fwd_missing, fwd_rank_curr = (
            self._compute_step_aligned_snapshot(
                local_db=self._forward_db,
                is_forward=True,
                step=safe_step_candidate,
            )
        )
        bwd_snapshot, bwd_ok, bwd_missing, bwd_rank_curr = (
            self._compute_step_aligned_snapshot(
                local_db=self._backward_db,
                is_forward=False,
                step=safe_step_candidate,
            )
        )

        # If candidate safe step isn't fully available, fall back to last safe step.
        if safe_step_candidate >= 0 and fwd_ok and bwd_ok:
            self._last_safe_step = safe_step_candidate
            missing = sorted(set(fwd_missing) | set(bwd_missing))
            self._join_status = DDPJoinStatus(
                safe_step=self._last_safe_step,
                incomplete=(len(missing) > 0),
                missing_ranks=missing,
                world_size=self._infer_world_size(),
            )
        else:
            if self._last_safe_step >= 0:
                fwd_snapshot, _, fwd_missing, fwd_rank_curr = (
                    self._compute_step_aligned_snapshot(
                        local_db=self._forward_db,
                        is_forward=True,
                        step=self._last_safe_step,
                    )
                )
                bwd_snapshot, _, bwd_missing, bwd_rank_curr = (
                    self._compute_step_aligned_snapshot(
                        local_db=self._backward_db,
                        is_forward=False,
                        step=self._last_safe_step,
                    )
                )
                missing = sorted(set(fwd_missing) | set(bwd_missing))
                self._join_status = DDPJoinStatus(
                    safe_step=self._last_safe_step,
                    incomplete=(len(missing) > 0),
                    missing_ranks=missing,
                    world_size=self._infer_world_size(),
                )
            else:
                self._join_status = DDPJoinStatus(
                    safe_step=None,
                    incomplete=False,
                    missing_ranks=[],
                    world_size=self._infer_world_size(),
                )

        # Merge into caches
        self._merge_cache(
            self._forward_cache,
            fwd_snapshot,
            alpha=self._ema_alpha,
        )
        self._merge_cache(
            self._backward_cache,
            bwd_snapshot,
            alpha=self._ema_alpha,
        )

        # Compute worst-rank per layer from per-rank currents at this safe step
        self._worst_rank_by_layer = self._compute_worst_ranks(
            fwd_rank_curr=fwd_rank_curr,
            bwd_rank_curr=bwd_rank_curr,
        )

        return self._build_rows_payload()

    def _infer_world_size(self) -> int:
        try:
            _, _, ws = get_ddp_info()
            return max(int(ws), 1)
        except Exception:
            return 1

    def _compute_candidate_safe_step_completed(self) -> int:
        """
        Compute candidate safe *completed* step for timing joins.
        For timing, we treat watermark step S as "in-flight", so the newest completed step is S-1.
        """
        self._update_rank_last_step_cache(is_forward=True)
        self._update_rank_last_step_cache(is_forward=False)

        ws = self._infer_world_size()

        if ws <= 1:
            last_f = self._rank_last_step_fwd.get(0, -1)
            last_b = self._rank_last_step_bwd.get(0, -1)
            base = min(last_f, last_b)
            return (base - 1) if base >= 1 else -1

        f_min = min(self._rank_last_step_fwd.get(r, -1) for r in range(ws))
        b_min = min(self._rank_last_step_bwd.get(r, -1) for r in range(ws))
        base = min(f_min, b_min)
        return (base - 1) if base >= 1 else -1

    def _update_rank_last_step_cache(self, is_forward: bool) -> None:
        """
        Update per-rank last-step-seen caches using only tail rows (cheap).
        """
        local_db = self._forward_db if is_forward else self._backward_db
        if local_db is None:
            return

        cache = (
            self._rank_last_step_fwd
            if is_forward
            else self._rank_last_step_bwd
        )
        sampler_name = local_db.sampler_name

        for rank, db in self._iter_rank_dbs(
            local_db=local_db,
            sampler_name=sampler_name,
        ):
            s = self._db_last_step(db)
            if s is None:
                continue
            cache[rank] = max(cache.get(rank, -1), int(s))

        cache.setdefault(0, -1)

    def _iter_rank_dbs(
        self,
        local_db: Database,
        sampler_name: str,
    ) -> Iterable[Tuple[int, Database]]:
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
        local_db: Optional[Database],
        is_forward: bool,
        step: int,
    ) -> Tuple[
        Dict[str, Dict[str, Any]],
        bool,
        List[int],
        Dict[str, Dict[int, float]],
    ]:
        """
        Build a step-aligned snapshot at `step`.

        Returns:
            snapshot:
              {
                layer: {
                  "current_ms": float,   # worst-rank current at step
                  "peak_ms": float,      # worst-rank peak up to step
                  "on_gpu": bool,
                }
              }
            ok:
              True if all expected ranks have step `step` (best-effort).
            missing_ranks:
              ranks lacking step `step`.
            rank_curr_map:
              per-layer per-rank current_ms at exactly step:
                { layer: { rank: current_ms_at_step } }
              Used to compute worst-rank id by total (fwd+bwd).
        """
        if step < 0 or local_db is None:
            return {}, False, [], {}

        sampler_name = local_db.sampler_name
        ws = self._infer_world_size()
        expected_ranks = list(range(ws)) if ws > 1 else [0]

        layer_current_worst: Dict[str, float] = {}
        layer_peak_worst: Dict[str, float] = {}
        layer_on_gpu: Dict[str, bool] = {}
        rank_curr_map: Dict[str, Dict[int, float]] = {}

        missing_ranks: List[int] = []

        for rank in expected_ranks:
            db = (
                local_db
                if rank == 0
                else (
                    self._remote_store.get_db(rank, sampler_name)
                    if self._remote_store
                    else None
                )
            )
            if db is None:
                missing_ranks.append(rank)
                continue

            rank_has_step = False
            tables_iter = db.all_tables().items()

            for layer, rows in tables_iter:
                if not rows:
                    continue

                row = self._row_at_step(rows, step)
                if row is not None:
                    rank_has_step = True
                    cur, on_gpu = self._row_duration_ms(row)

                    # record per-rank current (for worst-rank id)
                    rank_curr_map.setdefault(layer, {})[rank] = cur

                    # aggregate "current" as worst-rank max
                    layer_current_worst[layer] = max(
                        layer_current_worst.get(layer, 0.0),
                        cur,
                    )
                    layer_on_gpu[layer] = bool(
                        layer_on_gpu.get(layer, False) or on_gpu,
                    )

                # aggregate peak up to step
                p = self._peak_upto_step(rows, step)
                layer_peak_worst[layer] = max(
                    layer_peak_worst.get(layer, 0.0),
                    p,
                )

            if ws > 1 and not rank_has_step:
                missing_ranks.append(rank)

        ok = len(missing_ranks) == 0

        snapshot = {
            layer: {
                "current_ms": float(layer_current_worst.get(layer, 0.0)),
                "peak_ms": float(layer_peak_worst.get(layer, 0.0)),
                "on_gpu": bool(layer_on_gpu.get(layer, False)),
            }
            for layer in (set(layer_current_worst) | set(layer_peak_worst))
        }

        return snapshot, ok, missing_ranks, rank_curr_map

    def _compute_worst_ranks(
        self,
        fwd_rank_curr: Dict[str, Dict[int, float]],
        bwd_rank_curr: Dict[str, Dict[int, float]],
    ) -> Dict[str, int]:
        """
        Compute worst-rank id per layer by total current (fwd + bwd) at safe step.

        Args:
            fwd_rank_curr: per-layer {rank -> forward_current_ms}
            bwd_rank_curr: per-layer {rank -> backward_current_ms}

        Returns:
            Dict[layer -> worst_rank_id]. Only includes layers where at least one rank is present.
        """
        worst: Dict[str, int] = {}
        layers = set(fwd_rank_curr.keys()) | set(bwd_rank_curr.keys())
        for layer in layers:
            per_rank_total: Dict[int, float] = {}

            for r, v in fwd_rank_curr.get(layer, {}).items():
                per_rank_total[int(r)] = per_rank_total.get(
                    int(r),
                    0.0,
                ) + float(v)
            for r, v in bwd_rank_curr.get(layer, {}).items():
                per_rank_total[int(r)] = per_rank_total.get(
                    int(r),
                    0.0,
                ) + float(v)

            if not per_rank_total:
                continue

            worst_rank = max(
                per_rank_total.items(),
                key=lambda kv: float(kv[1]),
            )[0]
            worst[layer] = int(worst_rank)

        return worst

    @staticmethod
    def _db_last_step(db: Database) -> Optional[int]:
        """
        Return the maximum last 'step' across all tables.

        Uses O(1) tail access via Database.get_last_record().
        Safe for deque-backed tables.
        """
        last_step: Optional[int] = None
        for table_name in db.all_tables().keys():
            last = db.get_last_record(table_name)
            if not last:
                continue
            s = last.get("step")
            if s is None:
                continue
            try:
                s_i = int(s)
            except Exception:
                continue
            last_step = s_i if last_step is None else max(last_step, s_i)
        return last_step

    @staticmethod
    def _row_at_step(rows, step: int) -> Optional[Mapping[str, Any]]:
        """
        Find row where row['step'] == step by scanning backwards.
        Works for list and deque.
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
                return None
        return None

    @staticmethod
    def _row_duration_ms(row: Mapping[str, Any]) -> Tuple[float, bool]:
        """
        Select duration from a timing row.

        Returns:
            (duration_ms, on_gpu)
        """
        on_gpu = bool(row.get("on_gpu", False))
        d = (
            row.get("gpu_duration_ms")
            if on_gpu
            else row.get("cpu_duration_ms")
        )
        if d is None:
            return 0.0, on_gpu
        try:
            return float(d), on_gpu
        except Exception:
            return 0.0, on_gpu

    @staticmethod
    def _peak_upto_step(rows, step: int) -> float:
        """
        Peak duration over steps <= step for a single table.
        """
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
            d, _ = LayerCombinedTimerData._row_duration_ms(r)
            peak = max(peak, d)
        return float(peak)

    @staticmethod
    def _merge_cache(
        cache: Dict[str, Dict[str, Any]],
        snapshot: Dict[str, Dict[str, Any]],
        alpha: float,
    ) -> None:
        """
        Merge snapshot into cache.

        Cache per layer:
          - current: overwritten
          - avg: EMA over current
          - peak: monotonic max (peak so far up to safe steps)
          - on_gpu: sticky OR
        """
        if not snapshot:
            return

        a = float(alpha)
        a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)

        for layer, entry in snapshot.items():
            cur = float(entry.get("current_ms", 0.0))
            peak = float(entry.get("peak_ms", 0.0))
            on_gpu = bool(entry.get("on_gpu", False))

            if layer not in cache:
                cache[layer] = {
                    "current": cur,
                    "avg": cur,
                    "peak": peak,
                    "on_gpu": on_gpu,
                }
            else:
                cache[layer]["current"] = cur
                cache[layer]["avg"] = (1.0 - a) * float(
                    cache[layer].get("avg", cur),
                ) + a * cur
                cache[layer]["peak"] = max(
                    float(cache[layer].get("peak", 0.0)),
                    peak,
                )
                cache[layer]["on_gpu"] = bool(
                    cache[layer].get("on_gpu", False) or on_gpu,
                )

    def _build_rows_payload(self) -> Dict[str, Any]:
        """
        Convert caches into table rows and aggregates.
        """
        join = self._join_status or DDPJoinStatus(
            safe_step=None,
            incomplete=False,
            missing_ranks=[],
            world_size=1,
        )
        layers = set(self._forward_cache.keys()) | set(
            self._backward_cache.keys(),
        )

        if not layers:
            return self._empty_payload(safe_step=join.safe_step)

        rows: List[Dict[str, Any]] = []
        for layer in layers:
            f = self._forward_cache.get(layer, {})
            b = self._backward_cache.get(layer, {})

            f_cur = float(f.get("current", 0.0))
            f_avg = float(f.get("avg", 0.0))
            f_peak = float(f.get("peak", 0.0))

            b_cur = float(b.get("current", 0.0))
            b_avg = float(b.get("avg", 0.0))
            b_peak = float(b.get("peak", 0.0))

            on_gpu = bool(f.get("on_gpu", b.get("on_gpu", False)))

            rows.append(
                {
                    "layer": layer,
                    "forward_current": f_cur,
                    "forward_avg": f_avg,
                    "forward_peak": f_peak,
                    "backward_current": b_cur,
                    "backward_avg": b_avg,
                    "backward_peak": b_peak,
                    "on_gpu": on_gpu,
                    "worst_rank": self._worst_rank_by_layer.get(layer),
                },
            )

        total_current_sum_ms = sum(
            (r["forward_current"] + r["backward_current"]) for r in rows
        )
        for r in rows:
            layer_total = r["forward_current"] + r["backward_current"]
            r["pct"] = (
                (layer_total / total_current_sum_ms * 100.0)
                if total_current_sum_ms > 0
                else 0.0
            )

        rows_sorted = sorted(
            rows,
            key=lambda r: float(r["forward_avg"]) + float(r["backward_avg"]),
            reverse=True,
        )

        top_items = rows_sorted[: self._top_n]
        other_items = rows_sorted[self._top_n :]

        other_f_cur = sum(r["forward_current"] for r in other_items)
        other_b_cur = sum(r["backward_current"] for r in other_items)

        other = {
            "total_forward_current": other_f_cur,
            "total_forward_avg": sum(r["forward_avg"] for r in other_items),
            "total_backward_current": other_b_cur,
            "total_backward_avg": sum(r["backward_avg"] for r in other_items),
            "pct": (
                (((other_f_cur + other_b_cur) / total_current_sum_ms) * 100.0)
                if total_current_sum_ms > 0
                else 0.0
            ),
        }

        return {
            "top_items": top_items,
            "all_items": rows_sorted,
            "other": other,
            "safe_step": join.safe_step,
            "incomplete": join.incomplete,
            "missing_ranks": join.missing_ranks,
            "world_size": join.world_size,
        }

    @staticmethod
    def _empty_payload(safe_step: Optional[int]) -> Dict[str, Any]:
        return {
            "top_items": [],
            "all_items": [],
            "other": {
                "total_forward_current": 0.0,
                "total_forward_avg": 0.0,
                "total_backward_current": 0.0,
                "total_backward_avg": 0.0,
                "pct": 0.0,
            },
            "safe_step": safe_step,
            "incomplete": False,
            "missing_ranks": [],
            "world_size": 1,
        }


class LayerCombinedTimerSummary:
    """
    Computes global summary stats for log_summary() from local DBs.

    This does not currently join remote ranks.
    """

    def __init__(
        self,
        forward_db: Optional[Database],
        backward_db: Optional[Database] = None,
    ) -> None:
        self._forward_db = forward_db
        self._backward_db = backward_db

    def compute_layer_timing_summary(self) -> Dict[str, Any]:
        fwd = self._compute_db_summary(self._forward_db)
        bwd = self._compute_db_summary(self._backward_db)

        total_samples = max(fwd["total_samples"], bwd["total_samples"])
        total_layers_seen = len(
            set(fwd["layers_seen"]) | set(bwd["layers_seen"]),
        )

        return {
            "total_samples": total_samples,
            "total_layers_seen": total_layers_seen,
            "avg_forward_ms": fwd["average"],
            "p50_forward_ms": fwd["p50_ms"],
            "p95_forward_ms": fwd["p95_ms"],
            "avg_backward_ms": bwd["average"],
            "p50_backward_ms": bwd["p50_ms"],
            "p95_backward_ms": bwd["p95_ms"],
        }

    def _compute_db_summary(self, db: Optional[Database]) -> Dict[str, Any]:
        if db is None:
            return {
                "total_samples": 0,
                "layers_seen": set(),
                "average": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
            }

        layers_seen = set()
        durations: List[float] = []

        tables_iter = (
            db.iter_tables()
            if hasattr(db, "iter_tables")
            else db.all_tables().items()
        )
        for layer_name, rows in tables_iter:
            if not rows:
                continue
            layers_seen.add(layer_name)
            for r in rows:
                d = self._pick_duration_ms(r)
                if d is not None:
                    durations.append(float(d))

        total = len(durations)
        if total:
            durations_sorted = sorted(durations)
            average = sum(durations_sorted) / total
            p50 = durations_sorted[int(0.50 * (total - 1))]
            p95 = durations_sorted[int(0.95 * (total - 1))]
        else:
            average = p50 = p95 = 0.0

        return {
            "total_samples": total,
            "layers_seen": layers_seen,
            "average": average,
            "p50_ms": p50,
            "p95_ms": p95,
        }

    @staticmethod
    def _pick_duration_ms(row: Mapping[str, Any]) -> Optional[float]:
        on_gpu = bool(row.get("on_gpu", False))
        d = (
            row.get("gpu_duration_ms")
            if on_gpu
            else row.get("cpu_duration_ms")
        )
        if d is None:
            return None
        try:
            return float(d)
        except Exception:
            return None

    def compute_global_averages(self, is_forward: bool) -> Dict[str, float]:
        db = self._forward_db if is_forward else self._backward_db
        if db is None:
            return {}

        avgs: Dict[str, float] = {}
        tables_iter = (
            db.iter_tables()
            if hasattr(db, "iter_tables")
            else db.all_tables().items()
        )

        for layer_name, rows in tables_iter:
            vals: List[float] = []
            for r in rows:
                d = self._pick_duration_ms(r)
                if d is not None:
                    vals.append(float(d))
            avgs[layer_name] = (sum(vals) / len(vals)) if vals else 0.0

        return avgs

    @staticmethod
    def top_n_from_dict(d: Dict[str, float], n: int = 3):
        if not d:
            return []
        return sorted(d.items(), key=lambda kv: float(kv[1]), reverse=True)[:n]
