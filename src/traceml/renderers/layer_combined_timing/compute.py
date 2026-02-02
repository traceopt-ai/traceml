from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from traceml.database.database import Database
from traceml.database.remote_database_store import RemoteDBStore
from traceml.transport.distributed import get_ddp_info
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
    Layer-wise combined timing service: forward + backward timing
    snapshots across DDP ranks.

    CORE SEMANTICS
    --------------
    - Each sampler emits **one row per step**, only after the step is complete.
    - Therefore: if step S exists in all ranks, step S is safe to render.

    DATA MODEL
    ----------
    Each timing row (per rank, per step):
      {
        "model_id": int,
        "step": int,
        "device": str,
        "layers": [
            (layer_name, cpu_ms, gpu_ms, n_calls)
        ]
      }

    AGGREGATION RULES
    -----------------
    - current: worst-rank (max) at safe_step
    - avg: EMA over current (renderer-facing smoothing)
    - peak: worst-rank max over all steps ≤ safe_step
    """

    FORWARD_NAME = "LayerForwardTime"
    BACKWARD_NAME = "LayerBackwardTime"

    def __init__(
        self,
        remote_store: Optional[RemoteDBStore] = None,
        top_n_layers: Optional[int] = 20,
        ema_alpha: float = 0.10,
    ) -> None:
        self._remote_store = remote_store
        self._top_n = int(top_n_layers) if top_n_layers is not None else 20
        self._ema_alpha = float(ema_alpha)

        # Per-layer caches: {"current": float, "avg": float, "peak": float, "on_gpu": bool}
        self._forward_cache: Dict[str, Dict[str, Any]] = {}
        self._backward_cache: Dict[str, Dict[str, Any]] = {}

        # Per-rank last-step watermarks
        self._last_safe_step: int = -1
        self._join_status: Optional[DDPJoinStatus] = None

        # Latest "worst rank" computed for each layer at the safe step
        self._worst_rank_by_layer: Dict[str, int] = {}
        self.logger = get_error_logger("LayerCombinedTimerData")


    def compute_display_data(self) -> Dict[str, Any]:
        """
        Compute renderer-ready timing data.

        Returns
        -------
        Dict[str, Any]
            Stable payload consumed by CLI / dashboard / notebook.
        """
        ws = self._world_size()
        safe_step, missing = self._compute_safe_step(ws)
        if safe_step is None:
            self._join_status = DDPJoinStatus(
                safe_step=None,
                incomplete=True,
                missing_ranks=list(range(ws)),
                world_size=ws,
            )
            return self._empty_payload()

        fwd_snapshot, fwd_rank_map = self._compute_step_snapshot(
            sampler=self.FORWARD_NAME, step=safe_step, world_size=ws
        )
        bwd_snapshot, bwd_rank_map = self._compute_step_snapshot(
            sampler=self.BACKWARD_NAME, step=safe_step, world_size=ws
        )
        self._join_status = DDPJoinStatus(
            safe_step=safe_step,
            incomplete=bool(missing),
            missing_ranks=missing,
            world_size=ws,
        )

        self._merge_cache(self._forward_cache, fwd_snapshot)
        self._merge_cache(self._backward_cache, bwd_snapshot)

        self._worst_rank_by_layer = self._compute_worst_ranks(
            fwd_rank_map, bwd_rank_map
        )

        return self._build_rows_payload()

    def _compute_safe_step(self, world_size: int) -> Tuple[Optional[int], List[int]]:
        """
        Compute the latest step S such that all ranks reported S.
        """
        steps: Dict[int, int] = {}
        missing: List[int] = []

        for rank in range(world_size):
            fdb = self._get_db(rank, self.FORWARD_NAME)
            bdb = self._get_db(rank, self.BACKWARD_NAME)

            if not fdb or not bdb:
                missing.append(rank)
                continue

            last_f = self._db_last_step(fdb)
            last_b = self._db_last_step(bdb)

            if last_f is None or last_b is None:
                missing.append(rank)
                continue

            steps[rank] = min(last_f, last_b)

        if not steps:
            return None, missing

        safe_step = min(steps.values())
        return safe_step, missing


    def _compute_step_snapshot(
        self,
        sampler: str,
        step: int,
        world_size: int,
    ) -> Tuple[
        Dict[str, Dict[str, float]],
        Dict[str, Dict[int, float]],
    ]:
        """
        Build per-layer snapshot for a specific step.
        """
        layer_current: Dict[str, float] = {}
        layer_peak: Dict[str, float] = {}
        layer_on_gpu: Dict[str, bool] = {}
        rank_curr: Dict[str, Dict[int, float]] = {}

        for rank in range(world_size):
            db = self._get_db(rank, sampler)
            if not db:
                continue

            rows = next(iter(db.all_tables().values()), [])
            row = self._row_at_step(rows, step)
            if not row:
                continue

            for layer, cpu_ms, gpu_ms, _ in row.get("layers", []):
                cur = gpu_ms if gpu_ms is not None else cpu_ms
                rank_curr.setdefault(layer, {})[rank] = cur
                layer_current[layer] = max(layer_current.get(layer, 0.0), cur)
                layer_on_gpu[layer] = gpu_ms is not None

            for r in rows:
                if r.get("step", -1) > step:
                    continue
                for layer, cpu_ms, gpu_ms, _ in r.get("layers", []):
                    d = gpu_ms if gpu_ms is not None else cpu_ms
                    layer_peak[layer] = max(layer_peak.get(layer, 0.0), d)

        snapshot = {
            layer: {
                "current": layer_current.get(layer, 0.0),
                "peak": layer_peak.get(layer, 0.0),
                "on_gpu": layer_on_gpu.get(layer, False),
            }
            for layer in set(layer_current) | set(layer_peak)
        }

        return snapshot, rank_curr


    def _merge_cache(
            self,
            cache: Dict[str, Dict[str, Any]],
            snapshot: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        Merge snapshot into EMA cache.
        """
        a = self._ema_alpha

        for layer, v in snapshot.items():
            cur = float(v["current"])
            peak = float(v["peak"])
            on_gpu = bool(v["on_gpu"])

            if layer not in cache:
                cache[layer] = {
                    "current": cur,
                    "avg": cur,
                    "peak": peak,
                    "on_gpu": on_gpu,
                }
            else:
                cache[layer]["current"] = cur
                cache[layer]["avg"] = (1 - a) * cache[layer]["avg"] + a * cur
                cache[layer]["peak"] = max(cache[layer]["peak"], peak)
                cache[layer]["on_gpu"] |= on_gpu


    def _compute_worst_ranks(
            self,
            fwd: Dict[str, Dict[int, float]],
            bwd: Dict[str, Dict[int, float]],
    ) -> Dict[str, int]:
        """
        Compute worst-rank per layer (fwd + bwd current).
        """
        out: Dict[str, int] = {}
        layers = set(fwd) | set(bwd)

        for layer in layers:
            totals: Dict[int, float] = {}
            for r, v in fwd.get(layer, {}).items():
                totals[r] = totals.get(r, 0.0) + v
            for r, v in bwd.get(layer, {}).items():
                totals[r] = totals.get(r, 0.0) + v
            if totals:
                out[layer] = max(totals.items(), key=lambda x: x[1])[0]
        return out


    def _build_rows_payload(self) -> Dict[str, Any]:
        join = self._join_status
        layers = set(self._forward_cache) | set(self._backward_cache)

        rows: List[Dict[str, Any]] = []
        for layer in layers:
            f = self._forward_cache.get(layer, {})
            b = self._backward_cache.get(layer, {})

            rows.append(
                {
                    "layer": layer,
                    "forward_current": f.get("current", 0.0),
                    "forward_avg": f.get("avg", 0.0),
                    "forward_peak": f.get("peak", 0.0),
                    "backward_current": b.get("current", 0.0),
                    "backward_avg": b.get("avg", 0.0),
                    "backward_peak": b.get("peak", 0.0),
                    "on_gpu": f.get("on_gpu", b.get("on_gpu", False)),
                    "worst_rank": self._worst_rank_by_layer.get(layer),
                }
            )

        rows_sorted = sorted(
            rows,
            key=lambda r: r["forward_avg"] + r["backward_avg"],
            reverse=True,
        )

        top = rows_sorted[: self._top_n]
        rest = rows_sorted[self._top_n:]

        total = sum(r["forward_current"] + r["backward_current"] for r in rows_sorted)
        for r in rows_sorted:
            r["pct"] = (
                (r["forward_current"] + r["backward_current"]) / total * 100.0
                if total > 0
                else 0.0
            )

        other = {
            "total_forward_current": sum(r["forward_current"] for r in rest),
            "total_forward_avg": sum(r["forward_avg"] for r in rest),
            "total_backward_current": sum(r["backward_current"] for r in rest),
            "total_backward_avg": sum(r["backward_avg"] for r in rest),
            "pct": (
                sum(r["pct"] for r in rest) if total > 0 else 0.0
            ),
        }

        record =  {
            "top_items": top,
            "all_items": rows_sorted,
            "other": other,
            "safe_step": join.safe_step if join else None,
            "incomplete": join.incomplete if join else False,
            "missing_ranks": join.missing_ranks if join else [],
            "world_size": join.world_size if join else self._world_size(),
        }
        return record


    def _world_size(self) -> int:
        _, _, ws = get_ddp_info()
        return max(int(ws), 1)

    def _get_db(self, rank: int, sampler: str) -> Optional[Database]:
        if not self._remote_store:
            return None
        return self._remote_store.get_db(rank, sampler + "Sampler")

    @staticmethod
    def _db_last_step(db: Database) -> Optional[int]:
        last = None
        for rows in db.all_tables().values():
            if rows:
                s = rows[-1].get("step")
                if s is not None:
                    last = s if last is None else max(last, s)
        return last

    @staticmethod
    def _row_at_step(rows, step: int) -> Optional[dict]:
        for r in reversed(rows):
            if r.get("step") == step:
                return r
            if r.get("step", -1) < step:
                break
        return None

    @staticmethod
    def _empty_payload() -> Dict[str, Any]:
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
            "safe_step": None,
            "incomplete": True,
            "missing_ranks": [],
            "world_size": 1,
        }



class LayerCombinedTimerSummary:
    """
    Computes **global timing statistics** across all available ranks.

    Scope
    -----
    - Aggregator-side only
    - Reads from RemoteDBStore
    - Best-effort (partial data is allowed)

    Data Model Assumption
    ---------------------
    Each timing row has the shape:
        {
            "step": int,
            "device": str,
            "layers": [
                (layer_name, cpu_ms, gpu_ms, n_calls)
            ]
        }

    Notes
    -----
    - This summary is *not step-aligned*.
    - It scans all available rows across all ranks.
    - Intended for `log_summary()`, not live dashboards.
    """

    FORWARD_NAME = "LayerForwardTime"
    BACKWARD_NAME = "LayerBackwardTime"

    def __init__(self, remote_store: Optional[RemoteDBStore]) -> None:
        self._remote_store = remote_store
        self.logger = get_error_logger("LayerCombinedTimerSummary")


    def compute_layer_timing_summary(self) -> Dict[str, Any]:
        """
        Compute coarse global timing statistics.

        Returns
        -------
        Dict[str, Any]
            {
                total_samples: int,
                total_layers_seen: int,
                avg_forward_ms: float,
                p50_forward_ms: float,
                p95_forward_ms: float,
                avg_backward_ms: float,
                p50_backward_ms: float,
                p95_backward_ms: float,
            }
        """
        fwd = self._compute_sampler_summary(self.FORWARD_NAME)
        bwd = self._compute_sampler_summary(self.BACKWARD_NAME)

        return {
            "total_samples": max(fwd["total_samples"], bwd["total_samples"]),
            "total_layers_seen": len(
                set(fwd["layers_seen"]) | set(bwd["layers_seen"])
            ),
            "avg_forward_ms": fwd["average"],
            "p50_forward_ms": fwd["p50_ms"],
            "p95_forward_ms": fwd["p95_ms"],
            "avg_backward_ms": bwd["average"],
            "p50_backward_ms": bwd["p50_ms"],
            "p95_backward_ms": bwd["p95_ms"],
        }

    def compute_global_averages(self, is_forward: bool) -> Dict[str, float]:
        """
        Compute average execution time per layer across all ranks & steps.

        Used for "Top-K slowest layers" summaries.
        """
        sampler = self.FORWARD_NAME if is_forward else self.BACKWARD_NAME
        per_layer: Dict[str, List[float]] = {}

        for rank, db in self._iter_rank_dbs(sampler):
            rows = self._get_rows(db)
            for row in rows:
                for layer, cpu_ms, gpu_ms, _ in row.get("layers", []):
                    d = gpu_ms if gpu_ms is not None else cpu_ms
                    if d is not None:
                        per_layer.setdefault(layer, []).append(float(d))

        return {
            layer: (sum(vals) / len(vals)) if vals else 0.0
            for layer, vals in per_layer.items()
        }

    @staticmethod
    def top_n_from_dict(d: Dict[str, float], n: int = 3):
        """
        Return top-N items from a dict sorted by descending value.
        """
        if not d:
            return []
        return sorted(d.items(), key=lambda kv: float(kv[1]), reverse=True)[:n]


    def _compute_sampler_summary(self, sampler: str) -> Dict[str, Any]:
        """
        Scan all rows for a given sampler and compute summary stats.
        """
        durations: List[float] = []
        layers_seen: set[str] = set()

        for rank, db in self._iter_rank_dbs(sampler):
            rows = self._get_rows(db)
            for row in rows:
                for layer, cpu_ms, gpu_ms, _ in row.get("layers", []):
                    layers_seen.add(layer)
                    d = gpu_ms if gpu_ms is not None else cpu_ms
                    if d is not None:
                        durations.append(float(d))

        if not durations:
            return {
                "total_samples": 0,
                "layers_seen": set(),
                "average": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
            }

        durations.sort()
        n = len(durations)

        return {
            "total_samples": n,
            "layers_seen": layers_seen,
            "average": sum(durations) / n,
            "p50_ms": durations[int(0.50 * (n - 1))],
            "p95_ms": durations[int(0.95 * (n - 1))],
        }

    def _iter_rank_dbs(self, sampler: str):
        """
        Yield (rank, db) pairs for all available ranks.
        """
        if not self._remote_store:
            return

        _, _, world_size = get_ddp_info()
        world_size = max(int(world_size), 1)

        for rank in range(world_size):
            try:
                db = self._remote_store.get_db(rank, sampler + "Sampler")
                if db is not None:
                    yield rank, db
            except Exception:
                self.logger.exception(
                    "Failed to fetch DB for rank",
                    extra={"rank": rank, "sampler": sampler},
                )

    @staticmethod
    def _get_rows(db: Database) -> List[Dict[str, Any]]:
        """
        Return all rows from the single-table timing DB.
        """
        if not db:
            return []
        tables = db.all_tables()
        if not tables:
            return []
        # Single-table invariant
        return next(iter(tables.values()), [])