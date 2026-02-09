"""
Layer-wise combined execution time computation (aggregator-side).

This module computes a **tail-latency oriented** timing view:
    total_time = forward_time + backward_time

Key semantics
-------------
- Aggregator-only (reads all ranks)
- Step-aligned across ranks using a safe-step join
- Uses worst-rank (max) semantics
- Produces a renderer-facing typed result
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from traceml.database.database import Database
from traceml.database.remote_database_store import RemoteDBStore
from traceml.loggers.error_log import get_error_logger
from traceml.renderers.layer_combined_time.schema import (
    LayerCombinedTimerOther,
    LayerCombinedTimerResult,
    LayerCombinedTimerRow,
)
from traceml.samplers.schema.layer_forward_backward_time import (
    LayerForwardBackwardTimeSample,
)
from traceml.transport.distributed import get_ddp_info


@dataclass(frozen=True)
class DDPJoinStatus:
    """
    Metadata describing step alignment across ranks.
    """

    safe_step: Optional[int]
    incomplete: bool
    missing_ranks: List[int]
    world_size: int


class LayerCombinedTimerData:
    """
    Computes combined forward + backward execution time per layer.

    Semantics
    ---------
    - current : worst-rank value at safe_step
    - avg     : EMA over current (UI smoothing)
    - peak    : worst-rank max over all steps <= safe_step
    """

    FORWARD_NAME = "LayerForwardTime"
    BACKWARD_NAME = "LayerBackwardTime"

    def __init__(
        self,
        remote_store: Optional[RemoteDBStore] = None,
        top_n_layers: int = 20,
        ema_alpha: float = 0.10,
    ) -> None:
        self._remote_store = remote_store
        self._top_n = int(top_n_layers)
        self._ema_alpha = float(ema_alpha)

        self._forward_cache: Dict[str, Dict[str, float]] = {}
        self._backward_cache: Dict[str, Dict[str, float]] = {}

        self._last_safe_step: int = -1
        self._join_status: Optional[DDPJoinStatus] = None
        self._worst_rank_by_layer: Dict[str, int] = {}

        self.logger = get_error_logger("LayerCombinedTimerData")

    def compute_display_data(self) -> LayerCombinedTimerResult:
        """
        Compute renderer-ready combined timing data.
        """
        world_size = self._world_size()
        safe_step, missing = self._compute_candidate_safe_step(world_size)

        if safe_step is None:
            return self._empty_result(world_size)

        fwd_snap, fwd_rank, fwd_missing = self._compute_step_snapshot(
            self.FORWARD_NAME, safe_step, world_size
        )
        bwd_snap, bwd_rank, bwd_missing = self._compute_step_snapshot(
            self.BACKWARD_NAME, safe_step, world_size
        )

        missing_ranks = sorted(set(fwd_missing) | set(bwd_missing))
        self._join_status = DDPJoinStatus(
            safe_step=safe_step,
            incomplete=bool(missing_ranks),
            missing_ranks=missing_ranks,
            world_size=world_size,
        )

        self._merge_cache(self._forward_cache, fwd_snap)
        self._merge_cache(self._backward_cache, bwd_snap)

        self._worst_rank_by_layer = self._compute_worst_ranks(
            fwd_rank, bwd_rank
        )

        return self._build_result()

    def _compute_candidate_safe_step(
        self, world_size: int
    ) -> Tuple[Optional[int], List[int]]:
        """
        Compute the largest step that *should* exist on all ranks.
        """
        steps: List[int] = []
        missing: List[int] = []

        for rank in range(world_size):
            fdb = self._get_db(rank, self.FORWARD_NAME)
            bdb = self._get_db(rank, self.BACKWARD_NAME)
            if not fdb or not bdb:
                missing.append(rank)
                continue

            sf = self._db_last_step(fdb)
            sb = self._db_last_step(bdb)
            if sf is None or sb is None:
                missing.append(rank)
                continue

            steps.append(min(sf, sb))

        return (min(steps), missing) if steps else (None, missing)

    def _compute_step_snapshot(
        self,
        sampler: str,
        step: int,
        world_size: int,
    ) -> Tuple[
        Dict[str, Dict[str, float]],
        Dict[str, Dict[int, float]],
        List[int],
    ]:
        """
        Compute per-layer current & peak timing up to `step`.
        """
        layer_current: Dict[str, float] = {}
        layer_peak: Dict[str, float] = {}
        layer_on_gpu: Dict[str, bool] = {}
        rank_curr: Dict[str, Dict[int, float]] = {}
        missing: List[int] = []

        for rank in range(world_size):
            db = self._get_db(rank, sampler)
            if not db:
                missing.append(rank)
                continue

            rows = next(iter(db.all_tables().values()), [])
            samples = self._load_samples_backwards(rows, step)

            cur_sample = next((s for s in samples if s.step == step), None)
            if not cur_sample:
                missing.append(rank)
                continue

            for name, cpu, gpu in zip(
                cur_sample.payload.layer_names,
                cur_sample.payload.cpu_time_ms,
                cur_sample.payload.gpu_time_ms,
            ):
                val = gpu if gpu is not None else cpu
                rank_curr.setdefault(name, {})[rank] = val
                layer_current[name] = max(layer_current.get(name, 0.0), val)
                layer_on_gpu[name] = gpu is not None

            for s in samples:
                for name, cpu, gpu in zip(
                    s.payload.layer_names,
                    s.payload.cpu_time_ms,
                    s.payload.gpu_time_ms,
                ):
                    val = gpu if gpu is not None else cpu
                    layer_peak[name] = max(layer_peak.get(name, 0.0), val)

        snapshot = {
            k: {
                "current": layer_current.get(k, 0.0),
                "peak": layer_peak.get(k, 0.0),
                "on_gpu": layer_on_gpu.get(k, False),
            }
            for k in set(layer_current) | set(layer_peak)
        }

        return snapshot, rank_curr, missing

    def _load_samples_backwards(
        self, rows: Iterable[dict], min_step: int
    ) -> List[LayerForwardBackwardTimeSample]:
        """
        Load time samples scanning backwards with early stop.
        """
        out: List[LayerForwardBackwardTimeSample] = []
        for r in reversed(rows or []):
            try:
                s = LayerForwardBackwardTimeSample.from_wire(r)
            except Exception:
                continue
            if s.step < min_step:
                break
            out.append(s)
        return out

    def _merge_cache(
        self,
        cache: Dict[str, Dict[str, float]],
        snapshot: Dict[str, Dict[str, float]],
    ) -> None:
        a = self._ema_alpha
        for layer, v in snapshot.items():
            cur = float(v["current"])
            peak = float(v["peak"])
            if layer not in cache:
                cache[layer] = {
                    "current": cur,
                    "avg": cur,
                    "peak": peak,
                    "on_gpu": v["on_gpu"],
                }
            else:
                cache[layer]["current"] = cur
                cache[layer]["avg"] = (1 - a) * cache[layer]["avg"] + a * cur
                cache[layer]["peak"] = max(cache[layer]["peak"], peak)
                cache[layer]["on_gpu"] |= v["on_gpu"]

    def _compute_worst_ranks(
        self,
        fwd: Dict[str, Dict[int, float]],
        bwd: Dict[str, Dict[int, float]],
    ) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for layer in set(fwd) | set(bwd):
            totals: Dict[int, float] = {}
            for r, v in fwd.get(layer, {}).items():
                totals[r] = totals.get(r, 0.0) + v
            for r, v in bwd.get(layer, {}).items():
                totals[r] = totals.get(r, 0.0) + v
            if totals:
                out[layer] = max(totals.items(), key=lambda x: x[1])[0]
        return out

    def _build_result(self) -> LayerCombinedTimerResult:
        join = self._join_status
        layers = set(self._forward_cache) | set(self._backward_cache)

        rows: List[LayerCombinedTimerRow] = []
        total_sum = 0.0

        for layer in layers:
            f = self._forward_cache.get(layer, {})
            b = self._backward_cache.get(layer, {})

            total_cur = f.get("current", 0.0) + b.get("current", 0.0)
            total_avg = f.get("avg", 0.0) + b.get("avg", 0.0)
            total_peak = f.get("peak", 0.0) + b.get("peak", 0.0)

            rows.append(
                LayerCombinedTimerRow(
                    layer=layer,
                    forward_current=f.get("current", 0.0),
                    forward_avg=f.get("avg", 0.0),
                    forward_peak=f.get("peak", 0.0),
                    backward_current=b.get("current", 0.0),
                    backward_avg=b.get("avg", 0.0),
                    backward_peak=b.get("peak", 0.0),
                    total_current=total_cur,
                    total_avg=total_avg,
                    total_peak=total_peak,
                    pct=0.0,
                    worst_rank=self._worst_rank_by_layer.get(layer),
                    on_gpu=f.get("on_gpu", b.get("on_gpu", False)),
                )
            )
            total_sum += total_cur

        rows_sorted = sorted(rows, key=lambda r: r.total_avg, reverse=True)

        out: List[LayerCombinedTimerRow] = []
        for r in rows_sorted:
            pct = (r.total_current / total_sum * 100.0) if total_sum else 0.0
            out.append(LayerCombinedTimerRow(**{**r.__dict__, "pct": pct}))

        top = out[: self._top_n]
        rest = out[self._top_n :]

        other = LayerCombinedTimerOther(
            total_forward_current=sum(r.forward_current for r in rest),
            total_forward_avg=sum(r.forward_avg for r in rest),
            total_backward_current=sum(r.backward_current for r in rest),
            total_backward_avg=sum(r.backward_avg for r in rest),
            pct=sum(r.pct for r in rest),
        )

        status = (
            f"Step snapshot incomplete at step={join.safe_step}. "
            f"Missing ranks: {join.missing_ranks}"
            if join and join.incomplete
            else "Timing snapshot ready."
        )

        return LayerCombinedTimerResult(
            top_items=top,
            all_items=out,
            other=other,
            safe_step=join.safe_step if join else None,
            incomplete=join.incomplete if join else False,
            missing_ranks=join.missing_ranks if join else [],
            world_size=join.world_size if join else self._world_size(),
            status_message=status,
        )

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
    def _empty_result(world_size: int) -> LayerCombinedTimerResult:
        return LayerCombinedTimerResult(
            top_items=[],
            all_items=[],
            other=LayerCombinedTimerOther(
                total_forward_current=0.0,
                total_forward_avg=0.0,
                total_backward_current=0.0,
                total_backward_avg=0.0,
                pct=0.0,
            ),
            safe_step=None,
            incomplete=True,
            missing_ranks=list(range(world_size)),
            world_size=world_size,
            status_message="Waiting for timing data from all ranks.",
        )


class LayerCombinedTimerSummary:
    def __init__(self, remote_store=None):
        pass
