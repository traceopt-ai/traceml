"""Live compute for the optional deep layer-memory profiler."""

from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from traceml.database.database import Database
from traceml.database.remote_database_store import RemoteDBStore
from traceml.loggers.error_log import get_error_logger
from traceml.renderers.layer_combined_memory.schema import (
    LayerCombinedMemoryResult,
    LayerCombinedMemoryRow,
    LayerCombinedOther,
)
from traceml.samplers.schema.layer_forward_backward_memory import (
    LayerForwardBackwardMemorySample,
)
from traceml.samplers.schema.layer_memory import LayerMemorySample
from traceml.transport.distributed import get_ddp_info


@dataclass(frozen=True)
class DDPJoinStatus:
    """Alignment state for the deep layer-memory live snapshot."""

    safe_step: Optional[int]
    incomplete: bool
    missing_ranks: List[int]
    world_size: int


@dataclass(frozen=True)
class ModelSnapshotStatus:
    """Model snapshot availability across ranks."""

    ready: bool
    canonical_rank: Optional[int]
    canonical_model_index: Optional[int]
    canonical_layer_memory: Dict[str, float]
    missing_ranks: List[int]
    mismatched_ranks: List[int]
    world_size: int

    def message(self) -> str:
        if self.ready:
            return "Model snapshot ready (all ranks agree)."
        if self.missing_ranks and not self.mismatched_ranks:
            return (
                f"Waiting for model snapshot from ranks: {self.missing_ranks}"
            )
        if self.mismatched_ranks:
            return (
                "Model snapshot mismatch across ranks. "
                f"Mismatched ranks: {self.mismatched_ranks}"
            )
        return "Model snapshot not available yet."


class LayerCombinedMemoryData:
    """
    Compute the live deep-profile layer memory view.

    This class is intentionally scoped to the ``deep`` profile. It joins static
    parameter memory with forward/backward activation samples and uses the worst
    rank for a capacity-oriented view.
    """

    LAYER_MEMORY_NAME = "LayerMemory"
    LAYER_FORWARD_NAME = "LayerForwardMemory"
    LAYER_BACKWARD_NAME = "LayerBackwardMemory"

    def __init__(
        self,
        remote_store: Optional[RemoteDBStore] = None,
        top_n_layers: int = 20,
    ):
        self._remote_store = remote_store
        self._top_n = int(top_n_layers)

        self._forward_cache: Dict[str, Dict[str, float]] = {}
        self._backward_cache: Dict[str, Dict[str, float]] = {}

        self._last_safe_step: int = -1
        self._join_status: Optional[DDPJoinStatus] = None

        self.logger = get_error_logger("LayerCombinedMemoryData")

    def _load_samples_backwards(
        self,
        rows: Iterable[dict],
        sample_cls,
        min_step: int,
    ):
        """Load schema objects from recent wire rows, newest first."""
        out = []
        for r in reversed(rows or []):
            try:
                s = sample_cls.from_wire(r)
            except Exception:
                continue
            if s.step < min_step:
                break
            out.append(s)
        return out

    def compute_display_data(self) -> LayerCombinedMemoryResult:
        """Return the typed payload consumed by CLI and dashboard renderers."""
        model_status = self._get_model_snapshot_status()
        world_size = model_status.world_size

        if not model_status.ready:
            self._join_status = self._build_join_status(
                None, model_status.missing_ranks
            )
            return LayerCombinedMemoryResult(
                model_index=model_status.canonical_model_index,
                top_items=[],
                other=LayerCombinedOther(
                    param_memory=0.0,
                    forward_current=0.0,
                    forward_peak=0.0,
                    backward_current=0.0,
                    backward_peak=0.0,
                    total_current_memory=0.0,
                    pct=0.0,
                ),
                all_items=[],
                total_current_sum=0.0,
                total_peak_sum=0.0,
                safe_step=None,
                incomplete=True,
                missing_ranks=sorted(
                    set(model_status.missing_ranks)
                    | set(model_status.mismatched_ranks)
                ),
                world_size=world_size,
                status_message=model_status.message(),
            )

        param_layers = model_status.canonical_layer_memory
        model_index = model_status.canonical_model_index

        safe_step_candidate = self._compute_candidate_safe_step(world_size)

        fwd_snapshot, fwd_ok, fwd_missing = self._compute_step_snapshot(
            sampler_name=self.LAYER_FORWARD_NAME + "Sampler",
            step=safe_step_candidate,
            world_size=world_size,
        )
        bwd_snapshot, bwd_ok, bwd_missing = self._compute_step_snapshot(
            sampler_name=self.LAYER_BACKWARD_NAME + "Sampler",
            step=safe_step_candidate,
            world_size=world_size,
        )

        if safe_step_candidate >= 0 and fwd_ok and bwd_ok:
            self._last_safe_step = safe_step_candidate
            missing = sorted(set(fwd_missing) | set(bwd_missing))
            self._join_status = self._build_join_status(
                safe_step_candidate, missing
            )
        elif self._last_safe_step >= 0:
            # Keep deep UI stable during transient rank gaps.
            fwd_snapshot, _, fwd_missing = self._compute_step_snapshot(
                sampler_name=self.LAYER_FORWARD_NAME + "Sampler",
                step=self._last_safe_step,
                world_size=world_size,
            )
            bwd_snapshot, _, bwd_missing = self._compute_step_snapshot(
                sampler_name=self.LAYER_BACKWARD_NAME + "Sampler",
                step=self._last_safe_step,
                world_size=world_size,
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
            rows, key=lambda r: r.total_peak_memory, reverse=True
        )

        top_items = rows_sorted[: self._top_n]
        other_items = rows_sorted[self._top_n :]

        total_current_sum = sum(r.total_current_memory for r in rows_sorted)
        other = self._aggregate_other(other_items, total_current_sum)

        join = self._join_status

        status_message = model_status.message()
        if join and join.incomplete:
            status_message = (
                f"Step snapshot incomplete at step={join.safe_step}. "
                f"Missing ranks: {join.missing_ranks}"
            )

        return LayerCombinedMemoryResult(
            model_index=model_index,
            top_items=top_items,
            other=other,
            all_items=rows_sorted,
            total_current_sum=total_current_sum,
            total_peak_sum=sum(r.total_peak_memory for r in rows_sorted),
            safe_step=join.safe_step if join else None,
            incomplete=join.incomplete if join else False,
            missing_ranks=join.missing_ranks if join else [],
            world_size=join.world_size if join else world_size,
            status_message=status_message,
        )

    def _get_model_snapshot_status(self) -> ModelSnapshotStatus:
        """Validate that all ranks reported the same model snapshot."""
        world_size = self._world_size()
        canonical_rank: Optional[int] = None
        canonical_sig: Optional[str] = None
        canonical_layer_memory: Dict[str, float] = {}
        canonical_model_index: Optional[int] = None

        missing: List[int] = []
        mismatched: List[int] = []

        for rank in range(world_size):
            db = self._remote_store.get_db(
                rank, self.LAYER_MEMORY_NAME + "Sampler"
            )

            last = self._get_last_row(db)
            if not last:
                missing.append(rank)
                continue

            try:
                sample = LayerMemorySample.from_wire(last)
            except Exception:
                missing.append(rank)
                continue

            sig = sample.model_signature
            model_index = sample.model_index
            layer_memory = dict(
                zip(
                    sample.payload.layer_names,
                    sample.payload.layer_param_bytes,
                )
            )

            if canonical_rank is None:
                canonical_rank = rank
                canonical_sig = sig
                canonical_layer_memory = layer_memory
                canonical_model_index = (
                    int(model_index) if model_index is not None else None
                )
                continue

            if sig != canonical_sig or layer_memory != canonical_layer_memory:
                mismatched.append(rank)

        ready = (
            (not missing) and (not mismatched) and (canonical_rank is not None)
        )

        return ModelSnapshotStatus(
            ready=ready,
            canonical_rank=canonical_rank,
            canonical_model_index=canonical_model_index,
            canonical_layer_memory=canonical_layer_memory if ready else {},
            missing_ranks=missing,
            mismatched_ranks=mismatched,
            world_size=world_size,
        )

    def _compute_candidate_safe_step(self, world_size: int) -> int:
        """Return the largest step that should exist for all ranks."""

        def last_step(db: Optional[Database]) -> int:
            if not db:
                return -1
            return max(
                (
                    row.get("step", -1)
                    for rows in db.all_tables().values()
                    for row in rows
                ),
                default=-1,
            )

        steps: List[int] = []
        for rank in range(world_size):
            db_f = self._remote_store.get_db(
                rank, self.LAYER_FORWARD_NAME + "Sampler"
            )
            db_b = self._remote_store.get_db(
                rank, self.LAYER_BACKWARD_NAME + "Sampler"
            )
            if not db_f or not db_b:
                return -1
            steps.append(min(last_step(db_f), last_step(db_b)))

        return min(steps) if steps else -1

    def _compute_step_snapshot(
        self,
        sampler_name: str,
        step: int,
        world_size: int,
    ) -> Tuple[Dict[str, Dict[str, float]], bool, List[int]]:
        """Compute worst-rank current and peak memory up to ``step``."""
        if step < 0:
            return {}, False, list(range(world_size) if world_size > 1 else [])

        layer_current: Dict[str, float] = {}
        layer_peak: Dict[str, float] = {}
        missing: List[int] = []

        for rank in range(world_size):
            rdb = self._remote_store.get_db(rank, sampler_name)
            if rdb is None:
                missing.append(rank)
                continue

            rows = next(iter(rdb.all_tables().values()), [])

            samples = self._load_samples_backwards(
                rows, LayerForwardBackwardMemorySample, step
            )

            cur_sample: Optional[LayerForwardBackwardMemorySample] = next(
                (s for s in reversed(samples) if s.step == step),
                None,
            )

            if cur_sample is not None:
                for layer, mem in zip(
                    cur_sample.payload.layer_names,
                    cur_sample.payload.layer_memory_bytes,
                ):
                    layer_current[layer] = max(
                        layer_current.get(layer, 0.0), float(mem)
                    )
            else:
                missing.append(rank)

            for s in samples:
                if s.step > step:
                    continue
                for layer, mem in zip(
                    s.payload.layer_names,
                    s.payload.layer_memory_bytes,
                ):
                    layer_peak[layer] = max(
                        layer_peak.get(layer, 0.0), float(mem)
                    )

        snapshot = {
            layer: {
                "current_peak": layer_current.get(layer, 0.0),
                "global_peak": layer_peak.get(layer, 0.0),
            }
            for layer in set(layer_current) | set(layer_peak)
        }

        return snapshot, not missing, missing

    def _world_size(self) -> int:
        """Return the active distributed world size, defaulting to one."""
        _, _, world_size = get_ddp_info()
        return max(int(world_size), 1)

    @staticmethod
    def _get_last_row(db: Optional[Database]) -> Optional[dict]:
        if not db:
            return None
        rows = next(iter(db.all_tables().values()), [])
        return rows[-1] if rows else None

    @staticmethod
    def _row_at_step(rows: Deque, step: int) -> Optional[dict]:
        """Find the latest raw row exactly at ``step``."""
        for r in reversed(rows):
            if r.get("step") == step:
                return r
            if r.get("step", -1) < step:
                break
        return None

    @staticmethod
    def _merge_cache(
        cache: Dict[str, Dict[str, float]],
        snapshot: Dict[str, Dict[str, float]],
    ):
        """Merge one snapshot into the live cache."""
        for layer, v in snapshot.items():
            if layer not in cache:
                cache[layer] = {
                    "current": float(v["current_peak"]),
                    "global": float(v["global_peak"]),
                }
            else:
                cache[layer]["current"] = float(v["current_peak"])
                cache[layer]["global"] = max(
                    float(cache[layer]["global"]), float(v["global_peak"])
                )

    def _build_rows(
        self, param_layers: Dict[str, float]
    ) -> List[LayerCombinedMemoryRow]:
        """Build per-layer rows from static params and live caches."""
        rows: List[LayerCombinedMemoryRow] = []
        total_current_sum = 0.0

        for layer, param_mem in param_layers.items():
            fwd = self._forward_cache.get(layer, {})
            bwd = self._backward_cache.get(layer, {})

            current = (
                float(param_mem)
                + float(fwd.get("current", 0.0))
                + float(bwd.get("current", 0.0))
            )
            peak = (
                float(param_mem)
                + float(fwd.get("global", 0.0))
                + float(bwd.get("global", 0.0))
            )

            rows.append(
                LayerCombinedMemoryRow(
                    layer=layer,
                    param_memory=float(param_mem),
                    forward_current=float(fwd.get("current", 0.0)),
                    forward_peak=float(fwd.get("global", 0.0)),
                    backward_current=float(bwd.get("current", 0.0)),
                    backward_peak=float(bwd.get("global", 0.0)),
                    total_current_memory=float(current),
                    total_peak_memory=float(peak),
                    pct=0.0,  # filled below
                )
            )
            total_current_sum += current

        out: List[LayerCombinedMemoryRow] = []
        for r in rows:
            pct = (
                (r.total_current_memory / total_current_sum * 100.0)
                if total_current_sum
                else 0.0
            )
            out.append(
                LayerCombinedMemoryRow(
                    layer=r.layer,
                    param_memory=r.param_memory,
                    forward_current=r.forward_current,
                    forward_peak=r.forward_peak,
                    backward_current=r.backward_current,
                    backward_peak=r.backward_peak,
                    total_current_memory=r.total_current_memory,
                    total_peak_memory=r.total_peak_memory,
                    pct=pct,
                )
            )
        return out

    def _aggregate_other(
        self, rows: List[LayerCombinedMemoryRow], total_current_sum: float
    ) -> LayerCombinedOther:
        """Aggregate non-top layers into one renderer row."""
        cur = sum(r.total_current_memory for r in rows)
        return LayerCombinedOther(
            param_memory=sum(r.param_memory for r in rows),
            forward_current=sum(r.forward_current for r in rows),
            forward_peak=sum(r.forward_peak for r in rows),
            backward_current=sum(r.backward_current for r in rows),
            backward_peak=sum(r.backward_peak for r in rows),
            total_current_memory=cur,
            pct=(
                (cur / total_current_sum * 100.0) if total_current_sum else 0.0
            ),
        )

    def _build_join_status(
        self, step: Optional[int], missing: List[int]
    ) -> DDPJoinStatus:
        """Create a stable join-status object for renderers."""
        world_size = self._world_size()
        return DDPJoinStatus(
            safe_step=step,
            incomplete=bool(missing),
            missing_ranks=missing,
            world_size=world_size,
        )


__all__ = [
    "DDPJoinStatus",
    "LayerCombinedMemoryData",
    "ModelSnapshotStatus",
]
