"""
Layer-wise combined memory computation (aggregator-side).
This module computes a **capacity-oriented** view of memory usage by layer:
    total_memory =
        parameter_memory
      + forward_activation_memory
      + backward_activation_memory

Key semantics
------------------------
- Aggregator-only: reads ALL rank data from `RemoteDBStore`
- Validates that all ranks reported the same model snapshot (by signature)
- Selects a *safe step* that is step-aligned across ranks
- Computes:
    * current memory: worst rank at safe_step
    * peak memory: worst rank across time up to safe_step
- Produces a renderer-facing **typed result object** (no dict spelunking)

"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Deque, Iterable, Type, Set

from traceml.database.database import Database
from traceml.database.remote_database_store import RemoteDBStore
from traceml.transport.distributed import get_ddp_info
from traceml.loggers.error_log import get_error_logger

from traceml.samplers.schema.layer_memory import LayerMemorySample
from traceml.samplers.schema.layer_forward_backward_memory import (
    LayerForwardBackwardMemorySample,
)

from traceml.renderers.layer_combined_memory.schema import (
    LayerCombinedMemoryRow,
    LayerCombinedOther,
    LayerCombinedMemoryResult,
)



@dataclass(frozen=True)
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

    - A snapshot can be incomplete if at least one rank did not report for the chosen step.
    """

    safe_step: Optional[int]
    incomplete: bool
    missing_ranks: List[int]
    world_size: int


@dataclass(frozen=True)
class ModelSnapshotStatus:
    """
    Aggregator-side view of model snapshot availability across ranks.

    Semantics
    ---------
    - We expect a single "model snapshot" per rank (layer table) for the current run.
    - The snapshot is considered "ready" only if ALL ranks have reported it AND
      the model signatures match across ranks.
    """

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
            return f"Waiting for model snapshot from ranks: {self.missing_ranks}"
        if self.mismatched_ranks:
            return (
                "Model snapshot mismatch across ranks. "
                f"Mismatched ranks: {self.mismatched_ranks}"
            )
        return "Model snapshot not available yet."


class LayerCombinedMemoryData:
    """
    Computes **layer-wise combined memory** (DDP capacity view):
        param_memory + forward_activations + backward_activations

    Responsibilities:
    - Aggregator-only: reads ALL rank data from `RemoteDBStore`
    - Validates that all ranks reported the same model snapshot (by signature)
    - Selects a *safe step* that is step-aligned across ranks
    - Computes:
        * current memory: worst rank at safe_step
        * peak memory: worst rank across time up to safe_step
    - Produces a renderer-ready data structure

    - "current"  is **layer-wise worst across ranks**
    - "peak"     is **max of those worst values over time**
    - Rows DO NOT correspond to a single rank
    - This is a *capacity / safety* view
    - Returns a typed result object: `LayerCombinedMemoryResult`.
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

        # Cache structure (unchanged):
        #   layer -> {"current": float, "global": float}
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
        """
        Load schema objects from wire rows, scanning backwards.
        Stops early once rows fall below `min_step`.

        Parameters
        ----------
        rows : Iterable[dict]
            Wire-format rows (append-only, step-ordered).
        sample_cls : type
            Schema class with `from_wire`.
        min_step : int
            Lower bound on step index to load.

        Returns
        -------
        List[Any]
            Samples with step >= min_step, newest-first.
        """
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
        """
        Compute all layer-wise memory metrics required by renderers.

        Returns
        -------
        LayerCombinedMemoryResult
            Renderer-facing typed payload:
              - per-layer current & peak memory
              - top-N layers + "other"
              - DDP join metadata
              - status message
        """
        model_status = self._get_model_snapshot_status()
        world_size = model_status.world_size

        # Even when model is not ready, we keep the output contract stable.
        # Renderers can show "waiting" states based on `status_message`.
        if not model_status.ready:
            self._join_status = self._build_join_status(None, model_status.missing_ranks)
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
                    set(model_status.missing_ranks) | set(model_status.mismatched_ranks)
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
            self._join_status = self._build_join_status(safe_step_candidate, missing)
        elif self._last_safe_step >= 0:
            # Fall back to last known safe step for a stable UI during transient gaps.
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
            self._join_status = self._build_join_status(self._last_safe_step, missing)
        else:
            self._join_status = self._build_join_status(None, [])

        self._merge_cache(self._forward_cache, fwd_snapshot)
        self._merge_cache(self._backward_cache, bwd_snapshot)

        rows = self._build_rows(param_layers)
        rows_sorted = sorted(rows, key=lambda r: r.total_peak_memory, reverse=True)

        top_items = rows_sorted[: self._top_n]
        other_items = rows_sorted[self._top_n :]

        total_current_sum = sum(r.total_current_memory for r in rows_sorted)
        other = self._aggregate_other(other_items, total_current_sum)

        join = self._join_status

        # Status message: show missing ranks if any data is incomplete; otherwise
        # provide the "ready" message so UI can display a green state if desired.
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
        """
        Validate model snapshot availability and consistency across ranks.

        Strategy
        --------------------
        - Each rank is expected to publish its model snapshot to `LayerMemorySampler`.
        - We use the latest row per rank.
        - A "ready" snapshot requires:
            * all ranks present
            * all signatures identical
        - Canonical source is the first rank that provides a snapshot.

        Returns
        -------
        ModelSnapshotStatus
        """
        world_size = self._world_size()
        canonical_rank: Optional[int] = None
        canonical_sig: Optional[str] = None
        canonical_layer_memory: Dict[str, float] = {}
        canonical_model_index: Optional[int] = None

        missing: List[int] = []
        mismatched: List[int] = []

        for rank in range(world_size):
            db = self._remote_store.get_db(rank, self.LAYER_MEMORY_NAME + "Sampler")

            last = self._get_last_row(db)
            if not last:
                missing.append(rank)
                continue

            # Contract: load using schema
            try:
                sample = LayerMemorySample.from_wire(last)
            except Exception:
                missing.append(rank)
                continue

            sig = sample.model_signature
            model_index = sample.model_index
            layer_memory = dict(
                zip(sample.payload.layer_names, sample.payload.layer_param_bytes)
            )

            # First valid snapshot becomes canonical.
            if canonical_rank is None:
                canonical_rank = rank
                canonical_sig = sig
                canonical_layer_memory = layer_memory
                canonical_model_index = int(model_index) if model_index is not None else None
                continue

            # Any inconsistency should be surfaced (DDP expects identical models).
            if sig != canonical_sig or layer_memory != canonical_layer_memory:
                mismatched.append(rank)

        ready = (not missing) and (not mismatched) and (canonical_rank is not None)

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
        """
        Compute the largest step index that *should* be present on all ranks.

        Notes
        -----
        - This is a candidate only; availability is verified per-rank in `_compute_step_snapshot`.
        - We consider both forward and backward samplers and take the minimum across ranks.
        """

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
            db_f = self._remote_store.get_db(rank, self.LAYER_FORWARD_NAME + "Sampler")
            db_b = self._remote_store.get_db(rank, self.LAYER_BACKWARD_NAME + "Sampler")
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
        """
        Compute per-layer current & peak memory up to `step`.

        Semantics (unchanged)
        ---------------------
        - "current_peak" : worst-rank value at exactly `step`
        - "global_peak"  : worst-rank max over all steps <= `step`

        Returns
        -------
        snapshot : Dict[layer, {"current_peak", "global_peak"}]
        ok : bool
            True if all expected ranks had data for `step`
        missing_ranks : List[int]
        """
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

            # Contract: parse rows into schema objects once
            samples = self._load_samples_backwards(rows, LayerForwardBackwardMemorySample, step)

            # Current at step
            cur_sample: Optional[LayerForwardBackwardMemorySample] = next(
                (s for s in reversed(samples) if s.step == step),
                None,
            )

            if cur_sample is not None:
                for layer, mem in zip(
                    cur_sample.payload.layer_names,
                    cur_sample.payload.layer_memory_bytes,
                ):
                    layer_current[layer] = max(layer_current.get(layer, 0.0), float(mem))
            else:
                missing.append(rank)

            # Peak up to step
            for s in samples:
                if s.step > step:
                    continue
                for layer, mem in zip(
                    s.payload.layer_names,
                    s.payload.layer_memory_bytes,
                ):
                    layer_peak[layer] = max(layer_peak.get(layer, 0.0), float(mem))

        snapshot = {
            layer: {
                "current_peak": layer_current.get(layer, 0.0),
                "global_peak": layer_peak.get(layer, 0.0),
            }
            for layer in set(layer_current) | set(layer_peak)
        }

        return snapshot, not missing, missing

    def _world_size(self) -> int:
        """
        Obtain world size from the distributed runtime.

        We keep this as a function to make it easy to evolve later.
        """
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
        """
        Find the latest row exactly at `step`.

        NOTE: kept for compatibility with existing call sites/tests.
        The schema-based path no longer requires this function.
        """
        for r in reversed(rows):
            if r.get("step") == step:
                return r
            if r.get("step", -1) < step:
                break
        return None


    @staticmethod
    def _merge_cache(
        cache: Dict[str, Dict[str, float]], snapshot: Dict[str, Dict[str, float]]
    ):
        """
        Merge an incremental snapshot into the running cache.

        Cache semantics (unchanged):
        - "current": the latest value for the chosen safe step
        - "global": max observed up to that safe step (monotone)
        """
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


    def _build_rows(self, param_layers: Dict[str, float]) -> List[LayerCombinedMemoryRow]:
        """
        Build per-layer rows with current & peak totals.

        Notes (unchanged)
        -----------------
        - Param memory is considered static.
        - Forward/backward values come from caches and reflect the latest safe-step join.
        """
        rows: List[LayerCombinedMemoryRow] = []
        total_current_sum = 0.0

        for layer, param_mem in param_layers.items():
            fwd = self._forward_cache.get(layer, {})
            bwd = self._backward_cache.get(layer, {})

            current = float(param_mem) + float(fwd.get("current", 0.0)) + float(
                bwd.get("current", 0.0)
            )
            peak = float(param_mem) + float(fwd.get("global", 0.0)) + float(
                bwd.get("global", 0.0)
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

        # Fill pct
        out: List[LayerCombinedMemoryRow] = []
        for r in rows:
            pct = (r.total_current_memory / total_current_sum * 100.0) if total_current_sum else 0.0
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
        """
        Aggregate non-top layers into a single "other" bucket.
        This keeps the UI readable while preserving accurate totals.
        """
        cur = sum(r.total_current_memory for r in rows)
        return LayerCombinedOther(
            param_memory=sum(r.param_memory for r in rows),
            forward_current=sum(r.forward_current for r in rows),
            forward_peak=sum(r.forward_peak for r in rows),
            backward_current=sum(r.backward_current for r in rows),
            backward_peak=sum(r.backward_peak for r in rows),
            total_current_memory=cur,
            pct=(cur / total_current_sum * 100.0) if total_current_sum else 0.0,
        )

    def _build_join_status(self, step: Optional[int], missing: List[int]) -> DDPJoinStatus:
        """
        Create a stable join-status object for renderers.
        """
        world_size = self._world_size()
        return DDPJoinStatus(
            safe_step=step,
            incomplete=bool(missing),
            missing_ranks=missing,
            world_size=world_size,
        )


# TODO: WE SHOULD READ ALL RANKS AND ENTIRE DB AND COMPUTE SUMMARY (v1)
class LayerCombinedMemorySummary:
    """
    Computes coarse, global statistics for logging and reports (aggregator-side).

    Design (unchanged)
    ------------------
    - Reads from RemoteDBStore only.
    - Intended for *historical* questions, not real-time joins.
    - Avoids step-alignment semantics; it scans available DB content.

    Notes
    -----
    This summary is best-effort:
    - If some ranks have not reported yet, stats will be partial.
    """

    def __init__(
        self,
        remote_store: RemoteDBStore,
        *,
        layer_memory_name: str = "LayerMemory",
        layer_forward_name: str = "LayerForwardMemory",
        layer_backward_name: str = "LayerBackwardMemory",
    ):
        self._remote_store = remote_store
        self.layer_memory_name = layer_memory_name
        self.layer_forward_name = layer_forward_name
        self.layer_backward_name = layer_backward_name
        self.logger = get_error_logger("LayerCombinedMemorySummary")

    def compute_layer_memory_summary(self) -> Dict[str, Any]:
        """
        Summarize static model memory usage based on layer tables.

        Returns
        -------
        Dict[str, Any]
            - total_models_seen: number of distinct signatures seen across ranks
            - model_memory: average total parameter memory across available snapshots

        NOTE
        ----
        This function keeps the return type as a dict because it is used for
        logging/reporting and is not part of the renderer contract.
        """
        signatures: Set[str] = set()
        totals: List[float] = []

        for rank in self._remote_store.ranks():
            db = self._safe_get_db(rank, self.layer_memory_name + "Sampler")
            if not db:
                continue

            rows = db.get_table(self.layer_memory_name + "Table")
            if not rows:
                continue

            try:
                last = LayerMemorySample.from_wire(rows[-1])
            except Exception:
                continue

            if last.model_signature:
                signatures.add(last.model_signature)
            totals.append(float(last.total_param_bytes))

        return {
            "total_models_seen": len(signatures),
            "model_memory": (sum(totals) / len(totals)) if totals else 0.0,
        }

    def compute_global_peaks(self, is_forward: bool) -> Dict[str, float]:
        """
        Compute global (time-unbounded) peak memory per layer across all ranks.

        Parameters
        ----------
        is_forward : bool
            If True, scan forward sampler; otherwise scan backward sampler.

        Returns
        -------
        Dict[str, float]
            Mapping: layer name -> peak memory (bytes)

        NOTE
        ----
        This keeps a dict return because it's a utility output for logs/reports.
        """
        sampler = self.layer_forward_name if is_forward else self.layer_backward_name
        sampler_db_name = sampler + "Sampler"

        peaks: Dict[str, float] = {}

        for rank in self._remote_store.ranks():
            db = self._safe_get_db(rank, sampler_db_name)
            if not db:
                continue

            rows = db.get_table(sampler + "Table")
            if not rows:
                continue

            samples = LayerCombinedMemoryData._load_samples_backwards(
                rows, LayerForwardBackwardMemorySample, 0
            )
            for s in samples:
                for layer, mem in zip(
                    s.payload.layer_names,
                    s.payload.layer_memory_bytes,
                ):
                    peaks[layer] = max(peaks.get(layer, 0.0), float(mem))

        return peaks

    @staticmethod
    def top_n_from_dict(d: Dict[str, float], n: int = 3):
        """
        Return top-N items from a dict sorted by descending value.
        """
        return sorted(d.items(), key=lambda x: x[1], reverse=True)[:n] if d else []

    def _safe_get_db(self, rank: int, sampler_name: str) -> Optional[Database]:
        """
        Safely fetch a per-rank database from the remote store.

        Returns None if the database has not arrived yet or if an
        unexpected error occurs.
        """
        try:
            return self._remote_store.get_db(rank, sampler_name)
        except Exception:
            self.logger.exception(
                "Failed to fetch rank DB from remote store",
                extra={"rank": rank, "sampler_name": sampler_name},
            )
            return None
