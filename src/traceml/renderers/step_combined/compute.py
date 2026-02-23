import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from traceml.database.remote_database_store import RemoteDBStore
from traceml.loggers.error_log import get_error_logger
from traceml.renderers.step_combined.schema import (
    StepCombinedRankHeatmap,
    StepCombinedRankRow,
    StepCombinedTimeCoverage,
    StepCombinedTimeMetric,
    StepCombinedTimeResult,
    StepCombinedTimeSeries,
    StepCombinedTimeSummary,
)
from traceml.samplers.schema.time_schema import TimeEventSample


@dataclass(frozen=True)
class MetricSpec:
    """
    Table-driven metric specification.
    """
    metric_key: str
    table: str
    sampler: str = "TimeSampler"


# ----------------------------
# Configuration (edit here)
# ----------------------------

DEFAULT_METRIC_SPECS: List[MetricSpec] = [
    MetricSpec(metric_key="dataloader_fetch", table="_traceml_internal:dataloader_next"),
    MetricSpec(metric_key="forward", table="_traceml_internal:forward_time"),
    MetricSpec(metric_key="backward", table="_traceml_internal:backward_time"),
    MetricSpec(metric_key="optimizer_step", table="_traceml_internal:optimizer_step"),
    MetricSpec(metric_key="step_time", table="_traceml_internal:step_time"),
]

WAIT_METRIC_KEY = "wait_proxy"
WAIT_STEP_KEY = "step_time"
WAIT_GPU_KEYS: Tuple[str, ...] = ("forward", "backward", "optimizer_step")

DEFAULT_HEATMAP_KEYS: Tuple[str, ...] = (
    "dataloader_fetch",
    "forward",
    "backward",
    "optimizer_step",
    "wait_proxy",
    "step_time",
)


class StepCombinedComputer:
    """
    Rank-agnostic computation of step combined time metrics.

    Adds stability cache:
    - returns last good payload on transient failure/empty compute
    - optional TTL to avoid showing stale forever
    """

    def __init__(
        self,
        remote_store: RemoteDBStore,
        *,
        window_size: int = 100,
        metric_specs: Optional[Sequence[MetricSpec]] = None,
        heatmap_keys: Optional[Sequence[str]] = None,
        stale_ttl_s: Optional[float] = 30.0,
    ):
        self.store = remote_store
        self.window_size = int(window_size)
        self.logger = get_error_logger("StepCombinedComputer")

        self._metric_specs: List[MetricSpec] = (
            list(metric_specs) if metric_specs is not None else list(DEFAULT_METRIC_SPECS)
        )
        self._heatmap_keys: List[str] = (
            list(heatmap_keys) if heatmap_keys is not None else list(DEFAULT_HEATMAP_KEYS)
        )

        # constants for fast membership
        self._wait_gpu_key_set = set(WAIT_GPU_KEYS)

        # last-good-result cache to avoid blank UI
        self._last_ok: Optional[StepCombinedTimeResult] = None
        self._last_ok_ts: float = 0.0
        self._stale_ttl_s: Optional[float] = float(stale_ttl_s) if stale_ttl_s is not None else None

    def compute_cli(self) -> StepCombinedTimeResult:
        return self._compute(include_series=False, include_rank_heatmap=False)

    def compute_dashboard(self) -> StepCombinedTimeResult:
        return self._compute(include_series=False, include_rank_heatmap=True)

    # ----------------------------
    # Public wrapper with cache
    # ----------------------------

    def _compute(
        self,
        *,
        include_series: bool,
        include_rank_heatmap: bool,
    ) -> StepCombinedTimeResult:
        try:
            res = self._compute_impl(
                include_series=include_series,
                include_rank_heatmap=include_rank_heatmap,
            )
        except Exception as e:
            self.logger.exception("StepCombined compute failed")
            return self._return_stale_or_empty(f"STALE (exception: {type(e).__name__})")

        # If transiently empty (no metrics), keep old data so UI doesn't blank
        if not res.metrics:
            return self._return_stale_or_empty("STALE (no metrics this tick)")

        # Success: refresh cache
        self._last_ok = res
        self._last_ok_ts = time.time()
        return res

    def _return_stale_or_empty(self, msg: str) -> StepCombinedTimeResult:
        now = time.time()

        if self._last_ok is not None:
            if self._stale_ttl_s is None or (now - self._last_ok_ts) <= self._stale_ttl_s:
                return StepCombinedTimeResult(
                    metrics=self._last_ok.metrics,
                    status_message=msg,
                    rank_heatmap=self._last_ok.rank_heatmap,
                )

        # cache missing or too old -> empty
        return StepCombinedTimeResult(
            metrics=[],
            status_message="No fresh step-combined data",
            rank_heatmap=None,
        )

    # ----------------------------
    # Actual compute implementation
    # ----------------------------

    def _compute_impl(
        self,
        *,
        include_series: bool,
        include_rank_heatmap: bool,
    ) -> StepCombinedTimeResult:
        ranks = list(self.store.ranks())
        world_size = len(ranks)

        if world_size == 0:
            return StepCombinedTimeResult(
                metrics=[],
                status_message="No ranks available",
                rank_heatmap=None,
            )

        computed: Dict[str, StepCombinedTimeMetric] = {}
        per_metric_rank_sums: Dict[str, Dict[int, float]] = {}

        # 1) Base metrics
        for spec in self._metric_specs:
            metric, rank_sums = self._compute_one_metric(
                table=spec.table,
                sampler=spec.sampler,
                metric_key=spec.metric_key,
                ranks=ranks,
                world_size=world_size,
                include_series=include_series,
            )
            if metric is None:
                continue

            computed[spec.metric_key] = metric
            if rank_sums is not None:
                per_metric_rank_sums[spec.metric_key] = rank_sums

        # 2) Derived WAIT proxy
        wait_metric, wait_rank_sums = self._derive_wait_proxy(
            step_metric=computed.get(WAIT_STEP_KEY),
            gpu_metrics={k: v for k, v in computed.items() if k in self._wait_gpu_key_set},
            per_metric_rank_sums=per_metric_rank_sums,
        )
        if wait_metric is not None:
            computed[WAIT_METRIC_KEY] = wait_metric
            if wait_rank_sums is not None:
                per_metric_rank_sums[WAIT_METRIC_KEY] = wait_rank_sums

        status = "OK" if computed else "No complete metrics available"

        rank_heatmap = None
        if include_rank_heatmap and computed:
            cov_metric = computed.get(WAIT_STEP_KEY) or next(iter(computed.values()))
            rank_heatmap = self._build_rank_heatmap(
                coverage=cov_metric.coverage,
                per_metric_rank_sums=per_metric_rank_sums,
            )

        return StepCombinedTimeResult(
            metrics=list(computed.values()),
            status_message=status,
            rank_heatmap=rank_heatmap,
        )

    # ----------------------------
    # Optimized per-metric compute
    # ----------------------------

    def _compute_one_metric(
        self,
        *,
        table: str,
        sampler: str,
        metric_key: str,
        ranks: List[int],
        world_size: int,
        include_series: bool,
    ) -> Tuple[Optional[StepCombinedTimeMetric], Optional[Dict[int, float]]]:
        per_rank_steps: Dict[int, Dict[int, float]] = {}
        per_rank_clocks: Dict[int, str] = {}
        completed_steps: List[int] = []

        store = self.store
        ws = self.window_size
        from_wire = TimeEventSample.from_wire  # local bind

        for rank in ranks:
            db = store.get_db(rank, sampler)
            if not db:
                continue

            tbl = db.get_table(table)
            if not tbl:
                continue

            step_map: Dict[int, float] = {}
            clock: str = "cpu"

            # Deque-backed table: fast reverse scan using indexing from right end.
            # If tbl isn't indexable (should be deque), fallback to reversed().
            try:
                n = len(tbl)
                for i in range(n - 1, -1, -1):
                    row = tbl[i]
                    try:
                        s = from_wire(row)
                    except Exception:
                        continue

                    step_map[s.step] = float(s.duration_ms)
                    clock = "gpu" if s.is_gpu else "cpu"

                    if len(step_map) >= ws:
                        break
            except Exception:
                for row in reversed(tbl):
                    try:
                        s = from_wire(row)
                    except Exception:
                        continue
                    step_map[s.step] = float(s.duration_ms)
                    clock = "gpu" if s.is_gpu else "cpu"
                    if len(step_map) >= ws:
                        break

            if step_map:
                per_rank_steps[rank] = step_map
                per_rank_clocks[rank] = clock
                completed_steps.append(max(step_map.keys()))

        if not per_rank_steps:
            return None, None

        completed_step = min(completed_steps)
        ranks_present = len(per_rank_steps)

        # Same semantics as intersection-based:
        # "largest suffix of steps common across ranks (up to completed_step)"
        steps = self._common_window_steps_fast(per_rank_steps, completed_step, ws)
        if not steps:
            return None, None

        sums_arr, ranks_list, per_rank_sum = self._rank_window_sums_fast(per_rank_steps, steps)

        median_total = float(np.median(sums_arr))
        worst_idx = int(np.argmax(sums_arr))
        worst_total = float(sums_arr[worst_idx])
        worst_rank = int(ranks_list[worst_idx])

        if ranks_present <= 1:
            skew_ratio = 0.0
            skew_pct = 0.0
            median_total = worst_total
        else:
            if median_total > 0.0:
                skew_ratio = worst_total / median_total
                skew_pct = (worst_total - median_total) / median_total
            else:
                skew_ratio = 0.0
                skew_pct = 0.0

        clocks = list(per_rank_clocks.values())
        clock = max(set(clocks), key=clocks.count) if clocks else "cpu"

        coverage = StepCombinedTimeCoverage(
            expected_steps=ws,
            steps_used=len(steps),
            completed_step=completed_step,
            world_size=world_size,
            ranks_present=ranks_present,
            incomplete=(ranks_present < world_size),
        )

        series: Optional[StepCombinedTimeSeries] = None
        if include_series:
            # Same semantics as original, but uses numpy reductions.
            ranks_keys = list(per_rank_steps.keys())
            median_y: List[float] = []
            worst_y: List[float] = []
            sum_y: List[float] = []

            for s in steps:
                v = np.fromiter((per_rank_steps[r][s] for r in ranks_keys), dtype=np.float64)
                median_y.append(float(np.median(v)))
                worst_y.append(float(np.max(v)))
                sum_y.append(float(np.sum(v)))

            series = StepCombinedTimeSeries(
                steps=steps,
                median=median_y,
                worst=worst_y,
                sum=sum_y,
            )

        metric = StepCombinedTimeMetric(
            metric=metric_key,
            clock=clock,
            series=series,
            summary=StepCombinedTimeSummary(
                window_size=ws,
                steps_used=len(steps),
                median_total=median_total,
                worst_total=worst_total,
                worst_rank=worst_rank,
                skew_ratio=float(skew_ratio),
                skew_pct=float(skew_pct),
            ),
            coverage=coverage,
        )
        return metric, per_rank_sum

    @staticmethod
    def _common_window_steps_fast(
        per_rank_steps: Dict[int, Dict[int, float]],
        completed_step: int,
        window_size: int,
    ) -> List[int]:
        """
        Faster version of:
          intersect step indices across ranks (<= completed_step),
          return last window_size common steps, ascending.

        Walk backward from completed_step and test membership in all rank maps.
        Equivalent output to intersection+sorted+slice for the "suffix of common steps".
        """
        if not per_rank_steps:
            return []

        maps = list(per_rank_steps.values())
        ref = maps[0]

        out_rev: List[int] = []
        s = completed_step
        while s >= 0 and len(out_rev) < window_size:
            if s in ref:
                ok = True
                for m in maps[1:]:
                    if s not in m:
                        ok = False
                        break
                if ok:
                    out_rev.append(s)
            s -= 1

        if not out_rev:
            return []
        out_rev.reverse()
        return out_rev

    @staticmethod
    def _rank_window_sums_fast(
        per_rank_steps: Dict[int, Dict[int, float]],
        steps: Sequence[int],
    ) -> Tuple[np.ndarray, List[int], Dict[int, float]]:
        ranks_list = list(per_rank_steps.keys())
        n = len(ranks_list)
        sums_arr = np.empty(n, dtype=np.float64)
        per_rank_sum: Dict[int, float] = {}

        steps_local = steps
        for i, r in enumerate(ranks_list):
            smap = per_rank_steps[r]
            get = smap.get
            total = 0.0
            for s in steps_local:
                total += float(get(s, 0.0))
            sums_arr[i] = total
            per_rank_sum[r] = float(total)

        return sums_arr, ranks_list, per_rank_sum

    # ----------------------------
    # WAIT proxy (same semantics)
    # ----------------------------

    def _derive_wait_proxy(
        self,
        *,
        step_metric: Optional[StepCombinedTimeMetric],
        gpu_metrics: Dict[str, StepCombinedTimeMetric],
        per_metric_rank_sums: Dict[str, Dict[int, float]],
    ) -> Tuple[Optional[StepCombinedTimeMetric], Optional[Dict[int, float]]]:
        if step_metric is None or not gpu_metrics:
            return None, None

        gpu_median = float(sum(m.summary.median_total for m in gpu_metrics.values()))
        gpu_worst = float(sum(m.summary.worst_total for m in gpu_metrics.values()))

        wait_median_total = max(0.0, step_metric.summary.median_total - gpu_median)
        wait_worst_total = max(0.0, step_metric.summary.worst_total - gpu_worst)

        ranks_present = step_metric.coverage.ranks_present
        if ranks_present <= 1:
            skew_ratio = 0.0
            skew_pct = 0.0
        else:
            if wait_median_total > 0.0:
                skew_ratio = wait_worst_total / wait_median_total
                skew_pct = (wait_worst_total - wait_median_total) / wait_median_total
            else:
                skew_ratio = 0.0
                skew_pct = 0.0

        wait_metric = StepCombinedTimeMetric(
            metric=WAIT_METRIC_KEY,
            clock="mixed",
            series=None,
            summary=StepCombinedTimeSummary(
                window_size=step_metric.summary.window_size,
                steps_used=step_metric.summary.steps_used,
                median_total=wait_median_total,
                worst_total=wait_worst_total,
                worst_rank=step_metric.summary.worst_rank,
                skew_ratio=float(skew_ratio),
                skew_pct=float(skew_pct),
            ),
            coverage=step_metric.coverage,
        )

        step_sums = per_metric_rank_sums.get(WAIT_STEP_KEY)
        fwd_sums = per_metric_rank_sums.get("forward")
        bwd_sums = per_metric_rank_sums.get("backward")
        opt_sums = per_metric_rank_sums.get("optimizer_step")

        if not (step_sums and fwd_sums and bwd_sums and opt_sums):
            return wait_metric, None

        wait_rank_sums: Dict[int, float] = {}
        for r, step_sum in step_sums.items():
            gpu_sum = float(fwd_sums.get(r, 0.0) + bwd_sums.get(r, 0.0) + opt_sums.get(r, 0.0))
            wait_rank_sums[r] = float(max(0.0, step_sum - gpu_sum))

        return wait_metric, wait_rank_sums

    # ----------------------------
    # Heatmap build (same schema)
    # ----------------------------

    def _build_rank_heatmap(
        self,
        *,
        coverage: StepCombinedTimeCoverage,
        per_metric_rank_sums: Dict[str, Dict[int, float]],
    ) -> Optional[StepCombinedRankHeatmap]:
        if coverage.ranks_present == 0:
            return None

        keys = [k for k in self._heatmap_keys if (k in per_metric_rank_sums or k == WAIT_METRIC_KEY)]

        ref = per_metric_rank_sums.get(WAIT_STEP_KEY) or next(iter(per_metric_rank_sums.values()), {})
        ranks = list(ref.keys())

        pmrs = per_metric_rank_sums
        rows: List[StepCombinedRankRow] = []
        for r in ranks:
            sums_ms = {k: float(pmrs.get(k, {}).get(r, 0.0)) for k in keys}
            rows.append(StepCombinedRankRow(rank=r, sums_ms=sums_ms))

        def _sort_key(row: StepCombinedRankRow) -> Tuple[float, float]:
            sm = row.sums_ms
            return (sm.get(WAIT_STEP_KEY, 0.0), sm.get("dataloader_fetch", 0.0))

        rows.sort(key=_sort_key, reverse=True)

        return StepCombinedRankHeatmap(
            window_size=self.window_size,
            steps_used=coverage.steps_used,
            metric_keys=keys,
            rows=rows,
            sort_by=[WAIT_STEP_KEY, "dataloader_fetch"],
        )