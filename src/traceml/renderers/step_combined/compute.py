"""
Step Combined Time Computation (DDP-aware, rank-agnostic)

This module is the *pure compute layer* for TraceML's "step combined" metrics.

It supports two compute modes:

- compute_cli():
    Summary-only output (no per-step series, no rank heatmap).
    Intended for cheap terminal rendering.

- compute_dashboard():
    Summary output + a per-rank heatmap payload, where each cell is the
    *sum over the last K fully-common steps* (K = window_size).
    Intended for dashboards (NiceGUI/Plotly) and straggler analysis.

What this computer does
-----------------------
For each metric (dataloader, forward, backward, optimizer, step_time):
1) Read per-rank TimeEventSample rows from RemoteDBStore tables
2) Build per-rank {step -> duration_ms} maps (up to window_size recent samples)
3) Find the largest suffix of steps common across ranks (up to completed_step)
4) Compute:
   - per-rank window sums (ΣK) over those common steps
   - cross-rank summary stats on those sums (median, worst, skew, worst-rank)
   - optional per-step series (median/worst/sum) if requested

Derived metric: wait_proxy (mixed-clock proxy)
----------------------------------------------
wait_proxy is a residual meant for *directional* analysis only:

    wait_proxy = step_time - (forward + backward + optimizer_step)

This mixes CPU wall time (step_time) with GPU stream times (forward/backward/opt)
and therefore must be presented as a proxy.

Notes on correctness & stability
--------------------------------
- Step alignment is done via intersection of step indices across ranks.
- "Completed step" is defined as min(max_step_per_rank) across ranks.
- If ranks are missing or out of sync, coverage.incomplete=True and steps_used
  may be < window_size. Renderers should handle this gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

    Attributes
    ----------
    metric_key:
        Canonical metric key exposed to renderers/dashboards.
        Keep stable for UI compatibility.
    table:
        RemoteDBStore table name containing TimeEventSample rows.
    sampler:
        Sampler namespace in RemoteDBStore (default: "TimeSampler").
    """

    metric_key: str
    table: str
    sampler: str = "TimeSampler"


# ----------------------------
# Configuration (edit here)
# ----------------------------

DEFAULT_METRIC_SPECS: List[MetricSpec] = [
    MetricSpec(
        metric_key="dataloader_fetch",
        table="_traceml_internal:dataloader_next",
    ),
    MetricSpec(
        metric_key="forward",
        table="_traceml_internal:forward_time",
    ),
    MetricSpec(
        metric_key="backward",
        table="_traceml_internal:backward_time",
    ),
    MetricSpec(
        metric_key="optimizer_step",
        table="_traceml_internal:optimizer_step",
    ),
    # Canonical name: no random "_ms" suffix; values are still ms by definition.
    MetricSpec(
        metric_key="step_time",
        table="_traceml_internal:step_time",
    ),
]

WAIT_METRIC_KEY = "wait_proxy"
WAIT_STEP_KEY = "step_time"
WAIT_GPU_KEYS: Tuple[str, ...] = ("forward", "backward", "optimizer_step")

# Default heatmap columns for dashboard (ordered)
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

    Parameters
    ----------
    remote_store:
        RemoteDBStore that provides access to rank-local sampler tables.
    window_size:
        Number of recent common steps to include in the windowed aggregation.

    Customization
    -------------
    Metrics are table-driven via MetricSpec. For OSS friendliness, this class
    accepts an optional `metric_specs` so integrators can add/remove metrics
    without editing core logic.
    """

    def __init__(
        self,
        remote_store: RemoteDBStore,
        *,
        window_size: int = 100,
        metric_specs: Optional[Sequence[MetricSpec]] = None,
        heatmap_keys: Optional[Sequence[str]] = None,
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



    def compute_cli(self) -> StepCombinedTimeResult:
        """
        Compute the minimal payload for CLI rendering.

        Returns only:
        - StepCombinedTimeMetric summaries (no per-step series, no heatmap)
        """
        return self._compute(include_series=False, include_rank_heatmap=False)

    def compute_dashboard(self) -> StepCombinedTimeResult:
        """
        Compute a dashboard-oriented payload.

        Returns:
        - StepCombinedTimeMetric summaries (no per-step series by default)
        - rank_heatmap: per-rank window sums for straggler heatmap UI
        """
        return self._compute(include_series=False, include_rank_heatmap=True)


    def _compute(
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
            gpu_metrics={k: v for k, v in computed.items() if k in set(WAIT_GPU_KEYS)},
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
        """
        Compute one metric end-to-end.

        Returns
        -------
        metric:
            StepCombinedTimeMetric or None if insufficient data.
        per_rank_window_sum:
            Dict[rank -> ΣK(ms)] over the last K fully-common steps, or None.
        """
        per_rank_steps: Dict[int, Dict[int, float]] = {}
        per_rank_clocks: Dict[int, str] = {}
        completed_steps: List[int] = []

        # Collect per-rank step maps
        for rank in ranks:
            db = self.store.get_db(rank, sampler)
            if not db:
                continue

            tbl = db.get_table(table)
            if not tbl:
                continue

            step_map: Dict[int, float] = {}
            clock: Optional[str] = None

            # Reverse iteration: consume latest samples first
            for row in reversed(tbl):
                try:
                    s = TimeEventSample.from_wire(row)
                except Exception:
                    continue

                step_map[s.step] = float(s.duration_ms)
                clock = "gpu" if s.is_gpu else "cpu"

                if len(step_map) >= self.window_size:
                    break

            if step_map:
                per_rank_steps[rank] = step_map
                per_rank_clocks[rank] = clock or "cpu"
                completed_steps.append(max(step_map.keys()))

        if not per_rank_steps:
            return None, None

        completed_step = min(completed_steps)
        ranks_present = len(per_rank_steps)

        steps = self._common_window_steps(per_rank_steps.values(), completed_step, self.window_size)
        if not steps:
            return None, None

        # Per-rank window sums ΣK
        per_rank_sum: Dict[int, float] = {
            r: float(sum(per_rank_steps[r][s] for s in steps))
            for r in per_rank_steps
        }

        # Cross-rank summary stats on ΣK
        sums = list(per_rank_sum.values())
        median_total = float(np.median(sums))
        worst_total = float(max(sums))
        worst_rank = max(per_rank_sum, key=lambda r: per_rank_sum[r])

        if ranks_present <= 1:
            # Single rank: skew is not meaningful
            skew_ratio = 0.0
            skew_pct = 0.0
            median_total = worst_total
        else:
            skew_ratio = worst_total / median_total if median_total > 0 else 0.0
            skew_pct = (
                (worst_total - median_total) / median_total if median_total > 0 else 0.0
            )

        # Clock: majority vote across present ranks
        clocks = list(per_rank_clocks.values())
        clock = max(set(clocks), key=clocks.count) if clocks else "cpu"

        coverage = StepCombinedTimeCoverage(
            expected_steps=self.window_size,
            steps_used=len(steps),
            completed_step=completed_step,
            world_size=world_size,
            ranks_present=ranks_present,
            incomplete=(ranks_present < world_size),
        )

        series: Optional[StepCombinedTimeSeries] = None
        if include_series:
            # This is intentionally optional: more allocations & work
            median_y: List[float] = []
            worst_y: List[float] = []
            sum_y: List[float] = []

            for s in steps:
                vals = [per_rank_steps[r][s] for r in per_rank_steps]
                vals.sort()
                median_y.append(float(np.median(vals)))
                worst_y.append(float(vals[-1]))
                sum_y.append(float(sum(vals)))

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
                window_size=self.window_size,
                steps_used=len(steps),
                median_total=median_total,
                worst_total=worst_total,
                worst_rank=worst_rank,
                skew_ratio=skew_ratio,
                skew_pct=skew_pct,
            ),
            coverage=coverage,
        )
        return metric, per_rank_sum


    @staticmethod
    def _common_window_steps(
        per_rank_step_maps: Iterable[Dict[int, float]],
        completed_step: int,
        window_size: int,
    ) -> List[int]:
        """
        Intersect step indices across ranks up to completed_step.

        Returns the last `window_size` common steps, sorted ascending.
        """
        common: Optional[set[int]] = None
        for step_map in per_rank_step_maps:
            steps = {s for s in step_map.keys() if s <= completed_step}
            common = steps if common is None else (common & steps)

        if not common:
            return []

        return sorted(common)[-window_size:]

    def _derive_wait_proxy(
        self,
        *,
        step_metric: Optional[StepCombinedTimeMetric],
        gpu_metrics: Dict[str, StepCombinedTimeMetric],
        per_metric_rank_sums: Dict[str, Dict[int, float]],
    ) -> Tuple[Optional[StepCombinedTimeMetric], Optional[Dict[int, float]]]:
        """
        Derive WAIT proxy as a residual:

            wait_proxy = step_time - (forward + backward + optimizer_step)

        Returns
        -------
        wait_metric:
            StepCombinedTimeMetric (clock="mixed") or None.
        wait_rank_sums:
            Dict[rank -> ΣK(ms)] residuals for heatmap, or None.
        """
        if step_metric is None or not gpu_metrics:
            return None, None

        # Summary-level residual (clamped)
        gpu_median = float(sum(m.summary.median_total for m in gpu_metrics.values()))
        gpu_worst = float(sum(m.summary.worst_total for m in gpu_metrics.values()))

        wait_median_total = max(0.0, step_metric.summary.median_total - gpu_median)
        wait_worst_total = max(0.0, step_metric.summary.worst_total - gpu_worst)

        ranks_present = step_metric.coverage.ranks_present
        if ranks_present <= 1:
            skew_ratio = 0.0
            skew_pct = 0.0
        else:
            skew_ratio = wait_worst_total / wait_median_total if wait_median_total > 0 else 0.0
            skew_pct = (
                (wait_worst_total - wait_median_total) / wait_median_total
                if wait_median_total > 0
                else 0.0
            )

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
                skew_ratio=skew_ratio,
                skew_pct=skew_pct,
            ),
            coverage=step_metric.coverage,
        )

        # Rank-level residual for heatmap (optional)
        step_sums = per_metric_rank_sums.get(WAIT_STEP_KEY)
        fwd_sums = per_metric_rank_sums.get("forward")
        bwd_sums = per_metric_rank_sums.get("backward")
        opt_sums = per_metric_rank_sums.get("optimizer_step")

        if not (step_sums and fwd_sums and bwd_sums and opt_sums):
            return wait_metric, None

        wait_rank_sums: Dict[int, float] = {}
        for r, step_sum in step_sums.items():
            gpu_sum = float(
                fwd_sums.get(r, 0.0) + bwd_sums.get(r, 0.0) + opt_sums.get(r, 0.0)
            )
            wait_rank_sums[r] = float(max(0.0, step_sum - gpu_sum))

        return wait_metric, wait_rank_sums

    def _build_rank_heatmap(
        self,
        *,
        coverage: StepCombinedTimeCoverage,
        per_metric_rank_sums: Dict[str, Dict[int, float]],
    ) -> Optional[StepCombinedRankHeatmap]:
        """
        Build dashboard heatmap payload (rank x metric window sums).
        Sorting
        -------
        Rows are sorted by:
          (step_time desc, dataloader_fetch desc)

        Missing metrics/ranks are filled with 0.0 ms.
        """
        if coverage.ranks_present == 0:
            return None

        keys = [k for k in self._heatmap_keys if k in per_metric_rank_sums or k == WAIT_METRIC_KEY]

        ref = per_metric_rank_sums.get(WAIT_STEP_KEY) or next(iter(per_metric_rank_sums.values()), {})
        ranks = list(ref.keys())

        rows: List[StepCombinedRankRow] = []
        for r in ranks:
            sums_ms: Dict[str, float] = {}
            for k in keys:
                sums_ms[k] = float(per_metric_rank_sums.get(k, {}).get(r, 0.0))
            rows.append(StepCombinedRankRow(rank=r, sums_ms=sums_ms))

        def _sort_key(row: StepCombinedRankRow) -> Tuple[float, float]:
            return (
                row.sums_ms.get(WAIT_STEP_KEY, 0.0),
                row.sums_ms.get("dataloader_fetch", 0.0),
            )

        rows.sort(key=_sort_key, reverse=True)

        return StepCombinedRankHeatmap(
            window_size=self.window_size,
            steps_used=coverage.steps_used,
            metric_keys=keys,
            rows=rows,
            sort_by=[WAIT_STEP_KEY, "dataloader_fetch"],
        )