"""
Step Combined Time Computation (DDP-aware, rank-agnostic)

Pure compute layer for TraceML step combined time metrics.

- Reads TimeEventSample from RemoteDBStore
- Clock source (CPU/GPU) is derived from data (sample.is_gpu)
- Aggregates per-step across ranks
- Outputs renderer-facing StepCombinedTimeResult

Derived metrics
---------------
This computer may also emit a derived "wait proxy" metric:

    wait_proxy = step_time_cpu - (forward_gpu + backward_gpu + optimizer_gpu)

This is a mixed-clock residual (CPU wall - GPU stream time). It is intentionally
labeled as `clock="mixed"` and should be presented as a *proxy*, not as a precise
breakdown of wall time.
"""

from typing import Dict, List, Optional
import numpy as np

from traceml.database.remote_database_store import RemoteDBStore
from traceml.loggers.error_log import get_error_logger
from traceml.samplers.schema.time_schema import TimeEventSample

from traceml.renderers.step_combined.schema import (
    StepCombinedTimeResult,
    StepCombinedTimeMetric,
    StepCombinedTimeSeries,
    StepCombinedTimeSummary,
    StepCombinedTimeCoverage,
)


class StepCombinedComputer:
    """
    Rank-agnostic computation of step combined time metrics.
    """

    def __init__(
        self,
        remote_store: RemoteDBStore,
        *,
        window_size: int = 100,
    ):
        self.store = remote_store
        self.window_size = int(window_size)
        self.logger = get_error_logger("StepCombinedComputer")

        # Table-driven only — NO CPU/GPU assumptions here
        self._metrics = [
            ("_traceml_internal:dataloader_next", "TimeSampler", "dataloader_fetch"),
            ("_traceml_internal:forward_time", "TimeSampler", "forward"),
            ("_traceml_internal:backward_time", "TimeSampler", "backward"),
            ("_traceml_internal:optimizer_step", "TimeSampler", "optimizer_step"),
            ("_traceml_internal:step_time", "TimeSampler", "step_time_ms"),
        ]

    def compute(self) -> StepCombinedTimeResult:
        """
        Compute all step combined time metrics.
        """
        ranks = list(self.store.ranks())
        world_size = len(ranks)

        if world_size == 0:
            return StepCombinedTimeResult(metrics=[], status_message="No ranks available")

        computed: Dict[str, StepCombinedTimeMetric] = {}

        for table, sampler, metric_key in self._metrics:
            metric = self._compute_one_metric(
                table=table,
                sampler=sampler,
                metric_key=metric_key,
                ranks=ranks,
                world_size=world_size,
            )
            if metric is not None:
                computed[metric_key] = metric

        wait_metric = self._derive_wait_proxy(
            step_metric=computed.get("step_time_ms"),
            gpu_metrics={
                k: v for k, v in computed.items()
                if k in {"forward", "backward", "optimizer_step"}
            },
        )
        if wait_metric:
            computed["wait_proxy"] = wait_metric

        status = "OK" if computed else "No complete metrics available"

        return StepCombinedTimeResult(
            metrics=list(computed.values()),
            status_message=status,
        )

    def _compute_one_metric(
        self,
        *,
        table: str,
        sampler: str,
        metric_key: str,
        ranks: List[int],
        world_size: int,
    ) -> Optional[StepCombinedTimeMetric]:
        """
        Compute a single step combined metric.
        """
        per_rank_steps: Dict[int, Dict[int, float]] = {}
        per_rank_clocks: Dict[int, str] = {}
        completed_steps: List[int] = []

        for rank in ranks:
            db = self.store.get_db(rank, sampler)
            if not db:
                continue

            tbl = db.get_table(table)
            if not tbl:
                continue

            step_map: Dict[int, float] = {}
            clock: Optional[str] = None

            for row in reversed(tbl):
                try:
                    s = TimeEventSample.from_wire(row)
                except Exception:
                    continue

                step_map[s.step] = s.duration_ms
                clock = "gpu" if s.is_gpu else "cpu"

                if len(step_map) >= self.window_size:
                    break

            if step_map:
                per_rank_steps[rank] = step_map
                per_rank_clocks[rank] = clock or "cpu"
                completed_steps.append(max(step_map.keys()))

        if not per_rank_steps:
            return None

        completed_step = min(completed_steps)
        ranks_present = len(per_rank_steps)

        # Intersect steps across ranks
        common_steps = None
        for step_map in per_rank_steps.values():
            steps = {s for s in step_map.keys() if s <= completed_step}
            common_steps = steps if common_steps is None else common_steps & steps

        if not common_steps:
            return None

        steps = sorted(common_steps)[-self.window_size:]

        median_y: List[float] = []
        worst_y: List[float] = []
        sum_y: List[float] = []

        for s in steps:
            vals = [per_rank_steps[r][s] for r in per_rank_steps]
            vals.sort()
            median_y.append(float(np.median(vals)))
            worst_y.append(float(vals[-1]))
            sum_y.append(float(sum(vals)))

        # Window summary
        per_rank_sum = {
            r: sum(per_rank_steps[r][s] for s in steps)
            for r in per_rank_steps
        }

        sums = list(per_rank_sum.values())
        median_total = float(np.median(sums))
        worst_total = float(max(sums))
        worst_rank = max(per_rank_sum, key=lambda r: per_rank_sum[r])

        skew_ratio = worst_total / median_total if median_total > 0 else 0.0
        skew_pct = (worst_total - median_total) / median_total if median_total > 0 else 0.0

        # Derive clock from data (majority vote)
        clocks = list(per_rank_clocks.values())
        clock = max(set(clocks), key=clocks.count)

        incomplete = ranks_present < world_size

        return StepCombinedTimeMetric(
            metric=metric_key,
            clock=clock,
            series=StepCombinedTimeSeries(
                steps=steps,
                median=median_y,
                worst=worst_y,
                sum=sum_y,
            ),
            summary=StepCombinedTimeSummary(
                window_size=self.window_size,
                steps_used=len(steps),
                median_total=median_total,
                worst_total=worst_total,
                worst_rank=worst_rank,
                skew_ratio=skew_ratio,
                skew_pct=skew_pct,
            ),
            coverage=StepCombinedTimeCoverage(
                expected_steps=self.window_size,
                steps_used=len(steps),
                completed_step=completed_step,
                world_size=world_size,
                ranks_present=ranks_present,
                incomplete=incomplete,
            ),
        )


    def _derive_wait_proxy(
            self,
            *,
            step_metric: StepCombinedTimeMetric,
            gpu_metrics: Dict[str, StepCombinedTimeMetric],
    ) -> Optional[StepCombinedTimeMetric]:
        """
        Derive WAIT (proxy) metric as a mixed-clock residual.

        Definition:
            WAIT = step_time_cpu − (forward + backward + optimizer_step)_gpu

        Notes:
        - Uses mixed CPU/GPU clocks
        - Rough approximation (sync, comm, overlap, CPU wait)
        - Intended for straggler analysis, not precise accounting
        """

        if not step_metric or not gpu_metrics:
            return None

        steps = step_metric.series.steps
        if not steps:
            return None

        # Aggregate GPU compute per step
        gpu_median = [0.0] * len(steps)
        gpu_worst = [0.0] * len(steps)
        gpu_sum = [0.0] * len(steps)

        for m in gpu_metrics.values():
            if m.series.steps != steps:
                return None  # defensive: step mismatch

            for i in range(len(steps)):
                gpu_median[i] += m.series.median[i]
                gpu_worst[i] += m.series.worst[i]
                gpu_sum[i] += m.series.sum[i]

        # Residual (clamped to avoid negative noise)
        wait_median = [
            max(0.0, step_metric.series.median[i] - gpu_median[i])
            for i in range(len(steps))
        ]
        wait_worst = [
            max(0.0, step_metric.series.worst[i] - gpu_worst[i])
            for i in range(len(steps))
        ]
        wait_sum = [
            max(0.0, step_metric.series.sum[i] - gpu_sum[i])
            for i in range(len(steps))
        ]

        # Window summary
        median_total = float(np.median(wait_sum))
        worst_total = float(max(wait_sum))
        worst_rank = step_metric.summary.worst_rank

        skew_ratio = worst_total / median_total if median_total > 0 else 0.0
        skew_pct = (
            (worst_total - median_total) / median_total
            if median_total > 0 else 0.0
        )

        return StepCombinedTimeMetric(
            metric="wait_proxy",
            clock="mixed",
            series=StepCombinedTimeSeries(
                steps=steps,
                median=wait_median,
                worst=wait_worst,
                sum=wait_sum,
            ),
            summary=StepCombinedTimeSummary(
                window_size=step_metric.summary.window_size,
                steps_used=len(steps),
                median_total=median_total,
                worst_total=worst_total,
                worst_rank=worst_rank,
                skew_ratio=skew_ratio,
                skew_pct=skew_pct,
            ),
            coverage=step_metric.coverage,
        )
