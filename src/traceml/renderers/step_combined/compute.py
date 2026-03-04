import time
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
from traceml.samplers.schema.step_time_schema import StepTimeEventSample

STEP_TIME_SAMPLER = "StepTimeSampler"
STEP_TIME_TABLE = "StepTimeTable"

WAIT_METRIC_KEY = "wait_proxy"
WAIT_STEP_KEY = "step_time"
WAIT_GPU_KEYS: Tuple[str, ...] = ("forward", "backward", "optimizer_step")

EVENT_ALIASES = {
    "dataloader_fetch": "_traceml_internal:dataloader_next",
    "forward": "_traceml_internal:forward_time",
    "backward": "_traceml_internal:backward_time",
    "optimizer_step": "_traceml_internal:optimizer_step",
    "step_time": "_traceml_internal:step_time",
}

DEFAULT_METRIC_KEYS: Tuple[str, ...] = (
    "dataloader_fetch",
    "forward",
    "backward",
    "optimizer_step",
    "step_time",
)

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
    Compute step-combined time metrics across ranks from StepTimeSampler output.

    Data source (must match sampler):
      sampler = "StepTimeSampler"
      table   = "StepTimeTable"

    Each row is StepTimeEventSample (one row per step). Its payload:
      events[event_name][device] = {"duration_ms": float, "n_calls": int, "is_gpu": bool}

    Metric value definition:
      value(step, metric_key) = SUM(duration_ms over all devices for event_name == metric_key)

    Semantics preserved:
    - Uses the last K steps common across ranks (suffix intersection).
    - Computes per-rank window sums, then median/worst/skew.
    - WAIT proxy = step_time - (forward + backward + optimizer_step) (clamped at 0).
    - Optional dashboard rank heatmap (uses per-rank window sums).
    - Last-good cache + TTL to avoid blank UI on transient emptiness/failures.
    """

    def __init__(
        self,
        remote_store: RemoteDBStore,
        *,
        window_size: int = 100,
        metric_keys: Optional[Sequence[str]] = None,
        heatmap_keys: Optional[Sequence[str]] = None,
        stale_ttl_s: Optional[float] = 30.0,
        sampler: str = STEP_TIME_SAMPLER,
        table: str = STEP_TIME_TABLE,
    ) -> None:
        self.store = remote_store
        self.window_size = int(window_size)
        self.metric_keys = (
            list(metric_keys)
            if metric_keys is not None
            else list(DEFAULT_METRIC_KEYS)
        )
        self.heatmap_keys = (
            list(heatmap_keys)
            if heatmap_keys is not None
            else list(DEFAULT_HEATMAP_KEYS)
        )
        self.sampler = str(sampler)
        self.table = str(table)

        self.logger = get_error_logger("StepCombinedComputer")
        self._wait_gpu_set = set(WAIT_GPU_KEYS)

        self._last_ok: Optional[StepCombinedTimeResult] = None
        self._last_ok_ts: float = 0.0
        self._stale_ttl_s: Optional[float] = (
            float(stale_ttl_s) if stale_ttl_s is not None else None
        )

    def compute_cli(self) -> StepCombinedTimeResult:
        return self._compute(include_series=False, include_rank_heatmap=False)

    def compute_dashboard(self) -> StepCombinedTimeResult:
        return self._compute(include_series=False, include_rank_heatmap=True)

    def _compute(
        self, *, include_series: bool, include_rank_heatmap: bool
    ) -> StepCombinedTimeResult:
        try:
            res = self._compute_impl(
                include_series=include_series,
                include_rank_heatmap=include_rank_heatmap,
            )
        except Exception as e:
            self.logger.exception("StepCombined compute failed")
            return self._stale_or_empty(
                f"STALE (exception: {type(e).__name__})"
            )

        if not res.metrics:
            return self._stale_or_empty("STALE (no metrics this tick)")

        self._last_ok = res
        self._last_ok_ts = time.time()
        return res

    def _stale_or_empty(self, msg: str) -> StepCombinedTimeResult:
        now = time.time()
        if self._last_ok is not None:
            if (
                self._stale_ttl_s is None
                or (now - self._last_ok_ts) <= self._stale_ttl_s
            ):
                return StepCombinedTimeResult(
                    metrics=self._last_ok.metrics,
                    status_message=msg,
                    rank_heatmap=self._last_ok.rank_heatmap,
                )
        return StepCombinedTimeResult(
            metrics=[],
            status_message="No fresh step-combined data",
            rank_heatmap=None,
        )

    def _compute_impl(
        self, *, include_series: bool, include_rank_heatmap: bool
    ) -> StepCombinedTimeResult:
        ranks = list(self.store.ranks())
        world_size = len(ranks)
        if world_size == 0:
            return StepCombinedTimeResult(
                metrics=[],
                status_message="No ranks available",
                rank_heatmap=None,
            )

        per_rank_steps = self._load_last_steps(ranks)
        if not per_rank_steps:
            return StepCombinedTimeResult(
                metrics=[],
                status_message="No StepTime data available",
                rank_heatmap=None,
            )

        ranks_present = len(per_rank_steps)
        completed_step = min(max(m.keys()) for m in per_rank_steps.values())
        steps = self._common_steps(
            per_rank_steps, completed_step, self.window_size
        )
        if not steps:
            return StepCombinedTimeResult(
                metrics=[],
                status_message="No common step window yet",
                rank_heatmap=None,
            )

        coverage = StepCombinedTimeCoverage(
            expected_steps=self.window_size,
            steps_used=len(steps),
            completed_step=int(completed_step),
            world_size=int(world_size),
            ranks_present=int(ranks_present),
            incomplete=(ranks_present < world_size),
        )

        computed: Dict[str, StepCombinedTimeMetric] = {}
        per_metric_rank_sums: Dict[str, Dict[int, float]] = {}

        # Compute base metrics directly from payloads (no intermediate projection maps)
        for metric_key in self.metric_keys:
            metric, rank_sums = self._metric_from_steps(
                metric_key=metric_key,
                per_rank_steps=per_rank_steps,
                steps=steps,
                coverage=coverage,
                include_series=include_series,
            )
            if metric is not None:
                computed[metric_key] = metric
                per_metric_rank_sums[metric_key] = rank_sums

        # WAIT proxy (same semantics)
        wait_metric, wait_rank_sums = self._wait_proxy(
            computed, per_metric_rank_sums
        )
        if wait_metric is not None:
            computed[WAIT_METRIC_KEY] = wait_metric
            if wait_rank_sums is not None:
                per_metric_rank_sums[WAIT_METRIC_KEY] = wait_rank_sums

        status = "OK" if computed else "No complete metrics available"

        rank_heatmap = None
        if include_rank_heatmap and computed:
            # pick a reference metric for coverage fields (same as before)
            ref = computed.get(WAIT_STEP_KEY) or next(iter(computed.values()))
            rank_heatmap = self._rank_heatmap(
                ref.coverage, per_metric_rank_sums
            )

        return StepCombinedTimeResult(
            metrics=list(computed.values()),
            status_message=status,
            rank_heatmap=rank_heatmap,
        )

    def _load_last_steps(self, ranks: List[int]) -> Dict[int, Dict[int, Dict]]:
        """
        Load last `window_size` step rows per rank.

        Returns:
          per_rank_steps[rank][step] = events_payload
        """
        out: Dict[int, Dict[int, Dict]] = {}
        ws = self.window_size
        from_wire = StepTimeEventSample.from_wire

        for r in ranks:
            db = self.store.get_db(r, self.sampler)
            if not db:
                continue
            tbl = db.get_table(self.table)
            if not tbl:
                continue

            step_map: Dict[int, Dict] = {}

            try:
                n = len(tbl)
                it = (tbl[i] for i in range(n - 1, -1, -1))
            except Exception:
                it = reversed(tbl)

            for row in it:
                try:
                    s = from_wire(row)
                except Exception:
                    continue
                step_map[int(s.step)] = s.events
                if len(step_map) >= ws:
                    break

            if step_map:
                out[int(r)] = step_map

        return out

    @staticmethod
    def _common_steps(
        per_rank_steps: Dict[int, Dict[int, Dict]],
        completed_step: int,
        window_size: int,
    ) -> List[int]:
        """Common suffix of steps present for all ranks, ascending, limited to last K."""
        maps = list(per_rank_steps.values())
        if not maps:
            return []
        ref = maps[0]

        out_rev: List[int] = []
        s = int(completed_step)
        while s >= 0 and len(out_rev) < int(window_size):
            if s in ref and all(s in m for m in maps[1:]):
                out_rev.append(s)
            s -= 1

        out_rev.reverse()
        return out_rev

    @staticmethod
    def _event_total_ms(events_payload: Dict, metric_key: str) -> float:
        payload_key = EVENT_ALIASES.get(metric_key, metric_key)
        ev = events_payload.get(payload_key)
        if not ev:
            return 0.0
        return float(
            sum(float(rec.get("duration_ms", 0.0)) for rec in ev.values())
        )

    def _metric_from_steps(
        self,
        *,
        metric_key: str,
        per_rank_steps: Dict[int, Dict[int, Dict]],
        steps: Sequence[int],
        coverage: StepCombinedTimeCoverage,
        include_series: bool,
    ) -> Tuple[Optional[StepCombinedTimeMetric], Dict[int, float]]:
        """
        Compute one StepCombinedTimeMetric from step payloads.

        Returns (metric_or_none, per_rank_window_sums_ms).
        """
        ranks = list(per_rank_steps.keys())
        if not ranks:
            return None, {}

        # Per-rank window totals
        per_rank_sum: Dict[int, float] = {}
        sums_arr = np.empty(len(ranks), dtype=np.float64)

        for i, r in enumerate(ranks):
            step_map = per_rank_steps[r]
            total = 0.0
            for s in steps:
                total += self._event_total_ms(
                    step_map.get(int(s), {}), metric_key
                )
            per_rank_sum[int(r)] = float(total)
            sums_arr[i] = total

        median_total = float(np.median(sums_arr))
        worst_idx = int(np.argmax(sums_arr))
        worst_total = float(sums_arr[worst_idx])
        worst_rank = int(ranks[worst_idx])

        if coverage.ranks_present <= 1:
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

        series: Optional[StepCombinedTimeSeries] = None
        if include_series:
            # Per-step reductions across ranks
            median_y: List[float] = []
            worst_y: List[float] = []
            sum_y: List[float] = []

            for s in steps:
                v = np.fromiter(
                    (
                        self._event_total_ms(
                            per_rank_steps[r].get(int(s), {}), metric_key
                        )
                        for r in ranks
                    ),
                    dtype=np.float64,
                )
                median_y.append(float(np.median(v)))
                worst_y.append(float(np.max(v)))
                sum_y.append(float(np.sum(v)))

            series = StepCombinedTimeSeries(
                steps=list(steps), median=median_y, worst=worst_y, sum=sum_y
            )

        metric = StepCombinedTimeMetric(
            metric=str(metric_key),
            clock="mixed",  # payload can include CPU+GPU contributions per event
            series=series,
            summary=StepCombinedTimeSummary(
                window_size=int(coverage.expected_steps),
                steps_used=int(coverage.steps_used),
                median_total=float(median_total),
                worst_total=float(worst_total),
                worst_rank=int(worst_rank),
                skew_ratio=float(skew_ratio),
                skew_pct=float(skew_pct),
            ),
            coverage=coverage,
        )
        return metric, per_rank_sum

    def _wait_proxy(
        self,
        computed: Dict[str, StepCombinedTimeMetric],
        per_metric_rank_sums: Dict[str, Dict[int, float]],
    ) -> Tuple[Optional[StepCombinedTimeMetric], Optional[Dict[int, float]]]:
        step_metric = computed.get(WAIT_STEP_KEY)
        if step_metric is None:
            return None, None

        gpu_metrics = {
            k: computed[k] for k in computed.keys() if k in self._wait_gpu_set
        }
        if not gpu_metrics:
            return None, None

        gpu_median = float(
            sum(m.summary.median_total for m in gpu_metrics.values())
        )
        gpu_worst = float(
            sum(m.summary.worst_total for m in gpu_metrics.values())
        )

        wait_median_total = max(
            0.0, step_metric.summary.median_total - gpu_median
        )
        wait_worst_total = max(
            0.0, step_metric.summary.worst_total - gpu_worst
        )

        ranks_present = step_metric.coverage.ranks_present
        if ranks_present <= 1:
            skew_ratio = 0.0
            skew_pct = 0.0
        else:
            if wait_median_total > 0.0:
                skew_ratio = wait_worst_total / wait_median_total
                skew_pct = (
                    wait_worst_total - wait_median_total
                ) / wait_median_total
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
                median_total=float(wait_median_total),
                worst_total=float(wait_worst_total),
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
            gpu_sum = float(
                fwd_sums.get(r, 0.0)
                + bwd_sums.get(r, 0.0)
                + opt_sums.get(r, 0.0)
            )
            wait_rank_sums[int(r)] = float(max(0.0, float(step_sum) - gpu_sum))

        return wait_metric, wait_rank_sums

    def _rank_heatmap(
        self,
        coverage: StepCombinedTimeCoverage,
        per_metric_rank_sums: Dict[str, Dict[int, float]],
    ) -> Optional[StepCombinedRankHeatmap]:
        if coverage.ranks_present == 0:
            return None

        keys = [
            k
            for k in self.heatmap_keys
            if (k in per_metric_rank_sums or k == WAIT_METRIC_KEY)
        ]
        ref = per_metric_rank_sums.get(WAIT_STEP_KEY) or next(
            iter(per_metric_rank_sums.values()), {}
        )
        ranks = list(ref.keys())

        rows: List[StepCombinedRankRow] = [
            StepCombinedRankRow(
                rank=int(r),
                sums_ms={
                    k: float(per_metric_rank_sums.get(k, {}).get(r, 0.0))
                    for k in keys
                },
            )
            for r in ranks
        ]

        rows.sort(
            key=lambda row: (
                row.sums_ms.get(WAIT_STEP_KEY, 0.0),
                row.sums_ms.get("dataloader_fetch", 0.0),
            ),
            reverse=True,
        )

        return StepCombinedRankHeatmap(
            window_size=self.window_size,
            steps_used=coverage.steps_used,
            metric_keys=keys,
            rows=rows,
            sort_by=[WAIT_STEP_KEY, "dataloader_fetch"],
        )
