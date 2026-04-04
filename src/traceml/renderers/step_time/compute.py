import json
import sqlite3
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from traceml.loggers.error_log import get_error_logger
from traceml.renderers.step_time.schema import (
    StepCombinedRankHeatmap,
    StepCombinedRankRow,
    StepCombinedTimeCoverage,
    StepCombinedTimeMetric,
    StepCombinedTimeResult,
    StepCombinedTimeSeries,
    StepCombinedTimeSummary,
)

STEP_TIME_TABLE = "step_time_samples"

WAIT_METRIC_KEY = "wait_proxy"
WAIT_STEP_KEY = "step_time"

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
    Compute rank-combined step timing summaries directly from SQLite.

    Data source
    -----------
    Reads from the SQLite projection table `step_time_samples`, which contains:
    - rank
    - step
    - sample_ts_s
    - seq
    - events_json

    Output compatibility
    --------------------
    Output objects are intentionally kept identical to the previous
    RemoteDBStore-backed implementation so existing renderers can continue
    to work without modification.

    Design goals
    ------------
    - Avoid dependence on RemoteDBStore
    - Keep reads bounded to recent rows for predictable latency
    - Preserve stale-cache behavior to avoid dashboard flicker
    - Keep logic close to the previous implementation for easier review

    Notes
    -----
    - The compute still uses the last common suffix of steps across ranks.
    - Dynamic event names remain inside JSON and are parsed only for the
      recent rows involved in the current compute window.
    """

    def __init__(
        self,
        db_path: str,
        window_size: int = 100,
        metric_keys: Optional[Sequence[str]] = None,
        heatmap_keys: Optional[Sequence[str]] = None,
        stale_ttl_s: Optional[float] = 30.0,
        table: str = STEP_TIME_TABLE,
        lookback_factor: int = 4,
        rank_filter: Optional[Sequence[int]] = None,
    ) -> None:
        self.db_path = str(db_path)
        self.window_size = max(1, int(window_size))
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
        self.table = str(table)
        self.lookback_factor = max(1, int(lookback_factor))
        self.rank_filter = (
            {int(r) for r in rank_filter} if rank_filter is not None else None
        )

        self.logger = get_error_logger("StepCombinedComputer")
        self._last_ok: Optional[StepCombinedTimeResult] = None
        self._last_ok_ts = 0.0
        self._stale_ttl_s = (
            float(stale_ttl_s) if stale_ttl_s is not None else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_cli(self) -> StepCombinedTimeResult:
        return self._compute(include_series=True, include_rank_heatmap=False)

    def compute_dashboard(self) -> StepCombinedTimeResult:
        return self._compute(include_series=False, include_rank_heatmap=True)

    def _compute(
        self, *, include_series: bool, include_rank_heatmap: bool
    ) -> StepCombinedTimeResult:
        try:
            with self._connect() as conn:
                result = self._compute_impl(
                    conn,
                    include_series=include_series,
                    include_rank_heatmap=include_rank_heatmap,
                )
        except Exception as exc:
            self.logger.exception("StepCombined compute failed")
            return self._stale_or_empty(
                f"STALE (exception: {type(exc).__name__})"
            )

        if not result.metrics:
            return self._stale_or_empty("STALE (no metrics this tick)")

        self._last_ok = result
        self._last_ok_ts = time.time()
        return result

    # ------------------------------------------------------------------
    # Core compute
    # ------------------------------------------------------------------

    def _compute_impl(
        self,
        conn: sqlite3.Connection,
        *,
        include_series: bool,
        include_rank_heatmap: bool,
    ) -> StepCombinedTimeResult:
        ranks = self._load_ranks(conn)
        if not ranks:
            return StepCombinedTimeResult(
                metrics=[],
                status_message="No ranks available",
                rank_heatmap=None,
            )

        per_rank_steps = self._load_last_steps(conn, ranks)
        if not per_rank_steps:
            return StepCombinedTimeResult(
                metrics=[],
                status_message="No StepTime data available",
                rank_heatmap=None,
            )

        max_steps = [max(m.keys()) for m in per_rank_steps.values() if m]
        if not max_steps:
            return StepCombinedTimeResult(
                metrics=[],
                status_message="No usable StepTime data available",
                rank_heatmap=None,
            )

        completed_step = min(max_steps)
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
            world_size=len(ranks),
            ranks_present=len(per_rank_steps),
            incomplete=(len(per_rank_steps) < len(ranks)),
        )

        ranks_present = list(per_rank_steps.keys())
        per_metric_rank_sums = self._build_metric_sums(
            per_rank_steps, steps, self.metric_keys
        )

        step_sums = per_metric_rank_sums.get("step_time", {})
        fwd_sums = per_metric_rank_sums.get("forward", {})
        bwd_sums = per_metric_rank_sums.get("backward", {})
        opt_sums = per_metric_rank_sums.get("optimizer_step", {})

        wait_rank_sums = {
            r: max(
                0.0,
                step_sums.get(r, 0.0)
                - fwd_sums.get(r, 0.0)
                - bwd_sums.get(r, 0.0)
                - opt_sums.get(r, 0.0),
            )
            for r in ranks_present
        }
        per_metric_rank_sums[WAIT_METRIC_KEY] = wait_rank_sums

        overall_rank_scores = self._overall_rank_scores(
            per_metric_rank_sums, ranks_present
        )
        overall_worst_rank = (
            max(overall_rank_scores, key=overall_rank_scores.get)
            if overall_rank_scores
            else None
        )
        overall_median_rank = (
            self._median_rank(overall_rank_scores)
            if overall_rank_scores
            else None
        )

        metrics: Dict[str, StepCombinedTimeMetric] = {}
        for metric_key in self.metric_keys:
            rank_sums = per_metric_rank_sums.get(metric_key, {})
            metric = self._make_metric(
                metric_key=metric_key,
                rank_sums=rank_sums,
                ranks=ranks_present,
                coverage=coverage,
                include_series=include_series,
                per_rank_steps=per_rank_steps,
                steps=steps,
            )
            if metric is not None:
                if (
                    metric_key == WAIT_STEP_KEY
                    and overall_worst_rank is not None
                ):
                    metric = StepCombinedTimeMetric(
                        metric=metric.metric,
                        clock=metric.clock,
                        series=metric.series,
                        coverage=metric.coverage,
                        summary=StepCombinedTimeSummary(
                            window_size=metric.summary.window_size,
                            steps_used=metric.summary.steps_used,
                            median_total=metric.summary.median_total,
                            worst_total=metric.summary.worst_total,
                            worst_rank=int(overall_worst_rank),
                            skew_ratio=metric.summary.skew_ratio,
                            skew_pct=metric.summary.skew_pct,
                        ),
                    )
                metrics[metric_key] = metric

        wait_metric = self._make_metric(
            metric_key=WAIT_METRIC_KEY,
            rank_sums=wait_rank_sums,
            ranks=ranks_present,
            coverage=coverage,
            include_series=False,
            per_rank_steps=per_rank_steps,
            steps=steps,
        )
        if wait_metric is not None:
            metrics[WAIT_METRIC_KEY] = wait_metric

        rank_heatmap = None
        if include_rank_heatmap and metrics:
            keys = [k for k in self.heatmap_keys if k in per_metric_rank_sums]
            rows = [
                StepCombinedRankRow(
                    rank=int(r),
                    sums_ms={
                        k: float(per_metric_rank_sums.get(k, {}).get(r, 0.0))
                        for k in keys
                    },
                )
                for r in ranks_present
            ]
            if overall_rank_scores:
                rows.sort(
                    key=lambda row: (
                        overall_rank_scores.get(row.rank, 0.0),
                        row.sums_ms.get("step_time", 0.0),
                        row.sums_ms.get("dataloader_fetch", 0.0),
                    ),
                    reverse=True,
                )
                sort_by = ["overall_score", "step_time", "dataloader_fetch"]
            else:
                rows.sort(
                    key=lambda row: (
                        row.sums_ms.get("step_time", 0.0),
                        row.sums_ms.get("dataloader_fetch", 0.0),
                    ),
                    reverse=True,
                )
                sort_by = ["step_time", "dataloader_fetch"]

            rank_heatmap = StepCombinedRankHeatmap(
                window_size=self.window_size,
                steps_used=coverage.steps_used,
                metric_keys=keys,
                rows=rows,
                sort_by=sort_by,
            )

        status = "OK"
        if overall_worst_rank is not None:
            status += f" | overall_worst_rank=r{overall_worst_rank}"
        if overall_median_rank is not None:
            status += f" | overall_median_rank=r{overall_median_rank}"

        return StepCombinedTimeResult(
            metrics=list(metrics.values()),
            status_message=status,
            rank_heatmap=rank_heatmap,
        )

    # ------------------------------------------------------------------
    # SQLite loading
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """
        Open a short-lived read connection.

        One connection per compute call keeps the implementation simple and
        avoids cross-thread SQLite issues.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_ranks(self, conn: sqlite3.Connection) -> List[int]:
        """
        Load available ranks from the step-time table.

        Returns only non-null ranks, optionally filtered by `rank_filter`.
        """
        rows = conn.execute(
            f"""
            SELECT DISTINCT rank
            FROM {self.table}
            WHERE rank IS NOT NULL
            ORDER BY rank ASC;
            """
        ).fetchall()

        ranks = [int(r["rank"]) for r in rows if r["rank"] is not None]
        if self.rank_filter is not None:
            ranks = [r for r in ranks if r in self.rank_filter]
        return ranks

    def _load_last_steps(
        self,
        conn: sqlite3.Connection,
        ranks: List[int],
    ) -> Dict[int, Dict[int, Dict[str, Any]]]:
        """
        Load a bounded recent step map per rank.

        To avoid scanning the full table, this method only reads a recent
        suffix per rank. The lookback is intentionally larger than the final
        window size so the common-step suffix has room to form even when some
        ranks are slightly ahead/behind.

        Returns
        -------
        dict[int, dict[int, dict]]
            rank -> step -> events_payload
        """
        out: Dict[int, Dict[int, Dict[str, Any]]] = {}
        lookback = max(
            self.window_size * self.lookback_factor, self.window_size
        )

        for rank in ranks:
            try:
                rows = conn.execute(
                    f"""
                    SELECT step, events_json
                    FROM (
                        SELECT step, events_json
                        FROM {self.table}
                        WHERE rank = ?
                        ORDER BY step DESC, id DESC
                        LIMIT ?
                    )
                    ORDER BY step ASC;
                    """,
                    (int(rank), int(lookback)),
                ).fetchall()

                if not rows:
                    continue

                step_map: Dict[int, Dict[str, Any]] = {}
                for row in rows:
                    step_raw = row["step"]
                    if step_raw is None:
                        continue

                    step = int(step_raw)
                    # If duplicates somehow exist, keep the latest from the DESC/limit
                    # selection; the final ORDER BY ASC preserves chronological order.
                    if step in step_map:
                        continue

                    events_json = row["events_json"]
                    try:
                        parsed = json.loads(events_json) if events_json else {}
                    except Exception:
                        parsed = {}

                    step_map[step] = parsed if isinstance(parsed, dict) else {}

                if step_map:
                    out[int(rank)] = step_map

            except Exception:
                self.logger.exception(
                    "Failed loading step data for rank=%s", rank
                )

        return out

    # ------------------------------------------------------------------
    # Stale handling
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Step alignment helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _common_steps(
        per_rank_steps: Dict[int, Dict[int, Dict[str, Any]]],
        completed_step: int,
        window_size: int,
    ) -> List[int]:
        """
        Return the last common suffix of step ids across all available ranks.
        """
        maps = list(per_rank_steps.values())
        if not maps:
            return []

        out: List[int] = []
        for s in range(int(completed_step), -1, -1):
            if all(s in m for m in maps):
                out.append(s)
                if len(out) >= int(window_size):
                    break
        out.reverse()
        return out

    # ------------------------------------------------------------------
    # Metric extraction / aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(x: Any) -> float:
        try:
            v = float(x)
            return v if np.isfinite(v) else 0.0
        except Exception:
            return 0.0

    @classmethod
    def _event_total_ms(
        cls, events_payload: Dict[str, Any], metric_key: str
    ) -> float:
        """
        Sum a metric across all devices for one step payload.

        The event map is expected to follow the normalized JSON shape stored in
        `events_json`.
        """
        if not isinstance(events_payload, dict):
            return 0.0

        payload = events_payload.get(EVENT_ALIASES.get(metric_key, metric_key))
        if not isinstance(payload, dict):
            return 0.0

        return float(
            sum(
                cls._safe_float(rec.get("duration_ms", 0.0))
                for rec in payload.values()
                if isinstance(rec, dict)
            )
        )

    def _build_metric_sums(
        self,
        per_rank_steps: Dict[int, Dict[int, Dict[str, Any]]],
        steps: Sequence[int],
        metric_keys: Sequence[str],
    ) -> Dict[str, Dict[int, float]]:
        """
        Build per-rank total milliseconds for each metric over the common window.
        """
        out = {k: {} for k in metric_keys}

        for rank, step_map in per_rank_steps.items():
            totals = {k: 0.0 for k in metric_keys}
            for step in steps:
                payload = step_map.get(int(step), {})
                for k in metric_keys:
                    totals[k] += self._event_total_ms(payload, k)

            for k, total in totals.items():
                out[k][int(rank)] = float(total)

        return out

    def _make_metric(
        self,
        *,
        metric_key: str,
        rank_sums: Dict[int, float],
        ranks: List[int],
        coverage: StepCombinedTimeCoverage,
        include_series: bool,
        per_rank_steps: Dict[int, Dict[int, Dict[str, Any]]],
        steps: Sequence[int],
    ) -> Optional[StepCombinedTimeMetric]:
        """
        Build one combined metric object with median/worst summary and
        optional step-wise series.
        """
        if not ranks:
            return None

        arr = np.array(
            [self._safe_float(rank_sums.get(r, 0.0)) for r in ranks],
            dtype=np.float64,
        )
        if arr.size == 0:
            return None

        median_total = float(np.median(arr))
        worst_idx = int(np.argmax(arr))
        worst_total = float(arr[worst_idx])
        worst_rank = int(ranks[worst_idx])

        if coverage.ranks_present <= 1:
            median_total = worst_total
            skew_ratio = 0.0
            skew_pct = 0.0
        elif median_total > 0.0:
            skew_ratio = worst_total / median_total
            skew_pct = (worst_total - median_total) / median_total
        else:
            skew_ratio = 0.0
            skew_pct = 0.0

        series = None
        if include_series and metric_key != WAIT_METRIC_KEY:
            median_y, worst_y, sum_y = [], [], []

            for step in steps:
                vals = np.array(
                    [
                        self._event_total_ms(
                            per_rank_steps[r].get(int(step), {}), metric_key
                        )
                        for r in ranks
                    ],
                    dtype=np.float64,
                )
                median_y.append(float(np.median(vals)) if vals.size else 0.0)
                worst_y.append(float(np.max(vals)) if vals.size else 0.0)
                sum_y.append(float(np.sum(vals)) if vals.size else 0.0)

            series = StepCombinedTimeSeries(
                steps=list(steps),
                median=median_y,
                worst=worst_y,
                sum=sum_y,
            )

        return StepCombinedTimeMetric(
            metric=str(metric_key),
            clock="mixed",
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

    def _overall_rank_scores(
        self,
        per_metric_rank_sums: Dict[str, Dict[int, float]],
        ranks: List[int],
    ) -> Dict[int, float]:
        """
        Compute overall rank "badness" score used for dashboard identity.

        Same definition as before:
            dataloader_fetch + max(step_time, forward + backward + optimizer_step)
        """
        step_sums = per_metric_rank_sums.get("step_time", {})
        dl_sums = per_metric_rank_sums.get("dataloader_fetch", {})
        fwd_sums = per_metric_rank_sums.get("forward", {})
        bwd_sums = per_metric_rank_sums.get("backward", {})
        opt_sums = per_metric_rank_sums.get("optimizer_step", {})

        return {
            int(r): float(
                self._safe_float(dl_sums.get(r, 0.0))
                + max(
                    self._safe_float(step_sums.get(r, 0.0)),
                    self._safe_float(fwd_sums.get(r, 0.0))
                    + self._safe_float(bwd_sums.get(r, 0.0))
                    + self._safe_float(opt_sums.get(r, 0.0)),
                )
            )
            for r in ranks
        }

    def _median_rank(self, rank_scores: Dict[int, float]) -> Optional[int]:
        """
        Return the rank whose score is closest to the median score.
        """
        if not rank_scores:
            return None

        target = float(
            np.median(np.array(list(rank_scores.values()), dtype=np.float64))
        )
        return min(
            rank_scores,
            key=lambda r: (abs(rank_scores[r] - target), rank_scores[r], r),
        )
