# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""What the Process numbers mean.

This layer turns rows into the facts the card states. It owns the
metric decisions and nothing else: the SQL is in ``repository.py`` and the
drawing is in ``process_section.py``.

Semantics, unchanged from 0.3.7:
- seq-aligned across ranks, one entry per globally committed seq
- CPU and RAM are the max across ranks for that step
- the GPU entry comes from the rank with least headroom
- imbalance is the used-bytes spread across reporting ranks
- the card describes the last ``DASHBOARD_WINDOW`` committed steps

The window bound and the percentiles moved here from the section, where
they used to run on every tick inside the view. They are statements about
the metric, so they belong on this side of the boundary.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import replace
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from traceml_ai.renderers.shared.freshness import (
    CachedPayloadTTL,
    FreshnessPolicy,
)
from traceml_ai.renderers.shared.run_series import (
    DEFAULT_RUN_SERIES_POLICY,
    RunSeriesPolicy,
    finite,
    plan_run_series,
)

from .dashboard_models import (
    ChartSeries,
    ChartTrace,
    GpuSnapshot,
    MetricRollup,
    ProcessDashboardPayload,
    ProcessHistoryEntry,
    RankChart,
    RankCoverage,
    RankSnapshot,
    RankTrace,
)
from .repository import ProcessRepository

# How many committed steps the card describes. Kept at the value the
# section applied to its own history slice on 0.3.7.
DASHBOARD_WINDOW = 100

# Samples per rank in the recent-window view.
RANK_WINDOW_N = 100

# A rolling mean over minutes cannot visibly change between two ticks, so
# the whole-run reads refresh on their own slower clock. Recomputing them
# every tick was the single largest cost measured in this block, which is
# the demonstrated need this cache exists for.
RUN_REFRESH_S = 15.0

# Reserved-memory spread at which the per-rank rows have earned opening
# themselves. Below this the rows are a detail; at or above it, WHICH rank
# is holding more is the question the reader now has.
IMBALANCE_OPEN_PCT = 15.0


def percentile(values: Sequence[Optional[float]], p: float) -> float:
    """Linear-interpolated percentile over the values that exist.

    Missing samples are dropped rather than read as zero: a rank that did
    not report is not a rank reporting nothing.
    """
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return 0.0
    k = (len(clean) - 1) * p / 100.0
    low = int(k)
    high = min(low + 1, len(clean) - 1)
    if low == high:
        return float(clean[low])
    return float(clean[low] * (high - k) + clean[high] * (k - low))


def _opt_float(value: Any) -> Optional[float]:
    """A usable number from a database cell, or ``None``."""
    if value is None:
        return None
    try:
        return finite(float(value))
    except (TypeError, ValueError):
        return None


def _median(values: Sequence[float]) -> Optional[float]:
    """The middle value, or ``None`` when there is nothing to take."""
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    middle = len(clean) // 2
    if len(clean) % 2:
        return float(clean[middle])
    return float((clean[middle - 1] + clean[middle]) / 2.0)


def _gpu_reported(row: Any) -> bool:
    """Whether one row carries a usable GPU reading.

    ``gpu_available`` alone is not enough: a rank reports it as true while
    torch is still coming up, before any capacity is known.
    """
    try:
        if not row["gpu_available"]:
            return False
        total = _opt_float(row["gpu_mem_total_bytes"])
        return total is not None and total > 0
    except (KeyError, IndexError, TypeError):
        return False


def _cpu_capacity_of(row: Any) -> Optional[float]:
    """CPU as a share of the host's cores, not a sum across them.

    ``psutil`` reports process CPU summed over cores, so a healthy
    four-core rank reads 291%. Divided by the core count the number is
    bounded and comparable between ranks on different machines.
    """
    used = _opt_float(row["cpu_percent"])
    cores = _opt_float(row["cpu_logical_core_count"])
    if used is None or cores is None or cores <= 0:
        return None
    return used / cores


class ProcessDashboardComputer:
    """Advance the Process history and describe it.

    Parameters
    ----------
    db_path:
        Path to the SQLite database.
    dashboard_max_rows:
        Maximum retained rolling rows for UI history.
    stale_ttl_s:
        How long a cached payload may stand in for a failed read. This is
        a fallback for a broken READ, not a statement about whether the
        telemetry itself is current.
    """

    def __init__(
        self,
        db_path: str,
        dashboard_max_rows: int = 200,
        stale_ttl_s: Optional[float] = 30.0,
        sampler_interval_s: Optional[float] = None,
        run_series_policy: RunSeriesPolicy = DEFAULT_RUN_SERIES_POLICY,
    ) -> None:
        self._db = ProcessRepository(db_path=db_path)
        # The configured cadence, used only until the ranks show their
        # real one. Freshness is then judged on what arrived.
        self._configured_interval_s = sampler_interval_s
        self._run_policy = run_series_policy
        self._cache_ttl = CachedPayloadTTL(ttl_s=stale_ttl_s)
        # Per metric, because the two charts can be in different modes:
        # only a retained chart is worth caching.
        self._run_cache: Dict[str, RankChart] = {}
        self._run_cache_at: Dict[str, float] = {}
        self._dashboard_rollup: Deque[ProcessHistoryEntry] = deque(
            maxlen=max(1, int(dashboard_max_rows))
        )
        self._last_completed_seq: int = -1

        self._last_ok: Optional[ProcessDashboardPayload] = None
        self._last_ok_ts: float = 0.0
        self._stale_ttl_s: Optional[float] = (
            float(stale_ttl_s) if stale_ttl_s is not None else None
        )

    def compute(self) -> ProcessDashboardPayload:
        """Advance the history and return one payload for the card."""
        try:
            with self._db.connect() as conn:
                self._advance(conn)
                newest_ts = self._db.newest_sample_ts(conn)
                ranks, _policy, by_rank = self._rank_snapshots(conn, newest_ts)
                out = self._build_payload(ranks)
                out = self._with_charts(conn, out, by_rank)
        except Exception:
            return self._return_stale()

        self._last_ok = out
        self._last_ok_ts = time.time()
        return out

    # --- advancing -------------------------------------------------------
    def _advance(self, conn: Any) -> None:
        committed_upto = self._db.fetch_committed_seq(conn)
        if (
            committed_upto is None
            or committed_upto <= self._last_completed_seq
        ):
            return

        start_seq = self._last_completed_seq + 1
        end_seq = int(committed_upto)

        for row in self._db.fetch_seq_range_aggregates(
            conn, start_seq=start_seq, end_seq=end_seq
        ):
            self._dashboard_rollup.append(_entry_from_row(row))

        self._last_completed_seq = end_seq

    # --- per-rank facts --------------------------------------------------
    def _rank_snapshots(
        self,
        conn: Any,
        newest_ts: Optional[float],
    ) -> Tuple[
        Tuple[RankSnapshot, ...],
        FreshnessPolicy,
        Dict[int, List[Any]],
    ]:
        """Every rank's own state, read on its own clock.

        Two reads, on purpose. The windowed one carries each rank's recent
        history; the latest-row one carries every rank that has EVER
        reported, so a rank silent for longer than the window still appears
        with its true age instead of vanishing from the block.
        """
        window_rows = self._db.fetch_recent_rank_window(
            conn, window_n=RANK_WINDOW_N, newest_ts=newest_ts
        )
        latest_rows = self._db.fetch_rank_latest(conn)

        by_rank: Dict[int, List[Any]] = {}
        for row in window_rows:
            rank_id = row["global_rank"]
            if rank_id is None:
                rank_id = row["rank"]
            if rank_id is None:
                continue
            by_rank.setdefault(int(rank_id), []).append(row)

        newest_by_rank: Dict[int, Any] = {}
        for row in latest_rows:
            rank_id = row["global_rank"]
            if rank_id is None:
                rank_id = row["rank"]
            if rank_id is not None:
                newest_by_rank[int(rank_id)] = row

        policy = FreshnessPolicy.from_observed_cadence(
            self._observed_cadence(by_rank),
            configured_s=self._configured_interval_s,
        )
        now_s = self._newest_recv(list(newest_by_rank.values()))

        snapshots = [
            self._snapshot_for(
                rank_id,
                by_rank.get(rank_id, []),
                newest_by_rank[rank_id],
                policy=policy,
                now_s=now_s,
            )
            for rank_id in sorted(newest_by_rank)
        ]
        return tuple(snapshots), policy, by_rank

    def _observed_cadence(
        self, by_rank: Dict[int, List[Any]]
    ) -> Optional[float]:
        """The gap the ranks actually sample at, from the busiest rank."""
        best: Optional[float] = None
        for rows in by_rank.values():
            stamps = [
                value
                for value in (_opt_float(r["sample_ts_s"]) for r in rows)
                if value is not None
            ]
            if len(stamps) < 2:
                continue
            span = max(stamps) - min(stamps)
            cadence = finite(span / float(len(stamps) - 1))
            if cadence and cadence > 0:
                best = cadence if best is None else min(best, cadence)
        return best

    def _newest_recv(self, rows: Sequence[Any]) -> float:
        """The aggregator's newest arrival clock, the reference for age."""
        stamps = [
            value / 1e9
            for value in (_opt_float(r["recv_ts_ns"]) for r in rows)
            if value is not None
        ]
        return max(stamps) if stamps else 0.0

    def _snapshot_for(
        self,
        rank_id: int,
        rows: List[Any],
        newest_row: Any,
        *,
        policy: FreshnessPolicy,
        now_s: float,
    ) -> RankSnapshot:
        reported = [row for row in rows if _gpu_reported(row)]
        # The newest row is not always a reading: the last samples of a run
        # land during teardown, after torch has released the device, so
        # anchoring the GPU facts on it blanks them exactly when someone
        # inspects a finished run.
        newest_gpu = reported[-1] if reported else None

        recv = _opt_float(newest_row["recv_ts_ns"])
        age = policy.age_of(
            recv / 1e9 if recv is not None else None, now_s=now_s
        )

        return RankSnapshot(
            global_rank=rank_id,
            node_rank=_optional_int(newest_row["node_rank"]),
            # Same teardown reasoning as the byte fields below: a
            # rank's last sample lands after torch released the
            # device, so its device index is NULL there.
            gpu_index=_optional_int(
                (newest_gpu or newest_row)["gpu_device_index"]
            ),
            cpu_capacity_percent=_median(
                [
                    value
                    for value in (_cpu_capacity_of(r) for r in rows)
                    if value is not None
                ]
            ),
            ram_used_bytes=_opt_float(newest_row["ram_used_bytes"]),
            ram_used_p50_bytes=_median(
                [
                    value
                    for value in (
                        _opt_float(r["ram_used_bytes"]) for r in rows
                    )
                    if value is not None
                ]
            ),
            ram_total_bytes=_opt_float(newest_row["ram_total_bytes"]),
            gpu_allocated_p50_bytes=_median(
                [
                    value
                    for value in (
                        _opt_float(r["gpu_mem_used_bytes"]) for r in reported
                    )
                    if value is not None
                ]
            ),
            gpu_reserved_bytes=(
                _opt_float(newest_gpu["gpu_mem_reserved_bytes"])
                if newest_gpu is not None
                else None
            ),
            gpu_reserved_p50_bytes=_median(
                [
                    value
                    for value in (
                        _opt_float(r["gpu_mem_reserved_bytes"])
                        for r in reported
                    )
                    if value is not None
                ]
            ),
            gpu_total_bytes=(
                _opt_float(newest_gpu["gpu_mem_total_bytes"])
                if newest_gpu is not None
                else None
            ),
            age_s=age,
            freshness=policy.state_of(age),
        )

    # --- describing ------------------------------------------------------
    def _build_payload(
        self, ranks: Tuple[RankSnapshot, ...] = ()
    ) -> ProcessDashboardPayload:
        live = tuple(r for r in ranks if r.freshness == "fresh")
        stale = tuple(r for r in ranks if r.freshness == "stale")
        # Aggregates describe the ranks that are still reporting; a rank
        # that stopped is kept in `ranks` with its age so the card can say
        # it stopped, but it does not drag a headline with it. A rank whose
        # age is unknown stays in the aggregate: a missing timestamp is not
        # evidence that it died, and dropping it would discard a real
        # reading on the strength of a clock problem.
        reporting = tuple(r for r in ranks if r.freshness != "stale")
        aggregate_over = reporting or ranks
        coverage = RankCoverage(
            total=len(ranks),
            live=len(live),
            stale=len(stale),
            unknown=len(ranks) - len(live) - len(stale),
        )

        imbalance = self._reserved_imbalance(aggregate_over)
        rows_open = self._rows_trigger(aggregate_over, imbalance)

        history = tuple(self._dashboard_rollup)
        if not history:
            return ProcessDashboardPayload(
                ranks=ranks,
                coverage=coverage,
                cpu_capacity=self._cpu_rollup(aggregate_over),
                rss_worst=self._rss_rollup(aggregate_over),
                gpu_reserved=self._cuda_rollup(aggregate_over),
                gpu_allocated=self._alloc_rollup(aggregate_over),
                reserved_imbalance_percent=imbalance,
                rows_open=rows_open,
            )

        window = history[-DASHBOARD_WINDOW:]
        latest = window[-1]

        cpu_values = [e.cpu_percent_max for e in window]
        ram_values = [e.ram_used_bytes_max for e in window]
        gpu_values = [e.gpu.used_bytes for e in window if e.gpu is not None]

        return ProcessDashboardPayload(
            history=window,
            window_len=len(window),
            cpu=MetricRollup(
                now=cpu_values[-1],
                p50=percentile(cpu_values, 50),
                p95=percentile(cpu_values, 95),
            ),
            ram=MetricRollup(
                now=ram_values[-1],
                p95=percentile(ram_values, 95),
                total=latest.ram_total_bytes,
            ),
            gpu=(
                MetricRollup(
                    now=gpu_values[-1] if gpu_values else 0.0,
                    p95=percentile(gpu_values, 95) if gpu_values else 0.0,
                    total=(
                        latest.gpu.total_bytes
                        if latest.gpu is not None
                        else None
                    ),
                )
                if latest.gpu is not None
                else None
            ),
            gpu_used_imbalance_bytes=(
                latest.gpu.used_imbalance_bytes
                if latest.gpu is not None
                else None
            ),
            chart=_build_chart(window),
            ranks=ranks,
            coverage=coverage,
            reserved_imbalance_percent=imbalance,
            rows_open=rows_open,
            cpu_capacity=self._cpu_rollup(aggregate_over),
            rss_worst=self._rss_rollup(aggregate_over),
            gpu_reserved=self._cuda_rollup(aggregate_over),
            gpu_allocated=self._alloc_rollup(aggregate_over),
        )

    # --- rollups ---------------------------------------------------------
    def _cpu_rollup(
        self, ranks: Sequence[RankSnapshot]
    ) -> Optional[MetricRollup]:
        """CPU led by the WORST rank, named.

        A bottleneck finder has to answer "which rank", so the headline is
        the worst one rather than the cohort median, and the median rides
        alongside as the context that says whether it is an outlier.
        """
        pairs = [
            (r.global_rank, r.cpu_capacity_percent)
            for r in ranks
            if r.cpu_capacity_percent is not None
        ]
        if not pairs:
            return None
        worst_rank, worst = max(pairs, key=lambda item: item[1])
        return MetricRollup(
            now=float(worst),
            p50=_median([value for _r, value in pairs]),
            worst_rank=int(worst_rank),
        )

    def _rss_rollup(
        self, ranks: Sequence[RankSnapshot]
    ) -> Optional[MetricRollup]:
        """RSS of the worst rank, chosen on its median, shown at its newest.

        Which rank is worst is a judgement about its typical state, so it
        is made on the window median; the number displayed is still the
        newest value that rank actually sent.
        """
        pairs = [
            (r, r.ram_used_p50_bytes)
            for r in ranks
            if r.ram_used_p50_bytes is not None
        ]
        if not pairs:
            return None
        worst, _p50 = max(pairs, key=lambda item: item[1])
        shown = worst.ram_used_bytes
        if shown is None:
            shown = worst.ram_used_p50_bytes or 0.0
        return MetricRollup(
            now=float(shown),
            p50=_median([value for _r, value in pairs]),
            total=worst.ram_total_bytes,
            worst_rank=worst.global_rank,
        )

    def _cuda_rollup(
        self, ranks: Sequence[RankSnapshot]
    ) -> Optional[MetricRollup]:
        """Reserved memory on the rank with the least headroom.

        Not the largest user: the process at risk is the one closest to
        filling its card.
        """
        candidates = [
            r
            for r in ranks
            if r.gpu_reserved_bytes is not None
            and r.gpu_total_bytes is not None
            and r.gpu_total_bytes > 0
        ]
        if not candidates:
            return None
        worst = min(
            candidates,
            key=lambda r: (r.gpu_total_bytes or 0.0)
            - (r.gpu_reserved_bytes or 0.0),
        )
        return MetricRollup(
            now=float(worst.gpu_reserved_bytes or 0.0),
            total=worst.gpu_total_bytes,
            worst_rank=worst.global_rank,
        )

    def _alloc_rollup(
        self, ranks: Sequence[RankSnapshot]
    ) -> Optional[MetricRollup]:
        """Live tensors on the median rank.

        The median rather than the worst, because allocated bytes are the
        model's shape rather than a risk: on a healthy synchronous run
        every rank holds much the same, and the median says what that is
        while the reserved tile beside it names the rank at risk.

        Read from the ranks, not from the aggregated step history. The
        history's newest step carries no GPU snapshot once a run tears
        down, which left this tile reading "n/a" directly above rows that
        listed each rank's allocated bytes.
        """
        values = [
            r.gpu_allocated_p50_bytes
            for r in ranks
            if r.gpu_allocated_p50_bytes is not None
        ]
        if not values:
            return None
        return MetricRollup(now=float(_median(values) or 0.0))

    def _reserved_imbalance(
        self, ranks: Sequence[RankSnapshot]
    ) -> Optional[float]:
        """Spread of RESERVED memory across ranks, as a percentage.

        Reserved rather than allocated: the allocator's live bytes are a
        sawtooth this cadence undersamples, so their across-rank spread is
        phase noise on a healthy run. Reserved is what the process holds.
        Computed on window medians for the same reason.
        """
        values = [
            r.gpu_reserved_p50_bytes
            for r in ranks
            if r.gpu_reserved_p50_bytes is not None
            and r.gpu_reserved_p50_bytes > 0
        ]
        if len(values) < 2:
            return None
        low, high = min(values), max(values)
        if high <= 0:
            return None
        return float((high - low) / high * 100.0)

    def _rows_trigger(
        self,
        ranks: Sequence[RankSnapshot],
        imbalance: Optional[float],
    ) -> bool:
        """Whether the per-rank rows have earned opening themselves.

        Armed only once every reporting rank has HELD an allocation across
        its window. Ranks reach their first CUDA allocation seconds to
        minutes apart, and an unarmed trigger reads that ordinary ramp as
        total imbalance on every run's first ticks, so the rows would fly
        open on a healthy start.

        This is a judgement about the telemetry, which is why it is decided
        here and shipped as a fact. A view that compared the imbalance to a
        number of its own would be making a severity call in the layer that
        is only allowed to draw one.
        """
        armed = bool(ranks) and all(
            rank.gpu_reserved_p50_bytes is not None
            and rank.gpu_reserved_p50_bytes > 0
            for rank in ranks
        )
        if not armed or imbalance is None:
            return False
        return float(imbalance) >= IMBALANCE_OPEN_PCT

    # --- whole-run series ------------------------------------------------
    def _rank_charts(
        self,
        conn: Any,
        window_span_s: float,
        by_rank: Dict[int, List[Any]],
    ) -> Tuple[RankChart, RankChart]:
        """Per-rank CPU and RSS, over the window or over the whole run."""
        return (
            self._one_chart(conn, "cpu", window_span_s, by_rank),
            self._one_chart(conn, "rss", window_span_s, by_rank),
        )

    def _recent_chart(
        self,
        metric: str,
        by_rank: Dict[int, List[Any]],
        span_s: Optional[float],
    ) -> RankChart:
        """The live view: each rank's own samples, as they were taken.

        Built from the rows the snapshot read already returned rather than
        from a second query, and rebuilt every tick, because the whole
        point of the recent view is that it moves.
        """
        value_of = (
            _cpu_capacity_of
            if metric == "cpu"
            else (lambda row: _opt_float(row["ram_used_bytes"]))
        )
        traces = []
        for rank_id in sorted(by_rank):
            stamps, values = [], []
            for row in by_rank[rank_id]:
                stamp = _opt_float(row["sample_ts_s"])
                value = value_of(row)
                if stamp is None or value is None:
                    continue
                stamps.append(stamp)
                values.append(value)
            if stamps:
                traces.append(
                    RankTrace(
                        global_rank=rank_id,
                        timestamps=tuple(stamps),
                        values=tuple(values),
                    )
                )
        return RankChart(mode="recent", span_s=span_s, traces=tuple(traces))

    def _one_chart(
        self,
        conn: Any,
        metric: str,
        window_span_s: float,
        by_rank: Dict[int, List[Any]],
    ) -> RankChart:
        """One chart, in whichever mode the run has earned.

        Only the retained branch is cached. A rolling mean over minutes
        cannot visibly change between two ticks and recomputing it every
        tick was the largest cost measured in this block, but the recent
        view is the live one and a cached copy of it would be a chart that
        stops moving.
        """
        stats = (
            self._db.cpu_capacity_run_stats(conn)
            if metric == "cpu"
            else self._db.rss_run_stats(conn)
        )
        if stats is None:
            return self._recent_chart(metric, by_rank, None)

        mode = self._run_policy.mode_for(stats.span_s, window_span_s)
        if mode != "retained":
            return self._recent_chart(metric, by_rank, stats.span_s)

        now = time.time()
        cached = self._run_cache.get(metric)
        if (
            cached is not None
            and (now - self._run_cache_at.get(metric, 0.0)) < RUN_REFRESH_S
        ):
            return cached

        plan = plan_run_series(
            span_s=stats.span_s,
            sample_count=stats.samples_per_rank,
            policy=self._run_policy,
        )
        if plan is None:
            return self._recent_chart(metric, by_rank, stats.span_s)

        rows = (
            self._db.fetch_cpu_capacity_run(conn, plan)
            if metric == "cpu"
            else self._db.fetch_rss_run(conn, plan)
        )
        rolled: Dict[int, Tuple[List[float], List[float], List[float]]] = {}
        for rank_id, ts, roll_avg, roll_max in rows:
            stamps, values, peaks = rolled.setdefault(rank_id, ([], [], []))
            stamps.append(ts)
            values.append(roll_avg)
            peaks.append(roll_max)

        chart = RankChart(
            mode="retained",
            window_s=plan.window_s,
            span_s=stats.span_s,
            traces=tuple(
                RankTrace(
                    global_rank=rank_id,
                    timestamps=tuple(rolled[rank_id][0]),
                    values=tuple(rolled[rank_id][1]),
                    peaks=tuple(rolled[rank_id][2]),
                )
                for rank_id in sorted(rolled)
            ),
        )
        self._run_cache[metric] = chart
        self._run_cache_at[metric] = now
        return chart

    # --- degraded reads --------------------------------------------------
    def _with_charts(
        self,
        conn: Any,
        payload: ProcessDashboardPayload,
        by_rank: Dict[int, List[Any]],
    ) -> ProcessDashboardPayload:
        """Attach the per-rank charts, in whichever mode the run warrants."""
        window_span = _window_span(payload.history)
        cpu_chart, rss_chart = self._rank_charts(conn, window_span, by_rank)
        return replace(
            payload, cpu_capacity_chart=cpu_chart, rss_chart=rss_chart
        )

    def _return_stale(self) -> ProcessDashboardPayload:
        """Reuse the last good payload while a read keeps failing.

        This is about the READ, not the run: it says the database could
        not be queried just now, never that the ranks are still healthy.
        """
        now = time.time()
        if self._last_ok is not None and self._cache_ttl.may_reuse(
            now - self._last_ok_ts
        ):
            return self._last_ok
        return ProcessDashboardPayload()


def _optional_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _entry_from_row(row: Any) -> ProcessHistoryEntry:
    """One aggregated row, typed.

    The GPU block is built only when the query found a candidate rank, so
    a CPU-only step carries ``None`` rather than a set of absent keys.
    """
    gpu: Optional[GpuSnapshot] = None
    if row["gpu_used"] is not None:
        gpu = GpuSnapshot(
            used_bytes=float(row["gpu_used"] or 0.0),
            total_bytes=float(row["gpu_total"] or 0.0),
            headroom_bytes=float(row["gpu_headroom"] or 0.0),
            rank=(
                int(row["gpu_rank"]) if row["gpu_rank"] is not None else None
            ),
            used_imbalance_bytes=(
                float(row["gpu_used_imbalance"])
                if row["gpu_used_imbalance"] is not None
                else 0.0
            ),
        )

    return ProcessHistoryEntry(
        seq=int(row["seq"]),
        ts=(
            float(row["sample_ts_s"])
            if row["sample_ts_s"] is not None
            else None
        ),
        cpu_percent_max=float(row["cpu_max"] or 0.0),
        ram_used_bytes_max=float(row["ram_used_max"] or 0.0),
        ram_total_bytes=float(row["ram_total"] or 0.0),
        gpu=gpu,
    )


def _window_span(window: Sequence[ProcessHistoryEntry]) -> float:
    """Wall-clock seconds the recent window covers."""
    stamps = [e.ts for e in window if e.ts is not None]
    if len(stamps) < 2:
        return 0.0
    return max(0.0, max(stamps) - min(stamps))


def _build_chart(window: Sequence[ProcessHistoryEntry]) -> ChartSeries:
    """Both traces as a share of their own denominator.

    A share of capacity is what the metric means, so it is computed here.
    The axis, the colours and the tick labels are the card's business.
    """
    timestamps: Tuple[Optional[float], ...] = tuple(e.ts for e in window)

    ram_total = max(float(window[-1].ram_total_bytes or 0.0), 1.0)
    ram = ChartTrace(
        label="RAM",
        timestamps=timestamps,
        values=tuple(
            (e.ram_used_bytes_max / ram_total) * 100.0 for e in window
        ),
    )

    gpu_trace: Optional[ChartTrace] = None
    latest_gpu = window[-1].gpu
    if latest_gpu is not None and latest_gpu.total_bytes:
        gpu_total = max(float(latest_gpu.total_bytes), 1.0)
        stamps: List[Optional[float]] = []
        values: List[float] = []
        for entry in window:
            if entry.gpu is None:
                continue
            stamps.append(entry.ts)
            values.append((entry.gpu.used_bytes / gpu_total) * 100.0)
        gpu_trace = ChartTrace(
            label="GPU mem",
            timestamps=tuple(stamps),
            values=tuple(values),
        )

    return ChartSeries(ram_percent=ram, gpu_percent=gpu_trace)


__all__ = [
    "DASHBOARD_WINDOW",
    "ProcessDashboardComputer",
    "percentile",
]
