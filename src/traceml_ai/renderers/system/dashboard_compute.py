# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Dashboard compute for system telemetry."""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from traceml_ai.renderers.shared.freshness import CachedPayloadTTL
from traceml_ai.renderers.shared.run_series import (
    DEFAULT_RUN_SERIES_POLICY,
    RunSeriesPolicy,
    SeriesMode,
    plan_run_series,
)

from .common import gpu_reported as _gpu_reported
from .common import reading as _opt_float
from .dashboard_models import (
    SystemDashboardPayload,
    SystemRollups,
    SystemSeries,
)
from .repository import RunStats, SystemRepository


def _median(values: List[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return float(np.percentile(present, 50)) if present else None


def _positive(value: Any) -> Optional[float]:
    """A capacity-like value: None unless it is a real positive number."""
    v = _opt_float(value)
    return v if v is not None and v > 0.0 else None


def _empty_dashboard_series() -> Dict[str, Any]:
    """Return the complete dashboard-series schema with no observations."""
    return {
        "x_time": [],
        "cpu": [],
        "gpu_avg": [],
        "gpu_power": [],
        "cpu_run": {
            "t": [],
            "avg": [],
            "max": [],
            "span_s": 0.0,
            "window_s": 0.0,
        },
        "gpu_power_run": [],
        "cpu_run_mode": "recent",
        "power_run_mode": "recent",
    }


def _typed(
    *,
    window_len: int,
    gpu_available: bool,
    rollups: Dict[str, Any],
    series: Dict[str, Any],
) -> SystemDashboardPayload:
    """Wrap the compute layer's mappings in the payload's own types.

    The mappings are built as dicts here because that is how the readers
    and the numpy work naturally produce them; the types are the contract
    the card reads, and the models own the adaptation so the shape is
    described in one place.
    """
    return SystemDashboardPayload(
        window_len=window_len,
        gpu_available=gpu_available,
        rollups=SystemRollups.from_dict(rollups),
        series=SystemSeries.from_dict(series),
    )


def _nan_pct(values: Any, q: float) -> Optional[float]:
    """A percentile over the ticks that were measured.

    A window in which nothing was measured has no percentile, so this
    abstains rather than naming a number. The card already renders an
    absent field as "n/a"; a 0.0 here would render as a real measurement
    of an idle machine.
    """
    if values.size == 0 or bool(np.all(np.isnan(values))):
        return None
    return float(np.nanpercentile(values, q))


def _last_measured_index(values: Any) -> Optional[int]:
    """Index of the newest measured tick, or None if there is none."""
    listed = values.tolist()
    for i in range(len(listed) - 1, -1, -1):
        if listed[i] == listed[i]:  # not NaN
            return i
    return None


def _last_measured(values: Any) -> Optional[float]:
    """The newest measured value, skipping absent ticks.

    None when no tick in the window was measured: there is no newest
    value to report, and 0.0 is a level a machine can genuinely be at.
    """
    for value in reversed(values.tolist()):
        if value == value:  # not NaN
            return float(value)
    return None


def _paired_total(levels: Any, totals: Any) -> Optional[float]:
    """The capacity belonging to the tick the level was read from.

    None when no level was measured: with no tick to read from there is
    no capacity that belongs to the reported value, and pairing one
    anyway is how a level and its denominator come from two moments.
    """
    i = _last_measured_index(levels)
    if i is None:
        return None
    total = totals.tolist()[i]
    return float(total) if total == total else None


def _gap_list(values: Any) -> List[Optional[float]]:
    """A series where an absent tick is a gap rather than a zero."""
    return [None if v != v else float(v) for v in values.tolist()]


def _empty_cpu_run() -> Dict[str, Any]:
    """The whole-run CPU shape with nothing in it."""
    return {"t": [], "avg": [], "max": [], "span_s": 0.0, "window_s": 0.0}


def _gpu_medians(gpus: List[Dict[str, Any]]) -> List[tuple]:
    """Each GPU's representative utilisation, as (index, value).

    The window median where there is one, else the newest reading. One
    definition, used by everything that asks about the spread across
    devices, so two surfaces on the same card cannot answer differently.
    """
    out = []
    for g in gpus:
        value = g.get("util_p50")
        if value is None:
            value = g.get("util_now")
        if value is not None:
            out.append((int(g.get("gpu_idx", 0)), float(value)))
    return out


def _odd_gpus(gpus: List[Dict[str, Any]]) -> List[int]:
    """The GPUs on the smaller side of the utilisation split.

    Values split at the midpoint between the lowest and highest
    representative utilisation; the smaller group is the one worth
    pointing at, and a tie goes to the higher group.

    Moved here from the card unchanged. It derives a threshold and then
    selects entities by it, which is the shape of a diagnosis rule and not
    of a drawing decision, so it belongs on this side of the boundary. Its
    behaviour is deliberately preserved, including the consequence that on
    a two-GPU host the groups always tie and the busier card is the one
    marked.
    """
    pairs = _gpu_medians(gpus)
    if len(pairs) < 2:
        return []
    low_value = min(v for _i, v in pairs)
    high_value = max(v for _i, v in pairs)
    if high_value <= low_value:
        return []
    mid = (high_value + low_value) / 2.0
    high = [i for i, v in pairs if v > mid]
    low = [i for i, v in pairs if v <= mid]
    return sorted(high if len(high) <= len(low) else low)


# Utilisation spread, in percentage points, at which the per-GPU rows
# have earned opening themselves. A threshold against a measurement is a
# severity judgement, so it lives here and not in the card.
SPREAD_EXPAND_PTS = 20.0


def _util_range(gpus: List[Dict[str, Any]]) -> Optional[tuple]:
    """The lowest and highest representative utilisation, or None.

    The card computed this itself for its disclosure header, which made it
    a SECOND definition of "the spread across GPUs" living beside
    ``gpu_delta``, and the two could disagree: one is the range of window
    medians, the other the 95th percentile of per-tick max minus min.
    Both are legitimate; having two of them unnamed on one card was not.
    """
    pairs = _gpu_medians(gpus)
    if not pairs:
        return None
    values = [v for _i, v in pairs]
    return (min(values), max(values))


class SystemDashboardComputer:
    """Compute dashboard rollups and short time-series."""

    def __init__(
        self,
        db_path: str,
        node_rank: Optional[int] = None,
        stale_ttl_s: Optional[float] = 30.0,
        run_series_policy: RunSeriesPolicy = DEFAULT_RUN_SERIES_POLICY,
    ) -> None:
        self._db = SystemRepository(db_path=db_path, node_rank=node_rank)
        self._last_ok: Optional[SystemDashboardPayload] = None
        self._last_ok_ts: float = 0.0
        # Whether a cached payload may still answer says the DATABASE could
        # not be read; it never says the host is healthy. The shared type
        # keeps those two ideas from sharing one number, as they did here.
        self._cache_ttl = CachedPayloadTTL(ttl_s=stale_ttl_s)
        self._run_policy = run_series_policy

    def compute(self, window_n: int = 100) -> SystemDashboardPayload:
        """
        Compute dashboard rollups plus short series over the latest window.

        Returns cached values on transient failure or empty window if they are
        still within the configured stale TTL. Otherwise returns the default
        empty payload.
        """
        try:
            with self._db.connect() as conn:
                out = self._compute_impl(conn, window_n=max(1, int(window_n)))
        except Exception as e:
            return self._return_stale(f"STALE (exception: {type(e).__name__})")

        if out.window_len == 0 and self._last_ok is not None:
            return self._return_stale("STALE (empty window)")

        self._last_ok = out
        self._last_ok_ts = time.time()
        return out

    def _compute_impl(self, conn, window_n: int) -> SystemDashboardPayload:
        """
        Compute the dashboard payload from recent SQLite rows.

        Notes
        -----
        - `sample_ts_s` is used as the canonical sample timestamp.
        - `x_time` is emitted as ISO-8601 UTC strings so the UI can plot a
          real time axis instead of sample indices or relative negative values.
        """
        samples = self._db.fetch_recent_system_samples(conn, limit=window_n)
        if not samples:
            return self._empty_payload()

        # System telemetry is per machine. When the window holds more than
        # one host, show the leader node alone and say so, rather than a
        # series that zig-zags between two machines and GPU rows in which
        # both nodes' gpu0 collapse into one.
        system_node = self._pick_node(samples)
        if system_node["nodes_in_window"] > 1:
            samples = self._db.fetch_recent_system_samples(
                conn, limit=window_n, hostname=system_node["hostname"]
            )
            if not samples:
                return self._empty_payload()

        last = samples[-1]
        gpu_available = bool(last["gpu_available"] or False)

        ts_hist = np.array(
            [float(r["sample_ts_s"] or 0.0) for r in samples],
            dtype=np.float64,
        )
        # NaN for a missing reading, for the same reason the GPU arrays
        # below use it: `cpu_percent` is nullable, and a coerced zero is
        # both drawn as a real 0% and folded into the window percentile.
        cpu_hist = np.array(
            [_opt_float(r["cpu_percent"]) for r in samples],
            dtype=np.float64,
        )
        ram_used_hist = np.array(
            [_opt_float(r["ram_used_bytes"]) for r in samples],
            dtype=np.float64,
        )
        ram_total = _positive(last["ram_total_bytes"])

        gpu_rows = self._db.fetch_gpu_rows_for_samples(
            conn,
            sample_keys=[
                (sample["global_rank"], sample["seq"])
                for sample in samples
                if sample["seq"] is not None
            ],
        )
        gpu_rows_by_key = self._db.group_gpu_rows_by_global_rank_seq(gpu_rows)

        n = len(samples)
        # NaN, not zero: a tick in which nothing reported is an absence,
        # and a zero here is drawn as a real 0% and folded into the window
        # percentiles. Same rule the per-GPU histories below follow with
        # None, and the same rule the per-device aggregates follow.
        gpu_avg = np.full(n, np.nan, dtype=np.float64)
        gpu_delta = np.full(n, np.nan, dtype=np.float64)
        gpu_mem_worst = np.full(n, np.nan, dtype=np.float64)
        # The capacity of the device in gpu_mem_worst, per tick.
        # Kept per tick so the level and the capacity it is
        # measured against can be read from the SAME moment; a
        # level that skips back past a blind tick while its
        # capacity does not describes two different instants.
        gpu_mem_total_hist = np.full(n, np.nan, dtype=np.float64)
        gpu_mem_headroom_min = np.full(n, np.nan, dtype=np.float64)
        temp_max = np.full(n, np.nan, dtype=np.float64)
        # Util readings in the newest tick. This distinguishes a measured
        # zero from no current reading; it does not describe coverage of
        # the window median shown by the tile. A device can report and
        # still carry a NULL util column, so count readings, not devices.
        util_gpu_count = 0

        # Per-GPU history keyed by gpu_idx, one slot per tick. A tick in
        # which a GPU has no row stays None: a gap in its trace, not a 0.
        # Power feeds the power chart and the per-GPU rows; per-GPU util
        # gives each row its own window median.
        power_hist: Dict[int, List[Optional[float]]] = {}
        util_hist: Dict[int, List[Optional[float]]] = {}
        power_max_hist: List[Optional[float]] = [None] * n
        latest_rows: Dict[int, Any] = {}
        # The board limit is a constant per GPU; remember the largest one
        # reported anywhere in the window so one unreported tick does not
        # make the limit line flicker.
        power_limit: Optional[float] = None

        for i, sample in enumerate(samples):
            key = (sample["global_rank"], sample["seq"])
            rows = gpu_rows_by_key.get(key, [])

            if rows:
                # The aggregates describe the devices that REPORTED, for
                # the same reason the per-GPU histories below store None
                # rather than 0: an unread device is an absence, not a
                # measurement of zero. Substituting 0.0 pulled the mean
                # down and manufactured an across-GPU spread, which is
                # what opens the per-GPU rows.
                live = [g for g in rows if _gpu_reported(g)]
                utils = [
                    v
                    for v in (_opt_float(g["util"]) for g in live)
                    if v is not None
                ]
                mem_pairs = [
                    (used, _opt_float(g["mem_total_bytes"]))
                    for g, used in (
                        (g, _opt_float(g["mem_used_bytes"])) for g in live
                    )
                    if used is not None
                ]
                temps = [
                    v
                    for v in (_opt_float(g["temperature_c"]) for g in live)
                    if v is not None
                ]

                if i == n - 1:
                    util_gpu_count = len(utils)
                if utils:
                    gpu_avg[i] = sum(utils) / float(len(utils))
                    gpu_delta[i] = max(utils) - min(utils)
                if mem_pairs:
                    worst_used, worst_total = max(
                        mem_pairs, key=lambda pair: pair[0]
                    )
                    gpu_mem_worst[i] = worst_used
                    if worst_total is not None:
                        gpu_mem_total_hist[i] = worst_total
                if temps:
                    temp_max[i] = max(temps)
                if i == n - 1:
                    # Every row, reported or not: the per-GPU rows keep a
                    # slot for a silent device and mark it themselves.
                    latest_rows = {int(g["gpu_idx"]): g for g in rows}

                powers = []
                for g in rows:
                    idx = int(g["gpu_idx"])
                    reported = _gpu_reported(g)
                    power = (
                        _opt_float(g["power_usage_w"]) if reported else None
                    )
                    util = _opt_float(g["util"]) if reported else None
                    power_hist.setdefault(idx, [None] * n)[i] = power
                    util_hist.setdefault(idx, [None] * n)[i] = util
                    if power is not None:
                        powers.append(power)
                    limit = _positive(g["power_limit_w"])
                    if limit is not None and (
                        power_limit is None or limit > power_limit
                    ):
                        power_limit = limit
                power_max_hist[i] = max(powers) if powers else None

                # Headroom needs both halves from the same device, and a
                # device that did not report has neither.
                headrooms = [
                    max(total - used, 0.0)
                    for used, total in mem_pairs
                    if total is not None and total > 0.0
                ]
                if headrooms:
                    gpu_mem_headroom_min[i] = min(headrooms)
            else:
                # No rows for this tick at all: leave every slot absent,
                # which is what the pre-fill already says.
                pass

        ram_now = _last_measured(ram_used_hist)
        cpu_p50 = _nan_pct(cpu_hist, 50)
        cpu_p95 = _nan_pct(cpu_hist, 95)

        ram_p95 = _nan_pct(ram_used_hist, 95)

        # nanpercentile, not percentile: a blind tick must not be counted
        # as a measured zero when the window is summarised. All-NaN means
        # nothing was measured at all, which is 0.0 as it was before.
        gpu_p50 = _nan_pct(gpu_avg, 50)
        gpu_p95 = _nan_pct(gpu_avg, 95)
        delta_p95 = _nan_pct(gpu_delta, 95)
        mem_p95 = _nan_pct(gpu_mem_worst, 95)
        temp_p95 = _nan_pct(temp_max, 95)

        temp_now = _last_measured(temp_max)
        # A verdict needs a reading. With no temperature there is nothing
        # to be OK about, and "OK" is the answer a reader trusts most.
        temp_status = (
            None
            if temp_now is None
            else (
                "Hot" if temp_now >= 85 else "Warm" if temp_now >= 80 else "OK"
            )
        )

        host = (
            system_node["hostname"]
            if system_node["nodes_in_window"] > 1
            else None
        )
        window_span = max(float(ts_hist[-1] - ts_hist[0]), 0.0)
        cpu_stats = self._db.cpu_run_stats(conn, hostname=host)
        cpu_mode = self._mode_for(cpu_stats, window_span)
        cpu_run = (
            self._cpu_run_series(conn, cpu_stats, host)
            if cpu_mode == "retained"
            else _empty_cpu_run()
        )
        # A retained read may be unavailable (for example on an older SQLite
        # engine). The payload states the view it can actually draw, while the
        # policy above still decides whether the retained read is attempted.
        if cpu_mode == "retained" and not cpu_run.get("t"):
            cpu_mode = "recent"
        power_stats = (
            self._db.gpu_power_run_stats(conn, hostname=host)
            if gpu_available
            else None
        )
        power_mode = self._mode_for(power_stats, window_span)
        power_run = (
            self._power_run_series(conn, power_stats, host)
            if power_mode == "retained"
            else []
        )
        if power_mode == "retained" and not power_run:
            power_mode = "recent"
        run_power_floor = min(
            (v for e in power_run for v in (e.get("min") or [])),
            default=None,
        )
        window_power_floor = min(
            (
                v
                for values in power_hist.values()
                for v in values
                if v is not None
            ),
            default=None,
        )

        rollups = {
            "cpu": {
                "now": _last_measured(cpu_hist),
                "p50": cpu_p50,
                "p95": cpu_p95,
            },
            "ram": {
                "now": ram_now,
                "p95": ram_p95,
                "total": ram_total,
                # Free memory is the difference between two readings, so
                # it exists only when both of them do.
                "headroom": (
                    max(ram_total - ram_now, 0.0)
                    if ram_total is not None and ram_now is not None
                    else None
                ),
            },
            "gpu_util": {
                "now": _last_measured(gpu_avg),
                "p50": gpu_p50,
                "p95": gpu_p95,
            },
            "gpu_delta": {
                "now": _last_measured(gpu_delta),
                "p95": delta_p95,
            },
            "gpu_mem": {
                "now": _last_measured(gpu_mem_worst),
                "p95": mem_p95,
                "headroom": _last_measured(gpu_mem_headroom_min),
                # Capacity of the GPU shown in "now" (the max-used one),
                # so the tile can read "used / total".
                "total": _paired_total(gpu_mem_worst, gpu_mem_total_hist),
            },
            "temp": {
                "now": temp_now,
                "p95": temp_p95,
                "status": temp_status,
            },
            # Power is a per-GPU quantity; any aggregate cell is the max
            # GPU, never a sum. None when no GPU reported power.
            "gpu_power": {
                "now": power_max_hist[-1],
                "p50": _median(power_max_hist),
                "limit": power_limit,
                # Lowest reported power across the whole run, or across the
                # recent window when whole-run aggregation is skipped.
                "floor": (
                    run_power_floor
                    if run_power_floor is not None
                    else window_power_floor
                ),
            },
            "gpus": (
                self._gpu_rows(latest_rows, util_hist, power_hist)
                if gpu_available
                else []
            ),
        }

        rollups["util_gpu_count"] = util_gpu_count
        gpu_rows = rollups["gpus"]
        rollups["odd_gpus"] = _odd_gpus(gpu_rows)
        rollups["util_range"] = _util_range(gpu_rows)
        spread = (rollups.get("gpu_delta") or {}).get("p95")
        rollups["rows_over"] = (
            spread is not None and float(spread) > SPREAD_EXPAND_PTS
        )

        rollups["ctx"] = {
            "world_size": int(last["world_size"] or 0),
            "gpu_count": int(last["gpu_count"] or 0),
            "hostname": str(last["hostname"] or ""),
            # Which node this payload's series and rows describe.
            "system_node": system_node,
        }

        x_time = [self._format_time_iso(ts) for ts in ts_hist.tolist()]

        return _typed(
            window_len=len(samples),
            gpu_available=gpu_available,
            rollups=rollups,
            series={
                "x_time": x_time,
                "cpu": _gap_list(cpu_hist),
                "gpu_avg": (_gap_list(gpu_avg) if gpu_available else []),
                "gpu_power": (
                    [
                        {"gpu_idx": idx, "values": power_hist[idx]}
                        for idx in sorted(power_hist)
                    ]
                    if gpu_available
                    else []
                ),
                # Whole-run views (decimated in SQL): the window says what
                # the host is doing now, these say what it has done.
                "cpu_run": cpu_run,
                "gpu_power_run": power_run if gpu_available else [],
                # Which view each chart is in, decided once here rather
                # than twice in the card with two different rules, and
                # named in the shared vocabulary both blocks now use.
                "cpu_run_mode": cpu_mode,
                "power_run_mode": power_mode if gpu_available else "recent",
            },
        )

    def _mode_for(
        self, stats: Optional[RunStats], window_span_s: float
    ) -> SeriesMode:
        """Whether a chart describes the window or the whole run.

        The 1.2x factor that keeps a chart near the boundary from flipping
        every tick used to be `_RUN_VIEW_FACTOR` here, a local copy of the
        shared policy's `retained_factor`. Same number, one owner now.
        """
        if stats is None:
            return "recent"
        return self._run_policy.mode_for(stats.span_s, window_span_s)

    def _cpu_run_series(
        self,
        conn: Any,
        stats: Optional[RunStats],
        host: Optional[str],
    ) -> Dict[str, Any]:
        """Whole-run host CPU, planned by the shared policy."""
        if stats is None:
            return _empty_cpu_run()
        plan = plan_run_series(
            span_s=stats.span_s,
            sample_count=stats.sample_count,
            policy=self._run_policy,
        )
        if plan is None:
            return _empty_cpu_run()
        rows = self._db.fetch_cpu_run(conn, plan, hostname=host)
        if not rows:
            return _empty_cpu_run()
        return {
            "t": [r[0] for r in rows],
            "avg": [r[1] for r in rows],
            "max": [r[2] for r in rows],
            "span_s": stats.span_s,
            "window_s": plan.window_s,
        }

    def _power_run_series(
        self,
        conn: Any,
        stats: Optional[RunStats],
        host: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Whole-run per-GPU power, bucketed at the rolling window's width.

        This path buckets rather than rolls, so only the window duration
        carries over from the shared policy; there is no stride or point
        budget to apply, and the bucket count is therefore still unbounded.
        See the follow-up issue.
        """
        if stats is None:
            return []
        width = self._run_policy.window_for(stats.span_s)
        rows = self._db.fetch_gpu_power_run(
            conn,
            width_s=width,
            first_ts=stats.first_ts,
            hostname=host,
        )
        by_gpu: Dict[int, Dict[str, Any]] = {}
        for gpu_idx, ts, avg, low, high in rows:
            entry = by_gpu.setdefault(
                gpu_idx,
                {
                    "gpu_idx": gpu_idx,
                    "t": [],
                    "avg": [],
                    "min": [],
                    "max": [],
                },
            )
            entry["t"].append(ts)
            entry["avg"].append(avg)
            entry["min"].append(low)
            entry["max"].append(high)
        out = [by_gpu[i] for i in sorted(by_gpu)]
        for entry in out:
            entry["span_s"] = stats.span_s
            entry["window_s"] = width
        return out

    @staticmethod
    def _pick_node(samples: List[Any]) -> Dict[str, Any]:
        """The node whose window this is: the lowest node_rank seen.

        Hosts are told apart by hostname (node_rank breaks the tie order);
        ``nodes_in_window`` > 1 means other machines were dropped from
        this payload, which the block must say. Rows without a hostname
        (pre-0.3.0 writers, malformed meta) are not a node of their own:
        they neither count nor select, the same rule the strip's node
        count applies.
        """
        nodes: Dict[str, Optional[int]] = {}
        for row in samples:
            host = row["hostname"]
            if host is None or str(host) == "":
                continue
            host = str(host)
            rank = row["node_rank"]
            rank = int(rank) if rank is not None else None
            if host not in nodes or (
                rank is not None
                and (nodes[host] is None or rank < nodes[host])
            ):
                nodes[host] = rank
        if not nodes:
            return {"hostname": None, "node_rank": None, "nodes_in_window": 1}
        host = min(
            nodes,
            key=lambda h: (
                nodes[h] if nodes[h] is not None else float("inf"),
                h,
            ),
        )
        return {
            "hostname": host,
            "node_rank": nodes[host],
            "nodes_in_window": len(nodes),
        }

    @staticmethod
    def _gpu_rows(
        latest_rows: Dict[int, Any],
        util_hist: Dict[int, List[Optional[float]]],
        power_hist: Dict[int, List[Optional[float]]],
    ) -> List[Dict[str, Any]]:
        """One entry per GPU seen in the window, newest values per GPU.

        A GPU absent from the newest tick keeps its slot with None values
        rather than vanishing, so a row count never silently drops.
        """
        out: List[Dict[str, Any]] = []
        for idx in sorted(set(util_hist) | set(power_hist)):
            g = latest_rows.get(idx)
            reported = g is not None and _gpu_reported(g)
            if not reported:
                g = None  # Normalize unavailable and legacy-zero rows.
            out.append(
                {
                    "gpu_idx": idx,
                    # Stated, not inferred. The card used to work this out
                    # from which fields were None, with a rule that
                    # disagreed with this one about a GPU reporting a
                    # power limit but no memory total.
                    "reported": reported,
                    "util_now": (
                        _opt_float(g["util"]) if g is not None else None
                    ),
                    "util_p50": _median(util_hist.get(idx, [])),
                    "mem_used": (
                        _opt_float(g["mem_used_bytes"])
                        if g is not None
                        else None
                    ),
                    "mem_total": (
                        _positive(g["mem_total_bytes"])
                        if g is not None
                        else None
                    ),
                    "temp": (
                        _opt_float(g["temperature_c"])
                        if g is not None
                        else None
                    ),
                    "power": (
                        _opt_float(g["power_usage_w"])
                        if g is not None
                        else None
                    ),
                    "power_limit": (
                        _positive(g["power_limit_w"])
                        if g is not None
                        else None
                    ),
                }
            )
        return out

    def _format_time_iso(self, ts_s: float) -> str:
        """
        Convert one UNIX timestamp in seconds to an ISO-8601 UTC string.

        Returns an empty string on invalid input so callers can safely degrade.
        """
        try:
            if ts_s <= 0.0:
                return ""
            return datetime.fromtimestamp(
                float(ts_s), tz=timezone.utc
            ).isoformat()
        except Exception:
            return ""

    def _return_stale(self, msg: str) -> SystemDashboardPayload:
        """The last good payload while it is still within TTL.

        Carries a human-readable status so the card can say the numbers
        are held over rather than drawing them as current.
        """
        now = time.time()
        if self._last_ok is not None:
            if self._cache_ttl.may_reuse(now - self._last_ok_ts):
                cached = self._last_ok
                return replace(
                    cached,
                    rollups=replace(cached.rollups, status=msg),
                )

        return SystemDashboardPayload(
            rollups=SystemRollups(status="No fresh system data"),
            series=SystemSeries.from_dict(_empty_dashboard_series()),
        )

    def _empty_payload(self) -> SystemDashboardPayload:
        """An empty payload carrying the full expected schema."""
        return _typed(
            window_len=0,
            gpu_available=False,
            rollups={},
            series=_empty_dashboard_series(),
        )
