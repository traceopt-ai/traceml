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

from .dashboard_models import (
    SystemDashboardPayload,
    SystemRollups,
    SystemSeries,
)
from .repository import SystemRepository

# Whole-run charts add value only after the run outgrows the recent window.
_RUN_VIEW_FACTOR = 1.2


def _opt_float(value: Any) -> Optional[float]:
    """Float or None: an unreported column stays unreported, never 0."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _median(values: List[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return float(np.percentile(present, 50)) if present else None


def _positive(value: Any) -> Optional[float]:
    """A capacity-like value: None unless it is a real positive number."""
    v = _opt_float(value)
    return v if v is not None and v > 0.0 else None


def _gpu_reported(row: Any) -> bool:
    """False for the sampler's exception fallback (an all-zero GPU row).

    A real GPU always has a memory capacity and a board power limit; the
    sampler writes zeros for every field when NVML fails on a device, and
    those zeros must not render as 0 W, 0 C or 0 GB.
    """
    return (
        _positive(row["mem_total_bytes"]) is not None
        or _positive(row["power_limit_w"]) is not None
    )


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
        "cpu_run_whole": False,
        "power_run_whole": False,
    }


# A whole-run line needs more than two points to say anything about a
# trend. One rule, applied to both series: the card used to ask ">2
# points" of the CPU history and merely "non-empty" of the power history,
# so its two charts could disagree about which view they were showing.
_MIN_WHOLE_RUN_POINTS = 2


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


def _has_whole_run(entries: Any) -> bool:
    """Whether a whole-run series has enough shape to draw."""
    if isinstance(entries, dict):
        return len(entries.get("t") or ()) > _MIN_WHOLE_RUN_POINTS
    longest = 0
    for entry in entries or ():
        longest = max(longest, len(entry.get("t") or ()))
    return longest > _MIN_WHOLE_RUN_POINTS


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
    ) -> None:
        self._db = SystemRepository(db_path=db_path, node_rank=node_rank)
        self._last_ok: Optional[Dict[str, Any]] = None
        self._last_ok_ts: float = 0.0
        self._stale_ttl_s: Optional[float] = (
            float(stale_ttl_s) if stale_ttl_s is not None else None
        )

    def compute(self, window_n: int = 100) -> Dict[str, Any]:
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

    def _compute_impl(self, conn, window_n: int) -> Dict[str, Any]:
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
        cpu_hist = np.array(
            [float(r["cpu_percent"] or 0.0) for r in samples],
            dtype=np.float64,
        )
        ram_used_hist = np.array(
            [float(r["ram_used_bytes"] or 0.0) for r in samples],
            dtype=np.float64,
        )
        ram_total = float(last["ram_total_bytes"] or 0.0)

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
        gpu_avg = np.zeros(n, dtype=np.float64)
        gpu_delta = np.zeros(n, dtype=np.float64)
        gpu_mem_worst = np.zeros(n, dtype=np.float64)
        gpu_mem_headroom_min = np.zeros(n, dtype=np.float64)
        temp_max = np.zeros(n, dtype=np.float64)
        gpu_mem_worst_total = 0.0

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
                utils = [float(g["util"] or 0.0) for g in rows]
                mem_useds = [float(g["mem_used_bytes"] or 0.0) for g in rows]
                mem_totals = [float(g["mem_total_bytes"] or 0.0) for g in rows]
                temps = [float(g["temperature_c"] or 0.0) for g in rows]

                gpu_avg[i] = sum(utils) / float(len(utils))
                gpu_delta[i] = max(utils) - min(utils)
                gpu_mem_worst[i] = max(mem_useds)
                temp_max[i] = max(temps)
                if i == n - 1:
                    worst = max(range(len(rows)), key=lambda j: mem_useds[j])
                    gpu_mem_worst_total = mem_totals[worst]
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

                headrooms = [
                    max(mt - mu, 0.0)
                    for mu, mt in zip(mem_useds, mem_totals)
                    if mt > 0.0
                ]
                gpu_mem_headroom_min[i] = min(headrooms) if headrooms else 0.0
            else:
                gpu_avg[i] = 0.0
                gpu_delta[i] = 0.0
                gpu_mem_worst[i] = 0.0
                gpu_mem_headroom_min[i] = 0.0
                temp_max[i] = 0.0

        cpu_p50 = float(np.percentile(cpu_hist, 50)) if cpu_hist.size else 0.0
        cpu_p95 = float(np.percentile(cpu_hist, 95)) if cpu_hist.size else 0.0

        ram_p95 = (
            float(np.percentile(ram_used_hist, 95))
            if ram_used_hist.size
            else 0.0
        )

        gpu_p50 = float(np.percentile(gpu_avg, 50)) if gpu_avg.size else 0.0
        gpu_p95 = float(np.percentile(gpu_avg, 95)) if gpu_avg.size else 0.0

        delta_p95 = (
            float(np.percentile(gpu_delta, 95)) if gpu_delta.size else 0.0
        )

        mem_p95 = (
            float(np.percentile(gpu_mem_worst, 95))
            if gpu_mem_worst.size
            else 0.0
        )
        temp_p95 = float(np.percentile(temp_max, 95)) if temp_max.size else 0.0

        temp_now = float(temp_max[-1]) if temp_max.size else 0.0
        temp_status = (
            "Hot" if temp_now >= 85 else "Warm" if temp_now >= 80 else "OK"
        )

        host = (
            system_node["hostname"]
            if system_node["nodes_in_window"] > 1
            else None
        )
        window_span = max(float(ts_hist[-1] - ts_hist[0]), 0.0)
        min_run_span = window_span * _RUN_VIEW_FACTOR
        cpu_run = self._db.fetch_cpu_run_history(
            conn,
            hostname=host,
            min_span_s=min_run_span,
        )
        power_run = (
            self._db.fetch_gpu_power_run_history(
                conn,
                hostname=host,
                min_span_s=min_run_span,
            )
            if gpu_available
            else []
        )
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
            "gpu_available": gpu_available,
            "cpu": {
                "now": float(cpu_hist[-1]),
                "p50": cpu_p50,
                "p95": cpu_p95,
            },
            "ram": {
                "now": float(ram_used_hist[-1]),
                "p95": ram_p95,
                "total": ram_total,
                "headroom": max(ram_total - float(ram_used_hist[-1]), 0.0),
            },
            "gpu_util": {
                "now": float(gpu_avg[-1]),
                "p50": gpu_p50,
                "p95": gpu_p95,
            },
            "gpu_delta": {
                "now": float(gpu_delta[-1]),
                "p95": delta_p95,
            },
            "gpu_mem": {
                "now": float(gpu_mem_worst[-1]),
                "p95": mem_p95,
                "headroom": float(gpu_mem_headroom_min[-1]),
                # Capacity of the GPU shown in "now" (the max-used one),
                # so the tile can read "used / total".
                "total": gpu_mem_worst_total,
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
                "cpu": cpu_hist.astype(float).tolist(),
                "gpu_avg": (
                    gpu_avg.astype(float).tolist() if gpu_available else []
                ),
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
                # than twice in the card with two different rules.
                "cpu_run_whole": _has_whole_run(cpu_run),
                "power_run_whole": _has_whole_run(
                    power_run if gpu_available else []
                ),
            },
        )

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
                g = None  # the sampler's all-zero fallback: unreported
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
            if (
                self._stale_ttl_s is None
                or (now - self._last_ok_ts) <= self._stale_ttl_s
            ):
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
