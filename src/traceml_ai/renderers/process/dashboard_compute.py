"""
Dashboard compute for process telemetry.

One payload per tick, read from ``process_samples`` alone. Ranks are read on
their own clocks: every rank keeps its own window and its own liveness, so a
rank that dies leaves a dimmed row instead of stopping the block (the seq
alignment this replaces bounded every rank by the slowest one's last
committed seq).

Every decision the view would otherwise have to make is made here: which
view the charts show, which rank each tile speaks for, whether the per-rank
rows deserve to open. The section formats what it is handed.
"""

import statistics
import time
from typing import Any, Dict, List, Optional, Sequence

from traceml_ai.diagnostics.process.policy import DEFAULT_PROCESS_POLICY

from .common import ProcessDashboardPayload, ProcessMetricsDB

# Whole-run charts add nothing until the run outgrows its sample window.
_RUN_VIEW_FACTOR = 1.2

# A rank is stale once its newest sample is older than this many ticks,
# measured on the AGGREGATOR's arrival clock: rank wall clocks drift
# between machines, and one fast node would otherwise mark every other
# node dead forever.
_STALE_TICKS = 3.0
_DEFAULT_TICK_S = 2.0

# The rows open on the diagnosis engine's own imbalance bar, never on a
# number invented here.
IMBALANCE_OPEN_PCT = float(
    DEFAULT_PROCESS_POLICY.rank_gpu_memory_imbalance_warn_percent
)

# The trigger reads each rank's window MEDIAN reserved, not its newest
# sample. That debounces the staggered CUDA ramp without holding state
# across ticks, which a tick streak cannot do: a finished run rendered once
# (a replay, or a reader opening the page after training ends) never
# accumulates a streak, so its rows would stay shut however lopsided it
# was.

_CPU_CAPACITY_SQL = (
    "cpu_percent / (100.0 * NULLIF(cpu_logical_core_count, 0)) * 100.0"
)
_RSS_SQL = "ram_used_bytes"

# Samples per rank in the recent-window view.
_WINDOW_N = 100


def _empty_dashboard_series() -> Dict[str, Any]:
    """The complete series schema with no observations.

    Every degraded path returns this, so a consumer never has to ask
    whether a key exists on a payload that failed to read.
    """
    return {
        "cpu_capacity": [],
        "rss": [],
        "cpu_capacity_run": [],
        "rss_run": [],
    }


def _opt_float(value: Any) -> Optional[float]:
    """Float or None: an unreported column stays unreported, never 0."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _gpu_reported(row: Any) -> bool:
    """Whether a row's GPU columns are a real reading.

    The sampler writes zeros when torch reports no device, so a zero
    capacity is the absence marker, not a measurement. The System package
    excludes the same shape from its own reads.
    """
    total = _opt_float(row["gpu_mem_total_bytes"])
    return bool(row["gpu_available"]) and total is not None and total > 0.0


def _median(values: Sequence[float]) -> Optional[float]:
    return statistics.median(values) if values else None


class ProcessDashboardComputer:
    """Compute the dashboard payload for process telemetry."""

    def __init__(
        self,
        db_path: str,
        dashboard_max_rows: int = _WINDOW_N,
        stale_ttl_s: Optional[float] = 30.0,
    ) -> None:
        self._db = ProcessMetricsDB(db_path=db_path)
        # Per RANK, not in total: the System block's window is 100 samples
        # and the two blocks are read side by side.
        self._window_n = max(1, int(dashboard_max_rows))
        self._last_ok: Optional[Dict[str, Any]] = None
        self._last_ok_ts: float = 0.0
        self._stale_ttl_s: Optional[float] = (
            float(stale_ttl_s) if stale_ttl_s is not None else None
        )

    def compute(self, window_n: Optional[int] = None) -> Dict[str, Any]:
        """Return the dashboard payload, reusing the last good one on error."""
        try:
            with self._db.connect() as conn:
                out = self._compute(conn, int(window_n or self._window_n))
        except Exception:
            return self._return_stale()
        self._last_ok = out
        self._last_ok_ts = time.time()
        return out

    # --- payload ---------------------------------------------------------
    def _compute(self, conn: Any, window_n: int) -> Dict[str, Any]:
        rows = self._db.fetch_recent_rank_window(conn, window_n=window_n)
        if not rows:
            return self._empty_payload()

        by_rank: Dict[int, List[Any]] = {}
        for row in rows:
            key = row["global_rank"]
            key = int(key if key is not None else row["rank"])
            by_rank.setdefault(key, []).append(row)

        newest_recv = (
            max(float(row["recv_ts_ns"] or 0.0) for row in rows) / 1e9
        )
        tick = self._tick_seconds(by_rank)
        ranks = [
            self._rank_facts(rank_id, rank_rows, newest_recv, tick)
            for rank_id, rank_rows in sorted(by_rank.items())
        ]
        live = [rank for rank in ranks if not rank["stale"]]
        # Aggregates speak for the ranks that are still reporting: a rank
        # that died mid-step would otherwise pin a tile or the trigger to
        # whatever it happened to be doing when it stopped.
        source = live or ranks

        gpu_available = any(rank["gpu_total"] is not None for rank in ranks)
        window_span = self._window_span(rows)
        min_run_span = window_span * _RUN_VIEW_FACTOR
        cpu_run = self._db.fetch_rank_run_history(
            conn, _CPU_CAPACITY_SQL, min_span_s=min_run_span
        )
        rss_run = self._db.fetch_rank_run_history(
            conn, _RSS_SQL, min_span_s=min_run_span
        )

        imbalance = self._reserved_imbalance(source)
        rollups: Dict[str, Any] = {
            "ranks": ranks,
            "ranks_reporting": len(live),
            "ranks_total": len(ranks),
            "ranks_stale": len(ranks) - len(live),
            "gpu_available": gpu_available,
            "cpu_capacity": self._cpu_rollup(source),
            "rss": self._rss_rollup(source),
            "cuda": self._cuda_rollup(source),
            "reserved_imbalance_pct": imbalance,
            "rows_over": self._rows_trigger(source, imbalance),
            "tick_s": tick,
        }
        # One series per chart, never both: whichever view the charts will
        # draw is the only one worth the payload it costs.
        series = _empty_dashboard_series()
        series["cpu_capacity_run"] = cpu_run
        series["rss_run"] = rss_run
        if not cpu_run:
            series["cpu_capacity"] = self._window_series(
                by_rank, self._cpu_capacity_of
            )
        if not rss_run:
            series["rss"] = self._window_series(
                by_rank, lambda row: _opt_float(row["ram_used_bytes"])
            )

        return ProcessDashboardPayload(
            history=[],
            gpu_used_imbalance=None,
            series=series,
            window_len=max(
                (
                    len(entry["t"])
                    for entry in (
                        series["cpu_capacity"] or series["cpu_capacity_run"]
                    )
                ),
                default=0,
            ),
            gpu_available=gpu_available,
            rollups=rollups,
        ).to_dict()

    # --- per rank --------------------------------------------------------
    def _tick_seconds(self, by_rank: Dict[int, List[Any]]) -> float:
        """The fastest rank's own cadence, floored at the sampler default."""
        cadences = []
        for rank_rows in by_rank.values():
            stamps = [
                _opt_float(row["sample_ts_s"])
                for row in rank_rows
                if row["sample_ts_s"] is not None
            ]
            if len(stamps) > 1 and stamps[-1] > stamps[0]:
                cadences.append((stamps[-1] - stamps[0]) / (len(stamps) - 1))
        return (
            max(_DEFAULT_TICK_S, min(cadences))
            if cadences
            else (_DEFAULT_TICK_S)
        )

    def _cpu_capacity_of(self, row: Any) -> Optional[float]:
        """CPU as a share of the host's capacity, bounded 0-100.

        Raw ``cpu_percent`` sums cores, so the same busy trainer reads 100
        on a 4-core box and 100 on a 48-core box; only the bounded form is
        comparable, and the core count is already stored beside it.
        """
        used = _opt_float(row["cpu_percent"])
        cores = _opt_float(row["cpu_logical_core_count"])
        if used is None or not cores:
            return None
        return used / (100.0 * cores) * 100.0

    def _rank_facts(
        self,
        rank_id: int,
        rank_rows: List[Any],
        newest_recv: float,
        tick: float,
    ) -> Dict[str, Any]:
        newest = rank_rows[-1]
        caps = [
            value
            for value in (self._cpu_capacity_of(row) for row in rank_rows)
            if value is not None
        ]
        allocs = [
            _opt_float(row["gpu_mem_used_bytes"])
            for row in rank_rows
            if _gpu_reported(row)
        ]
        allocs = [value for value in allocs if value is not None]
        # The newest row is not always a reading: the last samples of a
        # run land during teardown, after torch has let the device go, so
        # anchoring the GPU tiles on it blanks them exactly when someone
        # inspects a finished run.
        reported = [row for row in rank_rows if _gpu_reported(row)]
        newest_gpu = reported[-1] if reported else None
        age = max(0.0, newest_recv - float(newest["recv_ts_ns"] or 0.0) / 1e9)
        return {
            "global_rank": rank_id,
            "node_rank": (
                int(newest["node_rank"])
                if newest["node_rank"] is not None
                else None
            ),
            "gpu_index": (
                int(newest["gpu_device_index"])
                if newest["gpu_device_index"] is not None
                else None
            ),
            # The window median, never the last tick: CPU is noisy and the
            # allocator's live bytes are a sawtooth this 2 s cadence
            # undersamples, so one raw sample lands wherever the step
            # happened to be.
            "cpu_capacity": _median(caps),
            # Reserved twice: the newest reading is what the tiles and the
            # rows show (it is what the allocator holds now), the window
            # median is what the trigger judges.
            "gpu_reserved_p50": _median(
                [
                    value
                    for value in (
                        _opt_float(row["gpu_mem_reserved_bytes"])
                        for row in rank_rows
                        if _gpu_reported(row)
                    )
                    if value is not None
                ]
            ),
            "ram_used": _opt_float(newest["ram_used_bytes"]),
            "ram_total": _opt_float(newest["ram_total_bytes"]),
            "gpu_alloc": _median(allocs) if allocs else None,
            "gpu_reserved": (
                _opt_float(newest_gpu["gpu_mem_reserved_bytes"])
                if newest_gpu is not None
                else None
            ),
            "gpu_total": (
                _opt_float(newest_gpu["gpu_mem_total_bytes"])
                if newest_gpu is not None
                else None
            ),
            "age_s": age,
            "stale": age > _STALE_TICKS * tick,
        }

    # --- rollups ---------------------------------------------------------
    def _cpu_rollup(self, ranks: List[Dict[str, Any]]) -> Dict[str, Any]:
        values = [
            (rank["cpu_capacity"], rank["global_rank"])
            for rank in ranks
            if rank["cpu_capacity"] is not None
        ]
        if not values:
            return {"p50": None, "worst": None, "worst_rank": None}
        worst, worst_rank = max(values)
        return {
            "p50": _median([value for value, _rank in values]),
            "worst": worst,
            "worst_rank": worst_rank,
        }

    def _rss_rollup(self, ranks: List[Dict[str, Any]]) -> Dict[str, Any]:
        values = [rank for rank in ranks if rank["ram_used"] is not None]
        if not values:
            return {"used": None, "total": None, "rank": None}
        worst = max(values, key=lambda rank: rank["ram_used"])
        return {
            "used": worst["ram_used"],
            "total": worst["ram_total"],
            "rank": worst["global_rank"],
        }

    def _cuda_rollup(self, ranks: List[Dict[str, Any]]) -> Dict[str, Any]:
        allocs = [
            rank["gpu_alloc"]
            for rank in ranks
            if rank["gpu_alloc"] is not None
        ]
        reserved = [
            rank
            for rank in ranks
            if rank["gpu_reserved"] is not None and rank["gpu_total"]
        ]
        out: Dict[str, Any] = {
            "alloc_p50": _median(allocs),
            "reserved": None,
            "reserved_total": None,
            "reserved_rank": None,
        }
        if reserved:
            # The rank with the least headroom, not the most reserved: on
            # mixed devices those are different ranks, and the one nearest
            # its own ceiling is the one that fails first.
            tightest = min(
                reserved,
                key=lambda rank: rank["gpu_total"] - rank["gpu_reserved"],
            )
            out["reserved"] = tightest["gpu_reserved"]
            out["reserved_total"] = tightest["gpu_total"]
            out["reserved_rank"] = tightest["global_rank"]
        return out

    def _reserved_imbalance(
        self, ranks: List[Dict[str, Any]]
    ) -> Optional[float]:
        """Reserved spread across ranks, the engine's own definition.

        Reserved, not allocated: allocated differs across ranks by which
        step phase each sampler caught, which reads as GB of imbalance on a
        perfectly healthy run.
        """
        values = [
            rank["gpu_reserved_p50"]
            for rank in ranks
            if rank["gpu_reserved_p50"] is not None
        ]
        if len(values) < 2:
            return None
        high = max(values)
        if high <= 0:
            return 0.0
        return max(0.0, (high - min(values)) / high * 100.0)

    def _rows_trigger(
        self, ranks: List[Dict[str, Any]], imbalance: Optional[float]
    ) -> bool:
        """Whether the per-rank rows have earned opening themselves.

        Armed only once every reporting rank has HELD an allocation across
        its window: ranks reach their first CUDA allocation seconds to
        minutes apart, and an unarmed trigger reads that ramp as total
        imbalance on every run's first ticks.
        """
        armed = bool(ranks) and all(
            rank["gpu_reserved_p50"] is not None
            and rank["gpu_reserved_p50"] > 0
            for rank in ranks
        )
        return bool(
            armed and imbalance is not None and imbalance >= IMBALANCE_OPEN_PCT
        )

    # --- series ----------------------------------------------------------
    def _window_span(self, rows: Sequence[Any]) -> float:
        stamps = [
            _opt_float(row["sample_ts_s"])
            for row in rows
            if row["sample_ts_s"] is not None
        ]
        stamps = [value for value in stamps if value is not None]
        return max(stamps) - min(stamps) if len(stamps) > 1 else 0.0

    def _window_series(
        self, by_rank: Dict[int, List[Any]], value_of: Any
    ) -> List[Dict[str, Any]]:
        """Raw per-rank samples, each rank on its own clock."""
        out = []
        for rank_id, rank_rows in sorted(by_rank.items()):
            stamps, values = [], []
            for row in rank_rows:
                stamp = _opt_float(row["sample_ts_s"])
                value = value_of(row)
                if stamp is None or value is None:
                    continue
                stamps.append(stamp)
                values.append(value)
            if stamps:
                out.append({"global_rank": rank_id, "t": stamps, "v": values})
        return out

    # --- degraded paths --------------------------------------------------
    def _return_stale(self) -> Dict[str, Any]:
        if self._last_ok is not None and (
            self._stale_ttl_s is None
            or (time.time() - self._last_ok_ts) <= self._stale_ttl_s
        ):
            return self._last_ok
        return self._empty_payload()

    def _empty_payload(self) -> Dict[str, Any]:
        return ProcessDashboardPayload(
            history=[],
            gpu_used_imbalance=None,
            series=_empty_dashboard_series(),
            window_len=0,
            gpu_available=False,
            rollups={},
        ).to_dict()
