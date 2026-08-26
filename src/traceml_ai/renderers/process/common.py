"""Shared models and SQLite helpers for process telemetry."""

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Rolling window for the whole-run charts, scaled to the run so the line
# stays readable at any length (mirrors the System block's ladder).
_ROLL_MIN_S = 30.0
_ROLL_MAX_S = 300.0
_ROLL_FRACTION = 50.0  # about a fiftieth of the run
_MAX_RUN_POINTS = 120
_DEFAULT_TICK_S = 2.0

# How far back the recent-window read looks before it has seen a cadence
# to size itself from. After the first tick the caller narrows it to what
# the ranks actually sample at.
_WINDOW_MAX_AGE_S = 20.0 * 60.0

# SQLite grew numeric RANGE frame offsets in 3.28. A time frame is the
# honest one here: unreported samples leave holes in the cadence, so a ROW
# frame silently spans more wall clock than it claims. Older engines fall
# back to the row frame the System block already ships.
_HAS_RANGE_FRAME = sqlite3.sqlite_version_info >= (3, 28, 0)


def choose_window_s(span_s: float) -> float:
    """The rolling window for a run of ``span_s`` seconds, in round steps."""
    if span_s <= 0:
        return _ROLL_MIN_S
    raw = max(_ROLL_MIN_S, min(_ROLL_MAX_S, span_s / _ROLL_FRACTION))
    for step in (30.0, 60.0, 120.0, 300.0):
        if raw <= step:
            return step
    return _ROLL_MAX_S


def frame_clause(window_s: float, cadence_s: float) -> str:
    """The window frame for a rolling aggregate over ``window_s`` seconds."""
    if _HAS_RANGE_FRAME:
        return f"RANGE BETWEEN {float(window_s):.6f} PRECEDING AND CURRENT ROW"
    preceding = max(1, int(round(window_s / max(cadence_s, 1e-6))) - 1)
    return f"ROWS BETWEEN {preceding} PRECEDING AND CURRENT ROW"


@dataclass(frozen=True)
class ProcessCLISnapshot:
    """Compact terminal snapshot for process telemetry."""

    seq: Optional[int]
    cpu_used: float
    gpu_used: Optional[float]
    gpu_reserved: Optional[float]
    gpu_total: Optional[float]
    gpu_rank: Optional[int]
    gpu_used_imbalance: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "cpu_used": self.cpu_used,
            "gpu_used": self.gpu_used,
            "gpu_reserved": self.gpu_reserved,
            "gpu_total": self.gpu_total,
            "gpu_rank": self.gpu_rank,
            "gpu_used_imbalance": self.gpu_used_imbalance,
        }


@dataclass(frozen=True)
class ProcessDashboardPayload:
    """
    Dashboard payload for process telemetry UI.

    Fields
    ------
    history:
        Seq-aligned rolling history. Each entry keeps the same keys your
        existing NiceGUI frontend already expects.
    gpu_used_imbalance:
        Current cross-rank GPU used imbalance from the latest history row,
        surfaced at top level for convenient tile rendering.
    series:
        Optional chart-friendly arrays. Included for future UI use.
    """

    history: List[Dict[str, Any]]
    gpu_used_imbalance: Optional[float]
    # Series carries per-rank sample arrays and whole-run histories, so its
    # values are intentionally heterogeneous.
    series: Dict[str, Any]
    window_len: int = 0
    gpu_available: bool = False
    rollups: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "history": self.history,
            "gpu_used_imbalance": self.gpu_used_imbalance,
            "series": self.series,
            "window_len": self.window_len,
            "gpu_available": self.gpu_available,
            "rollups": self.rollups,
        }


class ProcessMetricsDB:
    """
    SQLite helper for process telemetry compute.

    This class centralizes all SQL reads used by both CLI and dashboard paths.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = str(db_path)

    def connect(self) -> sqlite3.Connection:
        """
        Open a short-lived SQLite connection configured for named-row access.
        """
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def fetch_latest_seq(self, conn: sqlite3.Connection) -> Optional[int]:
        """
        Return the latest sequence number visible in `process_samples`.

        Returns
        -------
        Optional[int]
            Latest seq, or None if the table has no seq-bearing rows.
        """
        row = conn.execute("""
            SELECT seq
            FROM process_samples
            WHERE seq IS NOT NULL
            ORDER BY id DESC
            LIMIT 1;
            """).fetchone()
        if row is None or row["seq"] is None:
            return None
        return int(row["seq"])

    def fetch_latest_seq_per_rank(
        self, conn: sqlite3.Connection
    ) -> Dict[int, int]:
        """
        Return the latest seq observed for each rank.

        Returns
        -------
        dict[int, int]
            Mapping rank -> latest seq for that rank.
        """
        rows = conn.execute("""
            SELECT rank, MAX(seq) AS max_seq
            FROM process_samples
            WHERE rank IS NOT NULL
              AND seq IS NOT NULL
            GROUP BY rank
            ORDER BY rank ASC;
            """).fetchall()

        out: Dict[int, int] = {}
        for row in rows:
            if row["rank"] is None or row["max_seq"] is None:
                continue
            out[int(row["rank"])] = int(row["max_seq"])
        return out

    def fetch_rows_for_seq_all_ranks(
        self,
        conn: sqlite3.Connection,
        seq: int,
    ) -> List[sqlite3.Row]:
        """
        Fetch all rows for one exact seq across all ranks.

        Parameters
        ----------
        seq:
            Sequence number to read.

        Returns
        -------
        list[sqlite3.Row]
            Rows for that seq ordered by rank then id.
        """
        return conn.execute(
            """
            SELECT *
            FROM process_samples
            WHERE seq = ?
            ORDER BY rank ASC, id ASC;
            """,
            (int(seq),),
        ).fetchall()

    def fetch_committed_seq(self, conn: sqlite3.Connection) -> Optional[int]:
        """
        Return the latest seq completed by all active ranks.

        Semantics
        ---------
        Equivalent to the old in-memory logic:
        committed_seq = min(last_seq_per_rank.values())

        Returns
        -------
        Optional[int]
            Latest globally committed seq, or None if no active ranks exist.
        """
        per_rank = self.fetch_latest_seq_per_rank(conn)
        if not per_rank:
            return None
        return min(per_rank.values())

    def newest_sample_ts(self, conn: sqlite3.Connection) -> Optional[float]:
        """The newest sample clock, read once per tick and reused.

        Every bound below derives from it. Deriving them separately would
        let the retention pruner delete rows between two statements and
        leave one computation reading a window the other never saw.
        """
        row = conn.execute(
            "SELECT MAX(sample_ts_s) FROM process_samples"
        ).fetchone()
        return float(row[0]) if row and row[0] is not None else None

    def fetch_recent_rank_window(
        self,
        conn: sqlite3.Connection,
        window_n: int = 100,
        newest_ts: Optional[float] = None,
        max_age_s: float = _WINDOW_MAX_AGE_S,
    ) -> List[sqlite3.Row]:
        """
        The last ``window_n`` samples of EVERY rank, newest last.

        Per rank, not globally: a rank that stopped reporting must keep its
        own history instead of being squeezed out by livelier peers, and a
        rank that never reports must not shrink everyone else's window.
        The dashboard reads ranks on their own clocks, so nothing here is
        aligned on a shared seq (a dead rank froze the whole block when it
        was: its last committed seq bounded every other rank).
        """
        # Bounded by time before the partition runs: without it the
        # ROW_NUMBER scans every row ever written to rank the last hundred,
        # which grows with the run. The bound is generous (a slow sampler
        # still fills the window) and always inside the retention horizon.
        floor_ts = None
        if newest_ts is not None:
            floor_ts = newest_ts - max(60.0, float(max_age_s))
        return conn.execute(
            """
            WITH recent AS (
                SELECT * FROM process_samples
                WHERE COALESCE(global_rank, rank) IS NOT NULL
                  AND (? IS NULL OR sample_ts_s >= ?)
            ),
            ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY COALESCE(global_rank, rank)
                        ORDER BY seq DESC, id DESC
                    ) AS rn
                FROM recent
            )
            SELECT * FROM ranked
            WHERE rn <= ?
            ORDER BY COALESCE(global_rank, rank) ASC, seq ASC, id ASC;
            """,
            (floor_ts, floor_ts, int(max(1, window_n))),
        ).fetchall()

    def fetch_rank_latest(self, conn: sqlite3.Connection) -> List[sqlite3.Row]:
        """The newest row of EVERY rank, however long ago it arrived.

        The windowed read above is bounded by cadence, so a rank silent
        for longer than that bound has no rows in it and would drop out of
        the block entirely: the surface would forget the dead rank a few
        minutes after it died, which is exactly when its death starts to
        matter. This read is one row per rank and answers "who has ever
        reported, and when did each last speak".
        """
        return conn.execute("""
            SELECT * FROM process_samples
            WHERE id IN (
                SELECT MAX(id) FROM process_samples
                WHERE COALESCE(global_rank, rank) IS NOT NULL
                GROUP BY COALESCE(global_rank, rank)
            )
            ORDER BY COALESCE(global_rank, rank) ASC;
            """).fetchall()

    def fetch_rank_run_history(
        self,
        conn: sqlite3.Connection,
        value_sql: str,
        *,
        min_span_s: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Whole-run per-rank history of ``value_sql``, rolling and decimated.

        One row per kept point, at most ``_MAX_RUN_POINTS`` per rank, so the
        payload holds its size however long the run runs. Points whose
        rolling window is still partial are excluded rather than drawn
        short, and the stride is computed over the points that survive that
        exclusion (a stride over the raw count both overshoots the budget
        and throws away detail).

        Returns [] when the run has not yet outlived ``min_span_s``: the
        recent window already tells that story, so the query is skipped
        entirely rather than computed and discarded by the view.
        """
        try:
            row = conn.execute(
                f"SELECT MIN(sample_ts_s), MAX(sample_ts_s), COUNT(*), "
                f"COUNT(DISTINCT COALESCE(global_rank, rank)) "
                f"FROM process_samples WHERE sample_ts_s IS NOT NULL "
                f"AND ({value_sql}) IS NOT NULL"
            ).fetchone()
        except sqlite3.Error:
            return []
        if row is None or row[0] is None or row[1] is None:
            return []
        first, last, count, ranks = (
            float(row[0]),
            float(row[1]),
            int(row[2] or 0),
            max(1, int(row[3] or 1)),
        )
        span = last - first
        if span <= max(0.0, float(min_span_s)) or count <= 2:
            return []
        window_s = choose_window_s(span)
        per_rank = max(1, count // ranks)
        cadence = span / max(per_rank - 1, 1)
        preceding = max(1, int(round(window_s / max(cadence, 1e-6))) - 1)
        eligible = max(0, per_rank - preceding)
        stride = max(1, (eligible + _MAX_RUN_POINTS - 1) // _MAX_RUN_POINTS)
        frame = frame_clause(window_s, cadence)
        try:
            rows = conn.execute(
                f"""
                WITH base AS (
                    SELECT
                        COALESCE(global_rank, rank) AS rank_id,
                        sample_ts_s AS ts,
                        ({value_sql}) AS v,
                        ROW_NUMBER() OVER (
                            PARTITION BY COALESCE(global_rank, rank)
                            ORDER BY sample_ts_s ASC, id ASC
                        ) AS rn
                    FROM process_samples
                    WHERE sample_ts_s IS NOT NULL
                      AND ({value_sql}) IS NOT NULL
                      AND COALESCE(global_rank, rank) IS NOT NULL
                ),
                rolled AS (
                    SELECT
                        rank_id,
                        ts,
                        rn,
                        AVG(v) OVER (
                            PARTITION BY rank_id ORDER BY ts {frame}
                        ) AS roll_avg,
                        MAX(v) OVER (
                            PARTITION BY rank_id ORDER BY ts {frame}
                        ) AS roll_max
                    FROM base
                )
                SELECT rank_id, ts, roll_avg, roll_max
                FROM rolled
                WHERE rn % ? = 0 AND rn > ?
                ORDER BY rank_id ASC, ts ASC;
                """,
                (int(stride), int(preceding)),
            ).fetchall()
        except sqlite3.Error:
            return []
        by_rank: Dict[int, Dict[str, Any]] = {}
        for rank_id, ts, roll_avg, roll_max in rows:
            entry = by_rank.setdefault(
                int(rank_id),
                {
                    "global_rank": int(rank_id),
                    "t": [],
                    "avg": [],
                    "max": [],
                    "span_s": span,
                    "window_s": window_s,
                },
            )
            entry["t"].append(float(ts))
            entry["avg"].append(float(roll_avg))
            entry["max"].append(float(roll_max))
        return [by_rank[key] for key in sorted(by_rank)]

    def fetch_seq_range_aggregates(
        self,
        conn: sqlite3.Connection,
        start_seq: int,
        end_seq: int,
    ) -> List[sqlite3.Row]:
        """
        Aggregate dashboard history over a contiguous committed seq range.

        This query preserves the previous dashboard semantics:

        - one output row per seq
        - CPU = max(cpu_percent) across ranks
        - RAM = max(ram_used_bytes) across ranks
        - RAM total = max(ram_total_bytes) across ranks
        - GPU candidate chosen from the rank with least headroom
          where headroom = gpu_mem_total_bytes - gpu_mem_reserved_bytes
        - GPU used imbalance = max(gpu_mem_used_bytes) - min(gpu_mem_used_bytes)

        Parameters
        ----------
        start_seq:
            Inclusive sequence lower bound.
        end_seq:
            Inclusive sequence upper bound.

        Returns
        -------
        list[sqlite3.Row]
            One aggregated row per seq, ascending by seq.
        """
        if end_seq < start_seq:
            return []

        return conn.execute(
            """
            WITH seq_rows AS (
                SELECT *
                FROM process_samples
                WHERE seq BETWEEN ? AND ?
            ),
            seq_base AS (
                SELECT
                    seq,
                    MAX(cpu_percent) AS cpu_max,
                    MAX(ram_used_bytes) AS ram_used_max,
                    MAX(ram_total_bytes) AS ram_total,
                    MAX(sample_ts_s) AS sample_ts_s
                FROM seq_rows
                GROUP BY seq
            ),
            gpu_candidates AS (
                SELECT
                    seq,
                    rank,
                    gpu_mem_used_bytes AS gpu_used,
                    gpu_mem_total_bytes AS gpu_total,
                    (gpu_mem_total_bytes - gpu_mem_reserved_bytes) AS gpu_headroom,
                    ROW_NUMBER() OVER (
                        PARTITION BY seq
                        ORDER BY (gpu_mem_total_bytes - gpu_mem_reserved_bytes) ASC,
                                 rank ASC,
                                 id ASC
                    ) AS rn
                FROM seq_rows
                WHERE gpu_available = 1
                  AND gpu_mem_used_bytes IS NOT NULL
                  AND gpu_mem_reserved_bytes IS NOT NULL
                  AND gpu_mem_total_bytes IS NOT NULL
            ),
            gpu_choice AS (
                SELECT
                    seq,
                    rank AS gpu_rank,
                    gpu_used,
                    gpu_total,
                    gpu_headroom
                FROM gpu_candidates
                WHERE rn = 1
            ),
            gpu_imbalance AS (
                SELECT
                    seq,
                    CASE
                        WHEN COUNT(gpu_mem_used_bytes) > 0
                        THEN MAX(gpu_mem_used_bytes) - MIN(gpu_mem_used_bytes)
                        ELSE NULL
                    END AS gpu_used_imbalance
                FROM seq_rows
                WHERE gpu_mem_used_bytes IS NOT NULL
                GROUP BY seq
            )
            SELECT
                b.seq,
                b.sample_ts_s,
                b.cpu_max,
                b.ram_used_max,
                b.ram_total,
                g.gpu_used,
                g.gpu_total,
                g.gpu_headroom,
                g.gpu_rank,
                gi.gpu_used_imbalance
            FROM seq_base b
            LEFT JOIN gpu_choice g
                ON b.seq = g.seq
            LEFT JOIN gpu_imbalance gi
                ON b.seq = gi.seq
            ORDER BY b.seq ASC;
            """,
            (int(start_seq), int(end_seq)),
        ).fetchall()
