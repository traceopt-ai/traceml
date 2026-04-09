"""
Compact end-of-run process summary generation.

This module reads process-centric metrics from the `process_samples`
projection table and produces:

1. a compact text summary for end-of-run display and sharing
2. a structured JSON payload for future automation and compare features

Design goals
------------
- Keep the printed summary short and easy to scan
- Preserve the current JSON field names as much as possible
- Keep the process summary useful for single-process and distributed runs
"""

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _append_text(path: str, text: str) -> None:
    """
    Append text to a file, inserting a blank line first if the file already
    contains content.
    """
    with open(path, "a+", encoding="utf-8") as f:
        f.seek(0, 2)
        if f.tell() > 0:
            f.write("\n")
        f.write(text.rstrip() + "\n")


def _load_json_or_empty(path: str) -> Dict[str, Any]:
    """Load JSON if present; otherwise return an empty dict."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    """Write JSON with indentation."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


@dataclass
class ProcessSummaryAgg:
    """
    Aggregated process metrics loaded from `process_samples`.

    Notes
    -----
    - Memory values remain in raw bytes during aggregation and are converted
      only at formatting / serialization time.
    - Fields are intentionally broad enough to support both single-process and
      distributed runs.
    """

    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    process_samples: int = 0

    distinct_ranks: int = 0
    distinct_pids: int = 0

    cpu_avg_percent: Optional[float] = None
    cpu_peak_percent: Optional[float] = None
    cpu_logical_core_count: Optional[int] = None

    ram_avg_bytes: Optional[float] = None
    ram_peak_bytes: Optional[float] = None
    ram_total_bytes: Optional[float] = None

    gpu_available: Optional[bool] = None
    gpu_count: Optional[int] = None
    gpu_device_index: Optional[int] = None

    gpu_mem_used_avg_bytes: Optional[float] = None
    gpu_mem_used_peak_bytes: Optional[float] = None
    gpu_mem_reserved_avg_bytes: Optional[float] = None
    gpu_mem_reserved_peak_bytes: Optional[float] = None
    gpu_mem_total_bytes: Optional[float] = None


def _b_to_gb(x: Optional[float]) -> Optional[float]:
    """Convert bytes to decimal GB for display."""
    if x is None:
        return None
    return float(x) / 1e9


def _fmt(x: Optional[float], suffix: str = "", ndigits: int = 1) -> str:
    """Format an optional numeric value for human-readable output."""
    return "n/a" if x is None else f"{x:.{ndigits}f}{suffix}"


def _share(num: Optional[float], denom: Optional[float]) -> Optional[float]:
    """Return percentage share num/denom, or None if denom is not positive."""
    if num is None or denom is None or denom <= 0.0:
        return None
    return 100.0 * num / denom


def _duration_s(agg: ProcessSummaryAgg) -> Optional[float]:
    """Return summary duration in seconds if timestamps are valid."""
    if (
        agg.first_ts is None
        or agg.last_ts is None
        or agg.last_ts < agg.first_ts
    ):
        return None
    return agg.last_ts - agg.first_ts


def _load_process_summary_agg(
    conn: sqlite3.Connection,
    *,
    rank: Optional[int] = None,
    max_process_rows: int = 10_000,
) -> ProcessSummaryAgg:
    """
    Load aggregated process metrics directly from `process_samples`.

    Parameters
    ----------
    conn:
        Open SQLite connection.
    rank:
        Optional rank filter. If None, aggregates across all ranks.
    max_process_rows:
        Safety cap on rows included in aggregation.

    Returns
    -------
    ProcessSummaryAgg
        Aggregated summary values ready for formatting.
    """
    where_clause = ""
    params: list[Any] = []

    if rank is not None:
        where_clause = "WHERE rank = ?"
        params.append(int(rank))

    base_sql = f"""
        FROM (
            SELECT *
            FROM process_samples
            {where_clause}
            ORDER BY id ASC
            LIMIT ?
        )
    """

    count_row = conn.execute(
        f"""
        SELECT
            COUNT(*),
            MIN(sample_ts_s),
            MAX(sample_ts_s),
            COUNT(DISTINCT rank),
            COUNT(DISTINCT pid)
        {base_sql};
        """,
        (*params, int(max_process_rows)),
    ).fetchone()

    n_rows = int(count_row[0] or 0)
    first_ts = float(count_row[1]) if count_row[1] is not None else None
    last_ts = float(count_row[2]) if count_row[2] is not None else None
    distinct_ranks = int(count_row[3] or 0)
    distinct_pids = int(count_row[4] or 0)

    row = conn.execute(
        f"""
        SELECT
            AVG(cpu_percent),
            MAX(cpu_percent),
            MAX(cpu_logical_core_count),

            AVG(ram_used_bytes),
            MAX(ram_used_bytes),
            MAX(ram_total_bytes),

            MAX(gpu_available),
            MAX(gpu_count),
            MIN(gpu_device_index),

            AVG(gpu_mem_used_bytes),
            MAX(gpu_mem_used_bytes),
            AVG(gpu_mem_reserved_bytes),
            MAX(gpu_mem_reserved_bytes),
            MAX(gpu_mem_total_bytes)
        {base_sql};
        """,
        (*params, int(max_process_rows)),
    ).fetchone()

    return ProcessSummaryAgg(
        first_ts=first_ts,
        last_ts=last_ts,
        process_samples=n_rows,
        distinct_ranks=distinct_ranks,
        distinct_pids=distinct_pids,
        cpu_avg_percent=float(row[0]) if row[0] is not None else None,
        cpu_peak_percent=float(row[1]) if row[1] is not None else None,
        cpu_logical_core_count=int(row[2]) if row[2] is not None else None,
        ram_avg_bytes=float(row[3]) if row[3] is not None else None,
        ram_peak_bytes=float(row[4]) if row[4] is not None else None,
        ram_total_bytes=float(row[5]) if row[5] is not None else None,
        gpu_available=bool(row[6]) if row[6] is not None else None,
        gpu_count=int(row[7]) if row[7] is not None else None,
        gpu_device_index=int(row[8]) if row[8] is not None else None,
        gpu_mem_used_avg_bytes=float(row[9]) if row[9] is not None else None,
        gpu_mem_used_peak_bytes=(
            float(row[10]) if row[10] is not None else None
        ),
        gpu_mem_reserved_avg_bytes=(
            float(row[11]) if row[11] is not None else None
        ),
        gpu_mem_reserved_peak_bytes=(
            float(row[12]) if row[12] is not None else None
        ),
        gpu_mem_total_bytes=float(row[13]) if row[13] is not None else None,
    )


def _build_takeaway(agg: ProcessSummaryAgg) -> str:
    """
    Build one concise process-centric takeaway line.

    The goal is to highlight a useful process-local interpretation rather than
    merely restating raw averages.
    """
    if agg.process_samples <= 0:
        return "n/a"

    if agg.gpu_available and agg.gpu_mem_total_bytes is not None:
        used_peak_pct = _share(
            agg.gpu_mem_used_peak_bytes,
            agg.gpu_mem_total_bytes,
        )
        if (
            agg.gpu_mem_reserved_peak_bytes is not None
            and agg.gpu_mem_used_peak_bytes is not None
            and agg.gpu_mem_reserved_peak_bytes
            > agg.gpu_mem_used_peak_bytes * 1.25
        ):
            return "process reserved noticeably more GPU memory than it actively used"

        if used_peak_pct is not None and used_peak_pct >= 90.0:
            return "process came close to the GPU memory limit"
        if used_peak_pct is not None and used_peak_pct <= 20.0:
            return "process used only a small fraction of available GPU memory"

    if (
        agg.cpu_avg_percent is not None
        and agg.cpu_logical_core_count is not None
        and agg.cpu_logical_core_count > 0
    ):
        approx_single_core_pct = 100.0 / agg.cpu_logical_core_count
        if agg.cpu_avg_percent <= approx_single_core_pct * 1.5:
            return "process CPU usage stayed relatively low on average"

    return "process resource usage looked stable overall"


def _build_gpu_line(
    *,
    gpu_available: Optional[bool],
    gpu_count: Optional[int],
    gpu_device_index: Optional[int],
    gpu_mem_used_peak_gb: Optional[float],
    gpu_mem_total_gb: Optional[float],
    gpu_mem_used_peak_pct: Optional[float],
    gpu_mem_reserved_peak_gb: Optional[float],
    gpu_mem_reserved_peak_pct: Optional[float],
) -> str:
    """
    Build one compact GPU line for the printed summary.

    This keeps the printed summary concise while still surfacing the process
    GPU memory information that matters most for performance and efficiency:
    - whether the process touched a GPU
    - which device it reported against
    - peak used memory
    - how close peak used memory came to the limit
    - whether reserved memory materially exceeded used memory
    """
    if gpu_available and gpu_mem_total_gb is not None:
        device_text = (
            f"device {gpu_device_index}"
            if gpu_device_index is not None
            else "device n/a"
        )
        parts = [
            device_text,
            f"used peak {_fmt(gpu_mem_used_peak_gb, ' GB', 1)}"
            f" / {_fmt(gpu_mem_total_gb, ' GB', 1)}",
        ]

        if gpu_mem_used_peak_pct is not None:
            parts.append(f"{_fmt(gpu_mem_used_peak_pct, '%', 1)} of limit")

        if (
            gpu_mem_reserved_peak_gb is not None
            and gpu_mem_used_peak_gb is not None
            and gpu_mem_reserved_peak_gb > gpu_mem_used_peak_gb * 1.25
        ):
            parts.append(
                f"reserved peak {_fmt(gpu_mem_reserved_peak_gb, ' GB', 1)}"
            )
            if gpu_mem_reserved_peak_pct is not None:
                parts.append(
                    f"reserved {_fmt(gpu_mem_reserved_peak_pct, '%', 1)}"
                )

        return "GPU: " + " | ".join(parts)

    if gpu_available is False:
        return "GPU: no GPU process samples were recorded"

    if (gpu_count or 0) > 0:
        return "GPU: detected, but no process GPU memory samples were recorded"

    return "GPU: n/a"


def _build_process_card(
    agg: ProcessSummaryAgg,
) -> tuple[str, Dict[str, Any]]:
    """
    Build a compact, shareable end-of-run process summary.

    Returns
    -------
    tuple[str, Dict[str, Any]]
        - text block for stdout / saved text summaries
        - structured JSON payload

    Notes
    -----
    The printed text is intentionally compact. The JSON payload retains richer
    fields so compare and future UI features do not lose fidelity.
    """
    duration_s = _duration_s(agg)

    ram_avg_gb = _b_to_gb(agg.ram_avg_bytes)
    ram_peak_gb = _b_to_gb(agg.ram_peak_bytes)
    ram_total_gb = _b_to_gb(agg.ram_total_bytes)

    gpu_mem_used_avg_gb = _b_to_gb(agg.gpu_mem_used_avg_bytes)
    gpu_mem_used_peak_gb = _b_to_gb(agg.gpu_mem_used_peak_bytes)
    gpu_mem_reserved_avg_gb = _b_to_gb(agg.gpu_mem_reserved_avg_bytes)
    gpu_mem_reserved_peak_gb = _b_to_gb(agg.gpu_mem_reserved_peak_bytes)
    gpu_mem_total_gb = _b_to_gb(agg.gpu_mem_total_bytes)

    gpu_mem_used_peak_pct = _share(
        agg.gpu_mem_used_peak_bytes,
        agg.gpu_mem_total_bytes,
    )
    gpu_mem_reserved_peak_pct = _share(
        agg.gpu_mem_reserved_peak_bytes,
        agg.gpu_mem_total_bytes,
    )

    takeaway = _build_takeaway(agg)

    lines = [
        f"TraceML Process Summary | duration {_fmt(duration_s, 's', 1)} | samples {agg.process_samples}",
        "Process",
        (f"- Scope: ranks {agg.distinct_ranks} | pids {agg.distinct_pids}"),
        (
            f"- CPU: avg {_fmt(agg.cpu_avg_percent, '%', 1)}, "
            f"peak {_fmt(agg.cpu_peak_percent, '%', 1)}"
            + (
                f" | cores {_fmt(float(agg.cpu_logical_core_count), '', 0)}"
                if agg.cpu_logical_core_count is not None
                else ""
            )
        ),
        (
            f"- RSS: avg {_fmt(ram_avg_gb, ' GB', 1)}, "
            f"peak {_fmt(ram_peak_gb, ' GB', 1)} / {_fmt(ram_total_gb, ' GB', 1)}"
        ),
        (
            "- "
            + _build_gpu_line(
                gpu_available=agg.gpu_available,
                gpu_count=agg.gpu_count,
                gpu_device_index=agg.gpu_device_index,
                gpu_mem_used_peak_gb=gpu_mem_used_peak_gb,
                gpu_mem_total_gb=gpu_mem_total_gb,
                gpu_mem_used_peak_pct=gpu_mem_used_peak_pct,
                gpu_mem_reserved_peak_gb=gpu_mem_reserved_peak_gb,
                gpu_mem_reserved_peak_pct=gpu_mem_reserved_peak_pct,
            )
        ),
        f"- Takeaway: {takeaway}",
    ]
    card = "\n".join(lines)

    summary = {
        "duration_s": duration_s,
        "process_samples": agg.process_samples,
        "distinct_ranks": agg.distinct_ranks,
        "distinct_pids": agg.distinct_pids,
        "cpu_avg_percent": agg.cpu_avg_percent,
        "cpu_peak_percent": agg.cpu_peak_percent,
        "cpu_logical_core_count": agg.cpu_logical_core_count,
        "ram_avg_gb": ram_avg_gb,
        "ram_peak_gb": ram_peak_gb,
        "ram_total_gb": ram_total_gb,
        "gpu_available": agg.gpu_available,
        "gpu_count": agg.gpu_count,
        "gpu_device_index": agg.gpu_device_index,
        "gpu_mem_used_avg_gb": gpu_mem_used_avg_gb,
        "gpu_mem_used_peak_gb": gpu_mem_used_peak_gb,
        "gpu_mem_reserved_avg_gb": gpu_mem_reserved_avg_gb,
        "gpu_mem_reserved_peak_gb": gpu_mem_reserved_peak_gb,
        "gpu_mem_total_gb": gpu_mem_total_gb,
        "gpu_mem_used_peak_pct": gpu_mem_used_peak_pct,
        "gpu_mem_reserved_peak_pct": gpu_mem_reserved_peak_pct,
        "takeaway": takeaway,
        "units": {
            "memory": "GB",
            "cpu": "%",
        },
        # Keep `card` for backward compatibility with existing callers/files.
        "card": card,
    }
    return card, summary


def generate_process_summary_card(
    db_path: str,
    *,
    rank: Optional[int] = None,
    print_to_stdout: bool = True,
    max_process_rows: int = 10_000,
) -> Dict[str, Any]:
    """
    Generate a compact PROCESS summary from SQL projection tables.

    Parameters
    ----------
    db_path:
        Path to the SQLite DB file.
    rank:
        Optional rank filter. If None, summarizes across all ranks.
    print_to_stdout:
        If True, print the rendered summary.
    max_process_rows:
        Safety cap on rows included in aggregation.

    Returns
    -------
    Dict[str, Any]
        Structured summary JSON including the rendered `card`.
    """
    conn = sqlite3.connect(db_path)
    try:
        agg = _load_process_summary_agg(
            conn,
            rank=rank,
            max_process_rows=max_process_rows,
        )
    finally:
        conn.close()

    card, process_summary = _build_process_card(agg)

    _append_text(db_path + "_summary_card.txt", card)

    existing = _load_json_or_empty(db_path + "_summary_card.json")
    existing["process"] = process_summary
    _write_json(db_path + "_summary_card.json", existing)

    if print_to_stdout:
        print(card)

    return process_summary
