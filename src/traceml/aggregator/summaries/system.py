import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Optional


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


def _append_text(path: str, text: str) -> None:
    """Write text to a file, prepending a blank line if file already exists."""
    with open(path, "a+", encoding="utf-8") as f:
        f.seek(0, 2)
        if f.tell() > 0:
            f.write("\n")
        f.write(text.rstrip() + "\n")


@dataclass
class SystemSummaryAgg:
    """
    Aggregated system metrics loaded from `system_samples`.

    All memory fields remain in raw bytes while aggregating and are converted
    to GB only at formatting/output time.
    """

    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    system_samples: int = 0

    cpu_avg_percent: Optional[float] = None
    cpu_peak_percent: Optional[float] = None

    ram_avg_bytes: Optional[float] = None
    ram_peak_bytes: Optional[float] = None
    ram_total_bytes: Optional[float] = None

    gpu_available: Optional[bool] = None
    gpu_count: Optional[int] = None

    gpu_util_avg_percent: Optional[float] = None
    gpu_util_peak_percent: Optional[float] = None

    gpu_mem_avg_bytes: Optional[float] = None
    gpu_mem_peak_bytes: Optional[float] = None

    gpu_temp_avg_c: Optional[float] = None
    gpu_temp_peak_c: Optional[float] = None

    gpu_power_avg_w: Optional[float] = None
    gpu_power_peak_w: Optional[float] = None


def _b_to_gb(x: Optional[float]) -> Optional[float]:
    """Convert bytes to decimal GB for display."""
    if x is None:
        return None
    return float(x) / 1e9


def _fmt(x: Optional[float], suffix: str = "", ndigits: int = 1) -> str:
    """Format optional numeric values for card output."""
    return "n/a" if x is None else f"{x:.{ndigits}f}{suffix}"


def _load_system_summary_agg(
    conn: sqlite3.Connection,
    *,
    rank: Optional[int] = None,
    max_system_rows: int = 5_000,
) -> SystemSummaryAgg:
    """
    Load aggregated system metrics directly from `system_samples`.

    Parameters
    ----------
    conn:
        Open SQLite connection.
    rank:
        Optional rank filter. If None, aggregates across all ranks.
    max_system_rows:
        Safety cap on rows included in aggregation.

    Returns
    -------
    SystemSummaryAgg
        Aggregated summary values ready for formatting.
    """
    where_clause = ""
    params: list[Any] = []
    if rank is not None:
        where_clause = "WHERE rank = ?"
        params.append(int(rank))

    count_sql = f"""
        SELECT
            COUNT(*),
            MIN(sample_ts_s),
            MAX(sample_ts_s)
        FROM (
            SELECT sample_ts_s
            FROM system_samples
            {where_clause}
            ORDER BY id ASC
            LIMIT ?
        );
    """
    count_row = conn.execute(
        count_sql, (*params, int(max_system_rows))
    ).fetchone()
    n_rows = int(count_row[0] or 0)
    first_ts = float(count_row[1]) if count_row[1] is not None else None
    last_ts = float(count_row[2]) if count_row[2] is not None else None

    agg_sql = f"""
        SELECT
            AVG(cpu_percent),
            MAX(cpu_percent),

            AVG(ram_used_bytes),
            MAX(ram_used_bytes),
            MAX(ram_total_bytes),

            MAX(gpu_available),
            MAX(gpu_count),

            AVG(gpu_util_avg),
            MAX(gpu_util_peak),

            AVG(gpu_mem_used_avg_bytes),
            MAX(gpu_mem_used_peak_bytes),

            AVG(gpu_temp_avg_c),
            MAX(gpu_temp_peak_c),

            AVG(gpu_power_avg_w),
            MAX(gpu_power_peak_w)
        FROM (
            SELECT *
            FROM system_samples
            {where_clause}
            ORDER BY id ASC
            LIMIT ?
        );
    """
    row = conn.execute(agg_sql, (*params, int(max_system_rows))).fetchone()

    return SystemSummaryAgg(
        first_ts=first_ts,
        last_ts=last_ts,
        system_samples=n_rows,
        cpu_avg_percent=float(row[0]) if row[0] is not None else None,
        cpu_peak_percent=float(row[1]) if row[1] is not None else None,
        ram_avg_bytes=float(row[2]) if row[2] is not None else None,
        ram_peak_bytes=float(row[3]) if row[3] is not None else None,
        ram_total_bytes=float(row[4]) if row[4] is not None else None,
        gpu_available=bool(row[5]) if row[5] is not None else None,
        gpu_count=int(row[6]) if row[6] is not None else None,
        gpu_util_avg_percent=float(row[7]) if row[7] is not None else None,
        gpu_util_peak_percent=float(row[8]) if row[8] is not None else None,
        gpu_mem_avg_bytes=float(row[9]) if row[9] is not None else None,
        gpu_mem_peak_bytes=float(row[10]) if row[10] is not None else None,
        gpu_temp_avg_c=float(row[11]) if row[11] is not None else None,
        gpu_temp_peak_c=float(row[12]) if row[12] is not None else None,
        gpu_power_avg_w=float(row[13]) if row[13] is not None else None,
        gpu_power_peak_w=float(row[14]) if row[14] is not None else None,
    )


def _build_system_card(agg: SystemSummaryAgg) -> tuple[str, Dict[str, Any]]:
    """
    Build a clean, shareable SYSTEM summary card.

    Output is plain text with fixed-width boundaries so it renders cleanly in
    terminal, logs, Slack code blocks, GitHub comments, and saved text files.
    """
    duration_s = None
    if (
        agg.first_ts is not None
        and agg.last_ts is not None
        and agg.last_ts >= agg.first_ts
    ):
        duration_s = agg.last_ts - agg.first_ts

    ram_avg_gb = _b_to_gb(agg.ram_avg_bytes)
    ram_peak_gb = _b_to_gb(agg.ram_peak_bytes)
    ram_total_gb = _b_to_gb(agg.ram_total_bytes)

    gpu_mem_avg_gb = _b_to_gb(agg.gpu_mem_avg_bytes)
    gpu_mem_peak_gb = _b_to_gb(agg.gpu_mem_peak_bytes)

    width = 78
    inner_width = width - 4

    def border() -> str:
        return "+" + "-" * (width - 2) + "+"

    def row(text: str = "") -> str:
        return f"|  {text:<{inner_width}}|"

    header = (
        f"TraceML System Summary | duration {_fmt(duration_s, 's', 1)}"
        f" | samples {agg.system_samples}"
    )

    lines: list[str] = [
        border(),
        row(header),
        border(),
        row("SYSTEM"),
        row(),
        row(
            f"CPU       avg {_fmt(agg.cpu_avg_percent, '%', 1)}   "
            f"peak {_fmt(agg.cpu_peak_percent, '%', 1)}"
        ),
        row(
            f"RAM       avg {_fmt(ram_avg_gb, ' GB', 1)}   "
            f"peak {_fmt(ram_peak_gb, ' GB', 1)}   "
            f"total {_fmt(ram_total_gb, ' GB', 1)}"
        ),
    ]

    if agg.gpu_util_avg_percent is not None:
        lines.append(
            row(
                f"GPU util  avg {_fmt(agg.gpu_util_avg_percent, '%', 1)}   "
                f"peak {_fmt(agg.gpu_util_peak_percent, '%', 1)}"
            )
        )
        lines.append(
            row(
                f"GPU mem   avg {_fmt(gpu_mem_avg_gb, ' GB', 1)}   "
                f"peak {_fmt(gpu_mem_peak_gb, ' GB', 1)}"
            )
        )
        lines.append(
            row(
                f"GPU temp  avg {_fmt(agg.gpu_temp_avg_c, ' C', 1)}   "
                f"peak {_fmt(agg.gpu_temp_peak_c, ' C', 1)}"
            )
        )
        lines.append(
            row(
                f"GPU power avg {_fmt(agg.gpu_power_avg_w, ' W', 1)}   "
                f"peak {_fmt(agg.gpu_power_peak_w, ' W', 1)}"
            )
        )
    else:
        if agg.gpu_available is False:
            gpu_msg = (
                "unavailable (no NVIDIA GPU detected or NVML inaccessible)"
            )
        elif (agg.gpu_count or 0) > 0:
            gpu_msg = "detected, but no per-GPU samples were recorded"
        else:
            gpu_msg = "n/a"
        lines.append(row(f"GPU       {gpu_msg}"))

    lines.append(border())
    card = "\n".join(lines)

    summary = {
        "duration_s": duration_s,
        "system_samples": agg.system_samples,
        "cpu_avg_percent": agg.cpu_avg_percent,
        "cpu_peak_percent": agg.cpu_peak_percent,
        "ram_avg_gb": ram_avg_gb,
        "ram_peak_gb": ram_peak_gb,
        "ram_total_gb": ram_total_gb,
        "gpu_available": agg.gpu_available,
        "gpu_count": agg.gpu_count,
        "gpu_util_avg_percent": agg.gpu_util_avg_percent,
        "gpu_util_peak_percent": agg.gpu_util_peak_percent,
        "gpu_mem_avg_gb": gpu_mem_avg_gb,
        "gpu_mem_peak_gb": gpu_mem_peak_gb,
        "gpu_temp_avg_c": agg.gpu_temp_avg_c,
        "gpu_temp_peak_c": agg.gpu_temp_peak_c,
        "gpu_power_avg_w": agg.gpu_power_avg_w,
        "gpu_power_peak_w": agg.gpu_power_peak_w,
        "units": {
            "memory": "GB",
            "temperature": "C",
            "power": "W",
            "util": "%",
        },
        "card": card,
    }
    return card, summary


def generate_system_summary_card(
    db_path: str,
    *,
    rank: Optional[int] = None,
    print_to_stdout: bool = True,
    max_system_rows: int = 5_000,
) -> Dict[str, Any]:
    """
    Generate a shareable SYSTEM summary card from SQL projection tables.

    Parameters
    ----------
    db_path:
        Path to the SQLite DB file.
    rank:
        Optional rank filter. If None, summarizes across all ranks.
    print_to_stdout:
        If True, print the rendered card.
    max_system_rows:
        Safety cap on rows included in aggregation.

    Returns
    -------
    Dict[str, Any]
        Structured summary JSON including the rendered `card`.
    """
    conn = sqlite3.connect(db_path)
    try:
        agg = _load_system_summary_agg(
            conn,
            rank=rank,
            max_system_rows=max_system_rows,
        )
    finally:
        conn.close()

    card, summary = _build_system_card(agg)

    # ── Write text card (prepend so step_time appends after) ───────────────
    _append_text(db_path + "_summary_card.txt", card)

    # ── Merge into shared JSON under the "system" key ──────────────────────
    # step_time uses the same pattern: load → update key → write.
    # Never blindly overwrite so both sections co-exist in the same file.
    existing = _load_json_or_empty(db_path + "_summary_card.json")
    existing["system"] = summary
    _write_json(db_path + "_summary_card.json", existing)

    if print_to_stdout:
        print(card)

    return summary
