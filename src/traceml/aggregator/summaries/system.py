import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Optional

import msgspec


@dataclass
class _Agg:
    # envelope time
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    system_rows: int = 0

    # CPU
    cpu_sum: float = 0.0
    cpu_max: float = 0.0
    cpu_n: int = 0

    # RAM
    ram_used_sum_b: float = 0.0
    ram_used_max_b: float = 0.0
    ram_total_max_b: float = 0.0
    ram_n: int = 0

    # GPU availability
    gpu_available: Optional[bool] = None
    gpu_count: Optional[int] = None

    # GPU util
    gpu_util_sum: float = 0.0
    gpu_util_max: float = 0.0
    gpu_util_n: int = 0

    gpu0_util_sum: float = 0.0
    gpu0_util_max: float = 0.0
    gpu0_util_n: int = 0

    # GPU memory
    gpu_mem_used_sum_b: float = 0.0
    gpu_mem_used_max_b: float = 0.0
    gpu_mem_used_n: int = 0
    gpu_mem_total_max_b: float = 0.0

    # GPU temperature
    gpu_temp_sum: float = 0.0
    gpu_temp_max: float = 0.0
    gpu_temp_n: int = 0


def _b_to_gb(x: float) -> float:
    return x / 1e9  # GB


def _update_envelope_ts(agg: _Agg, msg: Dict[str, Any]) -> None:
    ts = msg.get("timestamp")
    if isinstance(ts, (int, float)):
        ts = float(ts)
        if agg.first_ts is None:
            agg.first_ts = ts
        agg.last_ts = ts


def _update_from_system_row(agg: _Agg, row: Dict[str, Any]) -> None:
    """
    Wire format per your schema:
      row["cpu"], row["ram_used"], row["ram_total"], row["gpu_available"], row["gpu_count"],
      row["gpus"] = [[util, mem_used, mem_total, temp, power, power_limit], ...]
    """
    agg.system_rows += 1

    cpu = row.get("cpu")
    if isinstance(cpu, (int, float)):
        c = float(cpu)
        agg.cpu_sum += c
        agg.cpu_max = max(agg.cpu_max, c)
        agg.cpu_n += 1

    ram_used = row.get("ram_used")
    if isinstance(ram_used, (int, float)):
        u = float(ram_used)
        agg.ram_used_sum_b += u
        agg.ram_used_max_b = max(agg.ram_used_max_b, u)
        agg.ram_n += 1

    ram_total = row.get("ram_total")
    if isinstance(ram_total, (int, float)):
        agg.ram_total_max_b = max(agg.ram_total_max_b, float(ram_total))

    ga = row.get("gpu_available")
    if isinstance(ga, bool):
        agg.gpu_available = ga

    gc = row.get("gpu_count")
    if isinstance(gc, int):
        agg.gpu_count = gc

    gpus = row.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        return

    for gi, g in enumerate(gpus):
        if not (isinstance(g, list) and len(g) >= 3):
            continue
        util, mem_used, mem_total = g[0], g[1], g[2]
        temp = g[3] if len(g) >= 4 else None

        if isinstance(util, (int, float)):
            u = float(util)
            agg.gpu_util_sum += u
            agg.gpu_util_max = max(agg.gpu_util_max, u)
            agg.gpu_util_n += 1

            if gi == 0:
                agg.gpu0_util_sum += u
                agg.gpu0_util_max = max(agg.gpu0_util_max, u)
                agg.gpu0_util_n += 1

        if isinstance(mem_used, (int, float)):
            mu = float(mem_used)
            agg.gpu_mem_used_sum_b += mu
            agg.gpu_mem_used_max_b = max(agg.gpu_mem_used_max_b, mu)
            agg.gpu_mem_used_n += 1

        if isinstance(mem_total, (int, float)):
            agg.gpu_mem_total_max_b = max(
                agg.gpu_mem_total_max_b, float(mem_total)
            )

        if isinstance(temp, (int, float)):
            t = float(temp)
            agg.gpu_temp_sum += t
            agg.gpu_temp_max = max(agg.gpu_temp_max, t)
            agg.gpu_temp_n += 1


def _fmt(x: Optional[float], suf: str = "", n_d: int = 1) -> str:
    return "n/a" if x is None else f"{x:.{n_d}f}{suf}"


def _make_card(agg: _Agg) -> tuple[str, Dict[str, Any]]:
    """
    Build a *shareable* runtime summary card (SYSTEM only).

    Notes
    -----
    - This function intentionally emits ONLY the SYSTEM section.
      Step-time / other cards are appended elsewhere.
    - Output is formatted with clear boundaries and spacing for easy pasting
      into Slack / GitHub / email.
    """
    duration_s = None
    if (
        agg.first_ts is not None
        and agg.last_ts is not None
        and agg.last_ts >= agg.first_ts
    ):
        duration_s = agg.last_ts - agg.first_ts

    cpu_avg = (agg.cpu_sum / agg.cpu_n) if agg.cpu_n else None
    cpu_peak = agg.cpu_max if agg.cpu_n else None

    ram_avg_gb = (
        _b_to_gb(agg.ram_used_sum_b / agg.ram_n) if agg.ram_n else None
    )
    ram_peak_gb = _b_to_gb(agg.ram_used_max_b) if agg.ram_n else None
    ram_total_gb = (
        _b_to_gb(agg.ram_total_max_b) if agg.ram_total_max_b else None
    )

    gpu_util_avg = (
        (agg.gpu_util_sum / agg.gpu_util_n) if agg.gpu_util_n else None
    )
    gpu_util_peak = agg.gpu_util_max if agg.gpu_util_n else None
    gpu0_util_avg = (
        (agg.gpu0_util_sum / agg.gpu0_util_n) if agg.gpu0_util_n else None
    )

    gpu_mem_avg_gb = (
        _b_to_gb(agg.gpu_mem_used_sum_b / agg.gpu_mem_used_n)
        if agg.gpu_mem_used_n
        else None
    )
    gpu_mem_peak_gb = (
        _b_to_gb(agg.gpu_mem_used_max_b) if agg.gpu_mem_used_n else None
    )
    gpu_mem_total_gb = (
        _b_to_gb(agg.gpu_mem_total_max_b) if agg.gpu_mem_total_max_b else None
    )

    gpu_temp_avg = (
        (agg.gpu_temp_sum / agg.gpu_temp_n) if agg.gpu_temp_n else None
    )
    gpu_temp_peak = agg.gpu_temp_max if agg.gpu_temp_n else None

    w = 72
    sep = "-" * w

    title = (
        f"TraceML Runtime Summary Card | duration {_fmt(duration_s, 's', 1)}"
        f" | samples {agg.system_rows}"
    )

    # Simple, aligned key/value rows (readable in monospace)
    rows: list[tuple[str, str]] = []
    rows.append(
        ("CPU", f"avg {_fmt(cpu_avg, '%', 1)} | peak {_fmt(cpu_peak, '%', 1)}")
    )
    rows.append(
        (
            "RAM",
            f"avg {_fmt(ram_avg_gb, 'GB', 1)} | peak {_fmt(ram_peak_gb, 'GB', 1)} | total {_fmt(ram_total_gb, 'GB', 1)}",
        )
    )

    if agg.gpu_util_n > 0:
        gpu_util = f"avg {_fmt(gpu_util_avg, '%', 1)} | peak {_fmt(gpu_util_peak, '%', 1)}"
        if agg.gpu0_util_n:
            gpu_util += f" | GPU0 avg {_fmt(gpu0_util_avg, '%', 1)}"
        rows.append(("GPU util", gpu_util))

        gpu_mem = (
            f"avg {_fmt(gpu_mem_avg_gb, 'GB', 1)} | peak {_fmt(gpu_mem_peak_gb, 'GB', 1)}"
            f" | total {_fmt(gpu_mem_total_gb, 'GB', 1)}"
        )
        if agg.gpu_temp_n:
            gpu_mem += (
                f" | temp avg {_fmt(gpu_temp_avg, '°C', 1)}"
                f" | peak {_fmt(gpu_temp_peak, '°C', 1)}"
            )
        rows.append(("GPU mem", gpu_mem))
    else:
        if agg.gpu_available is False:
            rows.append(
                (
                    "GPU",
                    "unavailable (no NVIDIA GPU detected or NVML not accessible)",
                )
            )
        elif (agg.gpu_count or 0) > 0:
            rows.append(
                ("GPU", "detected, but no per-GPU samples were recorded")
            )
        else:
            rows.append(("GPU", "n/a"))

    key_w = max(len(k) for k, _ in rows)
    system_lines = [f"{k:<{key_w}} : {v}" for k, v in rows]

    # Final card (SYSTEM only; step timing appended elsewhere)
    lines: list[str] = []
    lines.append(title)
    lines.append(sep)
    lines.append("SYSTEM")
    lines.append(sep)
    lines.extend(system_lines)
    card = "\n".join(lines)

    summary = {
        "duration_s": duration_s,
        "system_samples": agg.system_rows,
        "cpu_avg_percent": cpu_avg,
        "cpu_peak_percent": cpu_peak,
        "ram_avg_gb": ram_avg_gb,
        "ram_peak_gb": ram_peak_gb,
        "ram_total_gb": ram_total_gb,
        "gpu_available": agg.gpu_available,
        "gpu_count": agg.gpu_count,
        "gpu_util_avg_percent": gpu_util_avg,
        "gpu_util_peak_percent": gpu_util_peak,
        "gpu0_util_avg_percent": gpu0_util_avg,
        "gpu_mem_avg_gb": gpu_mem_avg_gb,
        "gpu_mem_peak_gb": gpu_mem_peak_gb,
        "gpu_mem_total_gb": gpu_mem_total_gb,
        "gpu_temp_avg_c": gpu_temp_avg,
        "gpu_temp_peak_c": gpu_temp_peak,
        "units": {"memory": "GB", "temp": "C", "util": "%"},
        "card": card,
    }
    return card, summary


def generate_system_summary_card(
    db_path: str,
    system_sampler_name: str = "SystemSampler",
    print_to_stdout: bool = True,
    max_system_rows: int = 50_000,
) -> Dict[str, Any]:
    """
    Generate a shareable SYSTEM summary card from the local TraceML DB.

    Parameters
    ----------
    db_path:
        Path to the SQLite DB file.
    system_sampler_name:
        Sampler name stored in `raw_messages.sampler` for system telemetry.
    print_to_stdout:
        If True, prints the generated card to stdout.
    max_system_rows:
        Safety cap on the number of system rows aggregated (prevents very large
        DB scans from taking too long). Default: 50_000.

    Returns
    -------
    Dict[str, Any]
        Parsed summary JSON including the rendered `card` string.
    """
    conn = sqlite3.connect(db_path)
    dec = msgspec.msgpack.Decoder(type=dict)
    agg = _Agg()

    cur = conn.execute(
        "SELECT payload_mp FROM raw_messages WHERE sampler = ? ORDER BY id ASC;",
        (system_sampler_name,),
    )

    for (blob,) in cur:
        if not blob:
            continue
        try:
            msg = dec.decode(blob)
        except Exception:
            continue

        _update_envelope_ts(agg, msg)

        tables = msg.get("tables")
        if not isinstance(tables, dict):
            continue

        for rows in tables.values():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue

                _update_from_system_row(agg, row)

                # Safety cap: stop once we've aggregated enough rows.
                if agg.system_rows >= max_system_rows:
                    break

            if agg.system_rows >= max_system_rows:
                break

        if agg.system_rows >= max_system_rows:
            break

    conn.close()

    card, summary = _make_card(agg)

    with open(db_path + "_summary_card.txt", "w", encoding="utf-8") as f:
        f.write(card + "\n")

    with open(db_path + "_summary_card.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if print_to_stdout:
        print(card)

    return summary
