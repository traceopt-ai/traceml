from typing import Any


def fmt_percent(x):
    try:
        return f"{float(x):.1f}%"
    except Exception:
        return "N/A"


def fmt_ratio(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "N/A"


def fmt_mem_new(num_bytes: Any) -> str:
    """
    Format a byte value into a human-friendly string (KB, MB, GB).
    Always uses binary units (1 KB = 1024 B).
    """
    try:
        v = float(num_bytes)
    except (TypeError, ValueError):
        return "N/A"

    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while v >= 1024 and idx < len(units) - 1:
        v /= 1024.0
        idx += 1

    if v >= 100 or idx == 0:  # e.g. 123 MB → "123 MB"
        return f"{v:.0f} {units[idx]}"
    elif v >= 10:
        return f"{v:.1f} {units[idx]}"
    else:
        return f"{v:.2f} {units[idx]}"



def fmt_time_ms(v: float) -> str:
    if v <= 0:
        return "—"
    if v < 1.0:
        return f"{v * 1000:.1f} µs"
    return f"{v:.2f} ms"