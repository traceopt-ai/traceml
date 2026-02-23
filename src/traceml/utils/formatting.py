from typing import Any, Tuple


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
    """
    Format a time duration given in milliseconds.
    """
    if v <= 0:
        return "—"

    # microseconds
    if v < 1.0:
        return f"{v * 1000:.1f} µs"

    # milliseconds
    if v < 1000.0:
        return f"{v:.2f} ms"

    # seconds
    sec = v / 1000.0
    if sec < 60.0:
        return f"{sec:.2f} s"

    # minutes
    minutes = sec / 60.0
    return f"{minutes:.2f} min"


_MEMORY_UNITS = ["B", "KB", "MB", "GB", "TB"]

def format_memory(num_bytes: Any) -> Tuple[float, str]:
    """
    Convert bytes to (value, unit) using binary units (1024 base).
    No rounding applied.
    """
    try:
        v = float(num_bytes)
    except (TypeError, ValueError):
        return 0.0, "B"


    idx = 0

    while v >= 1024 and idx < len(_MEMORY_UNITS) - 1:
        v /= 1024.0
        idx += 1

    return v, _MEMORY_UNITS[idx]


def fmt_mem_ratio(used: Any, total: Any) -> str:
    try:
        used_b = float(used)
        total_b = float(total)
    except (TypeError, ValueError):
        return "N/A"

    if total_b <= 0:
        return "N/A"

    # Scale using total (so both share same unit)
    total_val, unit = format_memory(total_b)
    used_val = used_b / (1024 ** _MEMORY_UNITS.index(unit))

    return f"{used_val:.1f}/{total_val:.1f} {unit}"


def fmt_mem_triple(used: Any, reserved: Any, total: Any) -> str:
    try:
        used_b = float(used)
        reserved_b = float(reserved)
        total_b = float(total)
    except (TypeError, ValueError):
        return "N/A"

    if total_b <= 0:
        return "N/A"

    total_val, unit = format_memory(total_b)
    try:
        scale = 1024 ** _MEMORY_UNITS.index(unit)
    except ValueError:
        # fallback if format_memory returns an unexpected unit
        scale = 1.0
        unit = "B"
        total_val = total_b

    used_val = used_b / scale
    reserved_val = reserved_b / scale

    return f"{used_val:.1f}/{reserved_val:.1f}/{total_val:.1f} {unit}"