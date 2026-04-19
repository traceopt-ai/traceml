"""
Shared formatting and numeric helpers for end-of-run summaries.

This module intentionally contains only small generic helpers that are reused
across multiple summary builders. Domain-specific logic should stay in the
relevant summary module.
"""

from typing import Any, Optional


def safe_float(x: Any) -> float:
    """
    Best-effort float conversion; return 0.0 on failure.
    """
    try:
        return float(x)
    except Exception:
        return 0.0


def bytes_to_gb(x: Optional[float]) -> Optional[float]:
    """
    Convert bytes to decimal gigabytes for display and summary JSON.
    """
    if x is None:
        return None
    return float(x) / 1e9


def format_optional(
    x: Optional[float],
    suffix: str = "",
    ndigits: int = 1,
) -> str:
    """
    Format an optional numeric value for human-readable output.
    """
    return "n/a" if x is None else f"{x:.{ndigits}f}{suffix}"


def format_ms(x: Optional[float]) -> str:
    """
    Format milliseconds for human-readable output.
    """
    return "n/a" if x is None else f"{x:.1f}ms"


def format_percent(x: Optional[float]) -> str:
    """
    Format a percentage value such as 12.3 as '12.3%'.
    """
    return "n/a" if x is None else f"{x:.1f}%"


def format_ratio_percent(x: Optional[float]) -> str:
    """
    Format a ratio value such as 0.123 as '12.3%'.
    """
    if x is None:
        return "n/a"
    return f"{float(x) * 100.0:.1f}%"


def share_percent(
    num: Optional[float],
    denom: Optional[float],
) -> Optional[float]:
    """
    Return percentage share num / denom, or None if denom is not positive.
    """
    if num is None or denom is None or denom <= 0.0:
        return None
    return 100.0 * num / denom


def duration_from_bounds(
    first_ts: Optional[float],
    last_ts: Optional[float],
) -> Optional[float]:
    """
    Return duration in seconds if timestamps are valid.
    """
    if first_ts is None or last_ts is None or last_ts < first_ts:
        return None
    return last_ts - first_ts
