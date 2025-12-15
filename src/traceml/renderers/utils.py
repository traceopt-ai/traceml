from typing import Dict
from traceml.utils.formatting import fmt_mem_new


def truncate_layer_name(s: str, max_len: int = 40) -> str:
    """
    Truncate long layer names keeping start and end.
    """
    if not isinstance(s, str):
        s = str(s)
    if len(s) <= max_len:
        return s
    half = (max_len - 1) // 2
    return s[:half] + "…" + s[-half:]


def format_cache_value(
    cache: Dict[str, Dict[str, float]],
    layer: str,
) -> str:
    """
    Return 'curr / global' string for a given layer from a cache; '—' if unknown.
    """
    rec = cache.get(layer)
    if not rec:
        return "—"
    return f"{fmt_mem_new(rec.get('current', 0.0))} / {fmt_mem_new(rec.get('global', 0.0))}"


def fmt_time_ms(v: float) -> str:
    if v <= 0:
        return "—"
    if v < 1.0:
        return f"{v * 1000:.1f} µs"
    return f"{v:.2f} ms"