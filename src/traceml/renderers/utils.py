from typing import Dict
from traceml.utils.formatting import fmt_mem_new
from pathlib import Path


CARD_STYLE = """
    border:2px solid #00bcd4;
    border-radius:10px;
    padding:14px;
    width:100%;
    min-height:190px;
    display:flex;
    flex-direction:column;
    justify-content:space-between;
    font-family:Arial, sans-serif;
"""

def truncate_layer_name(s: str, max_len: int = 20) -> str:
    """
    Truncate layer name by keeping the last max_len characters.
    Optimized for layer names where the suffix is most informative.
    """
    if not isinstance(s, str):
        s = str(s)
    if len(s) <= max_len:
        return s

    return "…" + s[-(max_len - 1) :]


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


def fmt_time_run(ms: float) -> str:
    """
    Format run / step-level durations given in milliseconds.
    """
    if ms <= 0:
        return "—"

    if ms < 1000.0:
        return f"{ms:.1f} ms"

    seconds = ms / 1000.0
    if seconds < 60.0:
        return f"{seconds:.2f} s"

    minutes = seconds / 60.0
    if minutes < 60.0:
        return f"{minutes:.2f} min"

    hours = minutes / 60.0
    return f"{hours:.2f} h"


def append_text(path: str, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
