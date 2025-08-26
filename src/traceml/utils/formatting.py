from typing import Any

def fmt_mem(mb: Any) -> str:
    try:
        v = float(mb)
        if v >= 1024:
            return f"{v / 1024:.2f} GB"
        return f"{v:.2f} MB"
    except Exception:
        return "N/A"


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