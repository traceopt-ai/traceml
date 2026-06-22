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
