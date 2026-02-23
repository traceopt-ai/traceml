import time

def choose_color(pct: float) -> str:
    """Return color based on standard dashboard thresholds."""

    if pct >= 80:
        return "#D32F2F"  # red - danger
    elif pct >= 50:
        return "#FFB300"  # orange - mid load
    else:
        return "#4CAF50"  # green - good


def extract_time_axis(table, key="timestamp"):
    t_raw = [rec.get(key) for rec in table][-100:]
    x_hist = [time.strftime("%H:%M:%S", time.localtime(t)) for t in t_raw]
    return x_hist


def extract_x_axis(table, key="seq"):
    x_hist = [rec.get(key) for rec in table][-100:]
    return x_hist


def build_fake_section():
    return {}


def update_fake_section():
    return
