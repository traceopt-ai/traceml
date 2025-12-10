import time

def level_bar_continuous(pct: float) -> str:
    pct = max(0, min(pct, 100))

    # Color rule
    color=choose_color(pct)    # green

    return f"""
    <div style="
        width: 140px;
        height: 10px;
        background: rgba(0,0,0,0.20);
        border-radius: 5px;
        overflow: hidden;
        display: inline-block;
        vertical-align: middle;
    ">
        <div style="
            width: {pct}%;
            height: 100%;
            background: {color};
            border-radius: 5px;
            transition: width 0.2s ease;
        "></div>
    </div>
    """


def choose_color(pct: float) -> str:
    """Return color based on standard dashboard thresholds."""

    if pct >= 80:
        return "#D32F2F"  # red - danger
    elif pct >= 50:
        return "#FFB300"  # orange - mid load
    else:
        return "#4CAF50"  # green - good


def extract_time_axis(table):
    t_raw = [rec.get("timestamp") for rec in table][-100:]
    x_hist = [time.strftime("%H:%M:%S", time.localtime(t)) for t in t_raw]
    return x_hist

def build_fake_section():
    return {}

def update_fake_section():
    return