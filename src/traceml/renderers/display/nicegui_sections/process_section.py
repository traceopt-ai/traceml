from nicegui import ui
from traceml.utils.formatting import fmt_mem_new
from traceml.renderers.display.nicegui_sections.helper import level_bar_continuous


def build_process_section():

    card = ui.card().classes("m-2 p-4 w-full")
    card.style("""
        background: rgba(245, 245, 245, 0.35);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    """)

    with card:

        ui.label("Process Metrics") \
            .classes("text-xl font-bold mb-3") \
            .style("color:#ff9800;")

        # -------- CPU Row --------
        with ui.row().classes("items-center justify-between w-full"):
            cpu_text = ui.html("CPU: –", sanitize=False).classes("text-lg").style("color:#333")
            cpu_bar  = ui.html("", sanitize=False)

        # -------- RAM Row --------
        with ui.row().classes("items-center justify-between w-full"):
            ram_text = ui.html("RAM: –", sanitize=False).classes("text-lg").style("color:#333")
            ram_bar  = ui.html("", sanitize=False)

        # -------- GPU Row --------
        with ui.row().classes("items-center justify-between w-full"):
            gpu_text = ui.html("GPU: –", sanitize=False).classes("text-lg").style("color:#333")
            gpu_bar  = ui.html("", sanitize=False)

    return {
        "cpu_text": cpu_text, "cpu_bar": cpu_bar,
        "ram_text": ram_text, "ram_bar": ram_bar,
        "gpu_text": gpu_text, "gpu_bar": gpu_bar,
    }


def update_process_section(panel, data):

    # ---------------- CPU ----------------
    cpu = data["cpu_used"]
    cores = data["cpu_logical_core_count"]

    panel["cpu_text"].content = (
        f"⚙️ CPU ({cores} cores): {cpu:.1f}%"
    )
    panel["cpu_bar"].content = level_bar_continuous(cpu/cores)

    # ---------------- RAM ----------------
    ru, rt = data["ram_used"], data["ram_total"]
    if rt:
        pct = (ru * 100.0) / rt
        panel["ram_text"].content = (
            f"💾 RAM: {fmt_mem_new(ru)} / {fmt_mem_new(rt)} ({pct:.1f}%)"
        )
        panel["ram_bar"].content = level_bar_continuous(pct)
    else:
        panel["ram_text"].content = "💾 RAM: –"
        panel["ram_bar"].content = ""

    # ---------------- GPU ----------------
    used = data["gpu_used"]
    reserved = data["gpu_reserved"]
    total = data["gpu_total"]

    if used is None or total is None:
        panel["gpu_text"].content = "🎮 GPU: Not available"
        panel["gpu_bar"].content = ""
        return

    used_pct = (used * 100.0) / total
    panel["gpu_text"].content = (
        f"🎮 GPU: {fmt_mem_new(used)} used / "
        f"{fmt_mem_new(reserved)} reserved / "
        f"{fmt_mem_new(total)} total"
    )
    panel["gpu_bar"].content = level_bar_continuous(used_pct)
