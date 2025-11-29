from nicegui import ui
from traceml.utils.formatting import fmt_mem_new


def build_system_section():
    """
    Creates the System Metrics card UI and returns a dict
    containing all labels that will be updated later.
    """

    with ui.card().classes("m-2 p-4 w-full"):
        ui.label("System Metrics").classes("text-xl font-bold text-cyan-400 mb-2")

        # Horizontal row: CPU | RAM | GPU
        with ui.row().classes("items-center justify-between w-full gap-8"):
            cpu_label = ui.label("CPU: –").classes("text-lg")
            ram_label = ui.label("RAM: –").classes("text-lg")
            gpu_label = ui.label("GPU: –").classes("text-lg")

    # return references so the update loop can update these labels
    return {
        "cpu": cpu_label,
        "ram": ram_label,
        "gpu": gpu_label,
    }


def update_system_section(panel, data):
    """
    Updates the System section labels using the data dict returned
    by the renderer's get_dashboard_renderable().

    panel = dict with label references
    data  = dict of numeric values
    """

    # CPU
    panel["cpu"].text = f"CPU: {data['cpu']:.1f}%"

    # RAM (with %)
    ru, rt = data["ram_used"], data["ram_total"]
    if rt:
        pct = (ru * 100.0) / rt
        panel["ram"].text = f"RAM: {fmt_mem_new(ru)} / {fmt_mem_new(rt)} ({pct:.1f}%)"
    else:
        panel["ram"].text = "RAM: –"

    # GPU
    if not data["gpu_available"]:
        panel["gpu"].text = "GPU: Not available"
        return

    panel["gpu"].text = (
        f"GPU Utility | Memory: {data['gpu_util_total']:.1f}% | "
        f"{fmt_mem_new(data['gpu_mem_used'])} / "
        f"{fmt_mem_new(data['gpu_mem_total'])}"
    )
