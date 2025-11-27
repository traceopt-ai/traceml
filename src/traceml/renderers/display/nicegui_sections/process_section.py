from nicegui import ui
from traceml.utils.formatting import fmt_mem_new


def build_process_section():
    """
    Builds a horizontal row card for process metrics:
    CPU  |  RAM  |  GPU (used/reserved/total)
    Returns a dict of label references for updating.
    """

    with ui.card().classes("m-2 p-4 w-full"):
        ui.label("Process Metrics").classes("text-xl font-bold text-blue-400 mb-2")

        # Horizontal row layout
        with ui.row().classes("items-center justify-between w-full gap-8"):
            cpu_label = ui.label("CPU: –").classes("text-lg")
            ram_label = ui.label("RAM: –").classes("text-lg")
            gpu_label = ui.label("GPU: –").classes("text-lg")

    return {
        "cpu": cpu_label,
        "ram": ram_label,
        "gpu": gpu_label,
    }


def update_process_section(panel, data):
    """
    Updates the Process Metrics panel with the dict from ProcessRenderer.get_dashboard_renderable().

    panel = dict of NiceGUI label references
    data  = dict: {
        cpu_used,
        ram_used,
        gpu_used,
        gpu_reserved,
        gpu_total
    }
    """

    # CPU
    cpu = data["cpu_used"]
    panel["cpu"].text = f"CPU: {cpu:.1f}%"

    # RAM
    ram_used = data["ram_used"]
    panel["ram"].text = f"RAM: {fmt_mem_new(ram_used)}"

    # GPU
    used = data["gpu_used"]
    reserved = data["gpu_reserved"]
    total = data["gpu_total"]

    if used is None or total is None:
        panel["gpu"].text = "GPU: Not available"
        return

    panel["gpu"].text = (
        f"GPU: {fmt_mem_new(used)} used / "
        f"{fmt_mem_new(reserved)} reserved / "
        f"{fmt_mem_new(total)} total"
    )
