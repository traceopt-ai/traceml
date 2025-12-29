
import numpy as np
from rich.panel import Panel
from rich.table import Table

from traceml.renderers.base_renderer import BaseRenderer
from traceml.database.database import Database
from traceml.renderers.display.cli_display_manager import MODEL_COMBINED_LAYOUT
from traceml.renderers.utils import fmt_time_run




class ModelCombinedRenderer(BaseRenderer):
    """
    Renderer for TraceML internal step timers.
    Shows approximate timings for core runtime signals.
    """

    FRIENDLY_NAMES = {
        "_traceml_internal:dataloader_next": "(~) DataLoader fetch time",
        "_traceml_internal:step_time": "(~) Step time",
    }

    def __init__(self, database: Database):
        super().__init__(
            name="Runtime Summary",
            layout_section_name=MODEL_COMBINED_LAYOUT,
        )
        self.db = database

    def get_data(self):
        cpu_table = self.db.create_or_get_table("step_timer_cpu")
        gpu_tables = {
            name: rows
            for name, rows in self.db.all_tables().items()
            if name.startswith("step_timer_cuda")
        }

        data = {}

        # CPU
        for row in cpu_table:
            name = row.get("event_name")
            if name not in self.FRIENDLY_NAMES:
                continue

            dur = float(row.get("duration_ms", 0.0))
            data.setdefault(name, {"cpu": [], "gpu": []})
            data[name]["cpu"].append(dur)

        # GPU
        for rows in gpu_tables.values():
            for row in rows:
                name = row.get("event_name")
                if name not in self.FRIENDLY_NAMES:
                    continue

                dur = float(row.get("duration_ms", 0.0))
                data.setdefault(name, {"cpu": [], "gpu": []})
                data[name]["gpu"].append(dur)

        return data

    def get_panel_renderable(self) -> Panel:
        data = self.get_data()

        table = Table(show_header=True, header_style="bold blue", box=None)
        table.add_column("Metric", justify="left", style="cyan")
        table.add_column("Avg (s)", justify="right")
        table.add_column("Peak (s)", justify="right")
        table.add_column("Device", justify="center", style="magenta")

        for key, vals in data.items():
            gpu_vals = vals["gpu"]
            cpu_vals = vals["cpu"]

            if gpu_vals:
                arr = np.array(gpu_vals)
                avg, peak = arr.mean(), arr.max()
                device = "GPU"
            else:
                arr = np.array(cpu_vals)
                avg, peak = arr.mean(), arr.max()
                device = "CPU"

            table.add_row(
                self.FRIENDLY_NAMES[key],
                fmt_time_run(avg),
                fmt_time_run(peak),
                device,
            )

        return Panel(
            table,
            title="[bold blue]Runtime Summary[/bold blue]",
            border_style="blue",
        )
