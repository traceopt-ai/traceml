import numpy as np
from rich.panel import Panel
from rich.table import Table
import shutil

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
        "_traceml_internal:dataloader_next": "dataLoader_fetch_time",
        "_traceml_internal:step_time": "step_time",
    }

    def __init__(self, database: Database, window: int = 100):
        super().__init__(
            name="Model Summary",
            layout_section_name=MODEL_COMBINED_LAYOUT,
        )
        self.db = database
        self.window = int(window)

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

    @staticmethod
    def _safe_percentile(x: np.ndarray, q: float) -> float:
        # np.percentile throws on empty arrays
        if x.size == 0:
            return 0.0
        return float(np.percentile(x, q))

    def get_panel_renderable(self) -> Panel:
        data = self.get_data()

        table = Table(show_header=True, header_style="bold blue", box=None)
        table.add_column("Metric", justify="left", style="cyan")
        table.add_column("Last", justify="right")
        table.add_column("p50(100)", justify="right")
        table.add_column("p95(100)", justify="right")
        table.add_column("Avg(100)", justify="right")
        table.add_column("Trend", justify="center")
        table.add_column("Device", justify="center", style="magenta")

        for key in self.FRIENDLY_NAMES.keys():
            vals = data.get(key, {"cpu": [], "gpu": []})
            gpu_vals = vals["gpu"]
            cpu_vals = vals["cpu"]

            if gpu_vals:
                arr = np.asarray(gpu_vals, dtype=np.float64)
                device = "GPU"
            else:
                arr = np.asarray(cpu_vals, dtype=np.float64)
                device = "CPU"

            if arr.size == 0:
                last = p50 = p95 = avg100 = 0.0
                trend = ""
            else:
                last = float(arr[-1])

                win100 = arr[-min(100, arr.size):]
                win200 = arr[-min(200, arr.size):]

                p50 = self._safe_percentile(win100, 50)
                p95 = self._safe_percentile(win100, 95)
                avg100 = float(win100.mean())

                # Trend: sign only, shown only if enough data
                if arr.size >= 200:
                    avg200 = float(win200.mean())
                    if avg200 > 0:
                        delta = (avg100 - avg200) / avg200
                        trend = "+" if delta > 0 else "-"
                    else:
                        trend = ""
                else:
                    trend = ""

            table.add_row(
                self.FRIENDLY_NAMES[key],
                fmt_time_run(last),
                fmt_time_run(p50),
                fmt_time_run(p95),
                fmt_time_run(avg100),
                trend,
                device,
            )

        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(90, int(cols * 0.65)), 100)

        return Panel(
            table,
            title="[bold blue]Model Summary[/bold blue]",
            border_style="blue",
            width=panel_width,
        )