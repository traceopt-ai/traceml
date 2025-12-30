import numpy as np
from rich.panel import Panel
from rich.table import Table
import shutil

from traceml.renderers.base_renderer import BaseRenderer
from traceml.database.database import Database
from traceml.renderers.display.cli_display_manager import MODEL_COMBINED_LAYOUT
from traceml.renderers.utils import fmt_time_run, fmt_mem_new


class ModelCombinedRenderer(BaseRenderer):
    """
    Renderer for TraceML step-level summary:
      - internal step timers
      - step-level peak GPU memory
    """

    FRIENDLY_NAMES = {
        "_traceml_internal:dataloader_next": "dataLoader_fetch_time",
        "_traceml_internal:step_time": "step_time",
    }

    def __init__(
        self,
        time_db: Database,
        memory_db: Database,
        window: int = 100,
    ):
        super().__init__(
            name="Model Summary",
            layout_section_name=MODEL_COMBINED_LAYOUT,
        )
        self.time_db = time_db
        self.memory_db = memory_db
        self.window = int(window)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_percentile(x: np.ndarray, q: float) -> float:
        if x.size == 0:
            return 0.0
        return float(np.percentile(x, q))

    def _compute_stats(self, arr: np.ndarray):
        if arr.size == 0:
            return 0.0, 0.0, 0.0, 0.0, ""

        last = float(arr[-1])
        win100 = arr[-min(100, arr.size):]
        win200 = arr[-min(200, arr.size):]

        p50 = self._safe_percentile(win100, 50)
        p95 = self._safe_percentile(win100, 95)
        avg100 = float(win100.mean())

        trend = ""
        if arr.size >= 200:
            avg200 = float(win200.mean())
            trend = "+" if avg100 > avg200 else "-"

        return last, p50, p95, avg100, trend

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _get_step_time_data(self):
        cpu_table = self.time_db.create_or_get_table("step_timer_cpu")
        gpu_tables = {
            name: rows
            for name, rows in self.time_db.all_tables().items()
            if name.startswith("step_timer_cuda")
        }

        data = {}

        for row in cpu_table:
            name = row.get("event_name")
            if name not in self.FRIENDLY_NAMES:
                continue
            data.setdefault(name, {"cpu": [], "gpu": []})
            data[name]["cpu"].append(float(row.get("duration_ms", 0.0)))

        for rows in gpu_tables.values():
            for row in rows:
                name = row.get("event_name")
                if name not in self.FRIENDLY_NAMES:
                    continue
                data.setdefault(name, {"cpu": [], "gpu": []})
                data[name]["gpu"].append(float(row.get("duration_ms", 0.0)))

        return data

    def _get_step_memory_data(self):
        table = self.memory_db.create_or_get_table("step_memory")
        gpu_vals = []

        for row in table:
            gpu_vals.append(float(row.get("peak_allocated_mb", 0.0)))

        return np.asarray(gpu_vals, dtype=np.float64)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def get_panel_renderable(self) -> Panel:
        table = Table(show_header=True, header_style="bold blue", box=None)
        table.add_column("Metric", justify="left", style="cyan")
        table.add_column("Last", justify="right")
        table.add_column("p50(100)", justify="right")
        table.add_column("p95(100)", justify="right")
        table.add_column("Avg(100)", justify="right")
        table.add_column("Trend", justify="center")
        table.add_column("Device", justify="center", style="magenta")

        # ---- Step timers ----
        time_data = self._get_step_time_data()

        for key in self.FRIENDLY_NAMES.keys():
            vals = time_data.get(key, {"cpu": [], "gpu": []})
            arr = (
                np.asarray(vals["gpu"], dtype=np.float64)
                if vals["gpu"]
                else np.asarray(vals["cpu"], dtype=np.float64)
            )
            device = "GPU" if vals["gpu"] else "CPU"

            last, p50, p95, avg100, trend = self._compute_stats(arr)

            table.add_row(
                self.FRIENDLY_NAMES[key],
                fmt_time_run(last),
                fmt_time_run(p50),
                fmt_time_run(p95),
                fmt_time_run(avg100),
                trend,
                device,
            )

        table.add_row("")
        # ---- Step memory ----
        mem_arr = self._get_step_memory_data()
        last, p50, p95, avg100, trend = self._compute_stats(mem_arr)

        table.add_row(
            "step_memory",
            fmt_mem_new(last),
            fmt_mem_new(p50),
            fmt_mem_new(p95),
            fmt_mem_new(avg100),
            trend,
            "GPU",
        )

        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(90, int(cols * 0.65)), 100)

        return Panel(
            table,
            title="[bold blue]Model Summary[/bold blue]",
            border_style="blue",
            width=panel_width,
        )
