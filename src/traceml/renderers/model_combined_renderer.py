import numpy as np
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
import shutil
from typing import Optional
from IPython.display import HTML

from traceml.renderers.base_renderer import BaseRenderer
from traceml.database.database import Database
from traceml.renderers.display.cli_display_manager import MODEL_COMBINED_LAYOUT
from traceml.renderers.utils import fmt_time_run, fmt_mem_new
from .utils import CARD_STYLE

class ModelCombinedRenderer(BaseRenderer):
    """
    Renderer for TraceML step-level summary:
      - internal step timers
      - step-level peak GPU memory
    """

    FRIENDLY_NAMES = {
        "_traceml_internal:dataloader_next": "dataLoader_fetch",
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
        self.current_step: Optional[int] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_current_step(self) -> Optional[int]:
        table = self.memory_db.create_or_get_table("step_memory")
        if not table:
            return None
        return int(table[-1].get("step"))

    @staticmethod
    def _safe_percentile(x: np.ndarray, q: float) -> float:
        if x.size == 0:
            return 0.0
        return float(np.percentile(x, q))

    def _compute_stats(self, arr: np.ndarray):
        if arr.size == 0:
            return 0.0, 0.0, 0.0, 0.0, ""

        last = float(arr[-1])
        win100 = arr[-min(100, arr.size) :]
        win200 = arr[-min(200, arr.size) :]

        p50 = self._safe_percentile(win100, 50)
        p95 = self._safe_percentile(win100, 95)
        avg100 = float(win100.mean())

        trend = ""
        trend_pct = 0.0

        if arr.size >= 200:
            avg200 = float(win200.mean())

            # Guard against divide-by-zero / near-zero noise
            if avg200 > 1e-9:
                trend_pct = (avg100 - avg200) / avg200 * 100.0

                # Ignore tiny fluctuations (< 1%)
                if abs(trend_pct) >= 1.0:
                    sign = "+" if trend_pct > 0 else ""
                    trend = f"{sign}{trend_pct:.1f}%"
                else:
                    trend = "≈0%"

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
            data[name]["cpu"].append(
                (int(row.get("step")), float(row.get("duration_ms", 0.0)))
            )

        for rows in gpu_tables.values():
            for row in rows:
                name = row.get("event_name")
                if name not in self.FRIENDLY_NAMES:
                    continue
                data.setdefault(name, {"cpu": [], "gpu": []})
                data[name]["gpu"].append(
                    (int(row.get("step")), float(row.get("duration_ms", 0.0)))
                )

        return data

    def build_live_telemetry_payload(self, last_n: int = 200):
        """
        Build UI-ready telemetry payload for the WebUI.
        No DB objects are exposed.
        """

        payload = {}

        # -------- Step timers --------
        time_data = self._get_step_time_data()

        for key in self.FRIENDLY_NAMES.keys():
            vals = time_data.get(key, {"cpu": [], "gpu": []})
            pairs = vals["gpu"] if vals["gpu"] else vals["cpu"]
            pairs = pairs[-last_n:]

            if len(pairs) > 0:
                steps, values = zip(*pairs)
                x = np.asarray(steps, dtype=np.int64)
                y = np.asarray(values, dtype=np.float64)
            else:
                x = np.asarray([])
                y = np.asarray([])

            stats = self._compute_stats(y)

            payload[self.FRIENDLY_NAMES[key]] = {
                "x": x,
                "y": y,
                "stats": {
                    "last": stats[0],
                    "p50": stats[1],
                    "p95": stats[2],
                    "avg100": stats[3],
                    "trend": stats[4],
                },
            }

        # -------- Step memory --------
        pairs = self._get_step_memory_data()
        pairs = pairs[-last_n:]

        if len(pairs) > 0:
            steps, values = zip(*pairs)
            x = np.asarray(steps, dtype=np.int64)
            y = np.asarray(values, dtype=np.float64)
        else:
            x = np.asarray([])
            y = np.asarray([])

        stats = self._compute_stats(y)

        payload["step_gpu_memory"] = {
            "x": x,
            "y": y,
            "stats": {
                "last": stats[0],
                "p50": stats[1],
                "p95": stats[2],
                "avg100": stats[3],
                "trend": stats[4],
            },
        }

        return payload

    def _get_step_memory_data(self):
        table = self.memory_db.create_or_get_table("step_memory")
        gpu_vals = []

        for row in table:
            gpu_vals.append(
                (int(row.get("step")), float(row.get("peak_allocated_mb", 0.0)))
            )

        return np.asarray(gpu_vals, dtype=np.float64)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _extract_values(self, pairs):
        if not pairs:
            return np.asarray([], dtype=np.float64)
        return np.asarray([v for _, v in pairs], dtype=np.float64)

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
        current_step = self._get_current_step()

        for key in self.FRIENDLY_NAMES.keys():
            vals = time_data.get(key, {"cpu": [], "gpu": []})
            pairs = vals["gpu"] if vals["gpu"] else vals["cpu"]
            arr = self._extract_values(pairs)
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
            "step_gpu_memory",
            fmt_mem_new(last),
            fmt_mem_new(p50),
            fmt_mem_new(p95),
            fmt_mem_new(avg100),
            trend,
            "GPU",
        )

        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(90, int(cols * 0.65)), 100)

        title = "[bold blue]Model Summary[/bold blue]"
        if current_step is not None:
            title += f" (Step {current_step})"

        return Panel(
            table,
            title=title,
            border_style="blue",
            width=panel_width,
        )

    def get_dashboard_renderable(self):
        """
        Return an object suitable for Streamlit's st.write():
        - str / Markdown
        - pandas.DataFrame
        - matplotlib/plotly figure
        - etc.
        """
        data = self.build_live_telemetry_payload()
        return data

    def get_notebook_renderable(self) -> HTML:
        telemetry = self.build_live_telemetry_payload()

        def metric_block(title, stats, fmt):
            trend = stats.get("trend", "")

            # Defaults
            trend_text = "—"
            trend_color = "#666"

            if isinstance(trend, str) and trend:
                if trend.startswith("+"):
                    trend_text = f"↑ {trend}"
                    trend_color = "#d32f2f"  # red (regression)
                elif trend.startswith("-"):
                    trend_text = f"↓ {trend}"
                    trend_color = "#2e7d32"  # green (improvement)
                elif "≈" in trend:
                    trend_text = trend
                    trend_color = "#666"

            return f"""
            <div style="
                flex:1;
                border:1px solid #eee;
                border-radius:8px;
                padding:10px;
                font-size:13px;
            ">
                <div style="font-weight:700; margin-bottom:6px;">
                    {title}
                </div>

                <table style="width:100%; border-collapse:collapse;">
                    <thead>
                        <tr style="border-bottom:1px solid #ddd;">
                            <th align="left">Last</th>
                            <th align="right">p50</th>
                            <th align="right">p95</th>
                            <th align="right">Avg</th>
                            <th align="center">Trend</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>{fmt(stats["last"])}</td>
                            <td align="right">{fmt(stats["p50"])}</td>
                            <td align="right">{fmt(stats["p95"])}</td>
                            <td align="right">{fmt(stats["avg100"])}</td>
                            <td align="center"
                                style="font-weight:700; color:{trend_color};">
                                {trend_text}
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """

        blocks = [
            metric_block(
                "Dataloader Fetch Time",
                telemetry["dataLoader_fetch"]["stats"],
                fmt_time_run,
            ),
            metric_block(
                "Training Step Time",
                telemetry["step_time"]["stats"],
                fmt_time_run,
            ),
            metric_block(
                "GPU Step Memory",
                telemetry["step_gpu_memory"]["stats"],
                fmt_mem_new,
            ),
        ]

        html = f"""
        <div style="{CARD_STYLE}; width:100%;">
            <h4 style="color:#d47a00; margin:0 0 12px 0;">
                Model Summary
            </h4>

            <div style="
                display:flex;
                gap:12px;
                align-items:stretch;
            ">
                {''.join(blocks)}
            </div>
        </div>
        """

        return HTML(html)


    def log_summary(self, path: Optional[str] = None) -> None:
        Console().print(self.get_panel_renderable())