import shutil
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from collections import defaultdict
from IPython.display import HTML
from rich.panel import Panel
from rich.table import Table
from rich.console import Console

from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.display.cli_display_manager import MODEL_COMBINED_LAYOUT
from traceml.renderers.utils import fmt_time_run, fmt_mem_new
from traceml.database.database import Database
from traceml.database.remote_database_store import RemoteDBStore
from traceml.distributed import get_ddp_info
from traceml.loggers.error_log import get_error_logger
from .utils import CARD_STYLE


# Utilities (deque-safe, bounded)
def _tail_rows(
    table,
    step_key: str,
    value_key: str,
    limit: int,
) -> List[Tuple[int, float]]:
    """
    Read at most `limit` rows from the *tail* of a deque-backed table.
    """
    if not table:
        return []

    out: List[Tuple[int, float]] = []
    for r in reversed(table):
        try:
            out.append((int(r[step_key]), float(r[value_key])))
        except Exception:
            continue
        if len(out) >= limit:
            break

    out.reverse()
    return out


def _safe_percentile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def _compute_stats(arr: np.ndarray) -> Dict[str, Any]:
    """
    Compute last / p50 / p95 / avg100 / trend on a 1D array.
    """
    if arr.size == 0:
        return dict(last=0.0, p50=0.0, p95=0.0, avg100=0.0, trend="")

    last = float(arr[-1])
    win100 = arr[-min(100, arr.size):]
    win200 = arr[-min(200, arr.size):]

    p50 = _safe_percentile(win100, 50)
    p95 = _safe_percentile(win100, 95)
    avg100 = float(win100.mean())

    trend = ""

    if arr.size >= 200:
        prev = float(win200[:100].mean())
        if prev > 1e-9:
            pct = (avg100 - prev) / prev * 100.0
            if abs(pct) >= 1.0:
                trend = f"{pct:+.1f}%"
            else:
                trend = "≈0%"

    return dict(
        last=last,
        p50=p50,
        p95=p95,
        avg100=avg100,
        trend=trend,
    )



class ModelCombinedRenderer(BaseRenderer):
    """
    Step-level model summary renderer.

    Shows:
      - Dataloader fetch time
      - Training step time
      - Step GPU memory
      - Rank skew (abs + pct)
      - Slowest rank (latest completed step)

    Only shows *latest fully completed step* across all ranks
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
        remote_store: Optional[RemoteDBStore] = None,
        max_points: int = 300,
    ):
        super().__init__(
            name="Model Summary",
            layout_section_name=MODEL_COMBINED_LAYOUT,
        )

        self.time_db = time_db
        self.memory_db = memory_db
        self.remote_store = remote_store
        self.window = int(window)
        self.max_points = int(max_points)

        # DDP info
        self.is_ddp, _, self.world_size = get_ddp_info()

        # ---- local render cache ----
        self._cached_completed_step: Optional[int] = None
        self._cached_payload: Optional[dict] = None
        self._cached_panel: Optional[Panel] = None
        self._cached_notebook: Optional[HTML] = None
        self.logger = get_error_logger("ModelCombinedRenderer")


    def _iter_rank_dbs(self, local_db: Database, sampler_name: str):
        """
        Yield (rank, Database) pairs.
        Rank 0 is always local.
        """
        yield 0, local_db

        if not self.remote_store:
            return

        for rank in self.remote_store.ranks():
            db = self.remote_store.get_db(rank, sampler_name)
            if db is not None:
                yield int(rank), db


    def _latest_step_for_db(self, db: Database, table_name: str) -> Optional[int]:
        table = db.create_or_get_table(table_name)
        if not table:
            return None
        try:
            return int(table[-1].get("step"))
        except Exception:
            return None


    def _compute_latest_completed_step(self) -> Optional[int]:
        """
        A step is considered completed if *all ranks* have sent data for it.
        We conservatively take min(latest_step_per_rank).
        """
        steps: List[int] = []

        # Step memory is the strictest signal (flushed at step end)
        local = self._latest_step_for_db(self.memory_db, "step_memory")
        if local is None:
            return None
        steps.append(local)

        if self.is_ddp and self.remote_store:
            for rank in self.remote_store.ranks():
                db = self.remote_store.get_db(rank, "StepMemorySampler")
                if not db:
                    return None
                s = self._latest_step_for_db(db, "step_memory")
                if s is None:
                    return None
                steps.append(s)

        return min(steps) if steps else None


    def _collect_time_series(
        self,
        event_name: str,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Collect per-rank time series for an event.
        """
        out: Dict[int, List[Tuple[int, float]]] = {}

        for rank, db in self._iter_rank_dbs(self.time_db, event_name):
            table = db.create_or_get_table(event_name)
            pairs = _tail_rows(
                table,
                step_key="step",
                value_key="duration_ms",
                limit=self.max_points,
            )
            if pairs:
                out[rank] = pairs

        return out

    def _collect_step_memory(self) -> Dict[int, List[Tuple[int, float]]]:
        out: Dict[int, List[Tuple[int, float]]] = {}

        for rank, db in self._iter_rank_dbs(self.memory_db, "StepMemorySampler"):
            table = db.create_or_get_table("step_memory")
            pairs = _tail_rows(
                table,
                step_key="step",
                value_key="peak_allocated_mb",
                limit=self.max_points,
            )
            if pairs:
                out[rank] = pairs

        return out


    def _aggregate_worst_case(
        self,
        per_rank: Dict[int, List[Tuple[int, float]]],
        completed_step: int,
    ) -> Tuple[np.ndarray, float, float, Optional[int]]:
        """
        Aggregate across ranks using worst-case (max) semantics.

        Returns:
          y          : ndarray of max values per step
          skew_abs   : abs(max - min) at latest step
          skew_pct   : (max - min) / max
          slowest_rank
        """
        if not per_rank:
            return np.asarray([]), 0.0, 0.0, None

        # Align by step <= completed_step
        aligned: Dict[int, Dict[int, float]] = defaultdict(dict)

        for rank, pairs in per_rank.items():
            for step, val in pairs:
                if step <= completed_step:
                    aligned[step][rank] = val

        if not aligned:
            return np.asarray([]), 0.0, 0.0, None

        steps = sorted(aligned.keys())[-self.max_points:]
        y = []
        slowest_rank = None

        for s in steps:
            vals = aligned[s]
            if not vals:
                continue
            max_rank = max(vals, key=lambda r: vals[r])
            y.append(vals[max_rank])
            if s == completed_step:
                slowest_rank = max_rank

        latest_vals = list(aligned[completed_step].values())
        max_v = max(latest_vals)
        min_v = min(latest_vals)

        skew_abs = max_v - min_v
        skew_pct = (skew_abs / max_v) if max_v > 0 else 0.0

        return np.asarray(y), skew_abs, skew_pct, slowest_rank



    def build_live_telemetry_payload(self) -> Dict[str, Any]:
        completed_step = self._compute_latest_completed_step()

        self.logger.error(f"Computed completed_step: {completed_step}")

        # ---- cache hit ----
        if (
            completed_step is None
            or (
                self._cached_completed_step is not None
                and completed_step <= self._cached_completed_step
            )
        ):
            return self._cached_payload or {}

        payload: Dict[str, Any] = {}

        # ---- timers ----
        for internal, friendly in self.FRIENDLY_NAMES.items():
            per_rank = self._collect_time_series(internal)
            y, skew_abs, skew_pct, slowest = self._aggregate_worst_case(
                per_rank, completed_step
            )
            stats = _compute_stats(y)

            payload[friendly] = dict(
                x=np.arange(len(y)),
                y=y,
                stats=dict(
                    **stats,
                    rank_skew_abs=skew_abs,
                    rank_skew_pct=skew_pct,
                    slowest_rank=slowest,
                ),
            )

        # ---- step memory ----
        per_rank_mem = self._collect_step_memory()
        y, skew_abs, skew_pct, slowest = self._aggregate_worst_case(
            per_rank_mem, completed_step
        )
        stats = _compute_stats(y)

        payload["step_gpu_memory"] = dict(
            x=np.arange(len(y)),
            y=y,
            stats=dict(
                **stats,
                rank_skew_abs=skew_abs,
                rank_skew_pct=skew_pct,
                slowest_rank=slowest,
            ),
        )

        # ---- update cache ----
        self._cached_completed_step = completed_step
        self._cached_payload = payload

        return payload



    def get_panel_renderable(self) -> Panel:
        payload = self.build_live_telemetry_payload()

        if not payload and self._cached_panel:
            return self._cached_panel

        table = Table(show_header=True, header_style="bold blue", box=None)
        table.add_column("Metric")
        table.add_column("Last")
        table.add_column("p50")
        table.add_column("p95")
        table.add_column("Avg(100)")
        table.add_column("Trend")
        table.add_column("Rank skew")
        table.add_column("Slowest rank")

        for name, entry in payload.items():
            s = entry["stats"]
            fmt = fmt_mem_new if "memory" in name else fmt_time_run

            table.add_row(
                name,
                fmt(s["last"]),
                fmt(s["p50"]),
                fmt(s["p95"]),
                fmt(s["avg100"]),
                s["trend"],
                f"{s['rank_skew_abs']:.2f} ({s['rank_skew_pct']*100:.1f}%)",
                str(s["slowest_rank"]) if s["slowest_rank"] is not None else "—",
            )

        cols, _ = shutil.get_terminal_size()
        width = min(max(100, int(cols * 0.75)), 120)

        title = "[bold blue]Model Summary[/bold blue]"
        if self._cached_completed_step is not None:
            title += f" (Step {self._cached_completed_step})"

        panel = Panel(table, title=title, width=width)
        self._cached_panel = panel
        return panel

    def get_dashboard_renderable(self):
        return self.build_live_telemetry_payload()

    def get_notebook_renderable(self) -> HTML:
        payload = self.build_live_telemetry_payload()

        # Cache fallback
        if not payload and self._cached_notebook:
            return self._cached_notebook

        def metric_block(title, stats, fmt):
            trend = stats.get("trend", "")

            trend_text = "—"
            trend_color = "#666"

            if isinstance(trend, str) and trend:
                if trend.startswith("+"):
                    trend_text = f"↑ {trend}"
                    trend_color = "#d32f2f"  # regression
                elif trend.startswith("-"):
                    trend_text = f"↓ {trend}"
                    trend_color = "#2e7d32"  # improvement
                elif "≈" in trend:
                    trend_text = trend

            skew_abs = stats.get("rank_skew_abs", 0.0)
            skew_pct = stats.get("rank_skew_pct", 0.0)
            slowest = stats.get("slowest_rank")

            skew_txt = f"{skew_abs:.2f} ({skew_pct * 100:.1f}%)"
            slowest_txt = str(slowest) if slowest is not None else "—"

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
                            <th align="right">Rank skew</th>
                            <th align="center">Slowest</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>{fmt(stats["last"])}</td>
                            <td align="right">{fmt(stats["p50"])}</td>
                            <td align="right">{fmt(stats["p95"])}</td>
                            <td align="right">{fmt(stats["avg100"])}</td>
                            <td align="center" style="color:{trend_color}; font-weight:700;">
                                {trend_text}
                            </td>
                            <td align="right">{skew_txt}</td>
                            <td align="center">{slowest_txt}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """

        blocks = [
            metric_block(
                "Dataloader Fetch Time",
                payload["dataLoader_fetch"]["stats"],
                fmt_time_run,
            ),
            metric_block(
                "Training Step Time",
                payload["step_time"]["stats"],
                fmt_time_run,
            ),
            metric_block(
                "GPU Step Memory",
                payload["step_gpu_memory"]["stats"],
                fmt_mem_new,
            ),
        ]

        html = HTML(f"""
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
        """)

        self._cached_notebook = html
        return html


    def log_summary(self, path: Optional[str] = None) -> None:
        Console().print(self.get_panel_renderable())
