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



def _tail_rows(
    table,
    step_key: str,
    value_key: str,
    limit: int,
) -> List[Tuple[int, float]]:
    """
    Read at most `limit` rows from the *tail* of a deque-backed table.

    Notes
    -----
    - Designed to be safe for deque or list-like storage.
    - Ignores malformed rows gracefully.
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
    """Percentile helper that never throws on empty arrays."""
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def _compute_stats(arr: np.ndarray) -> Dict[str, Any]:
    """
    Compute basic telemetry stats over a 1D series.

    Metrics
    -------
    last   : latest value
    p50    : median over last <=100 points
    p95    : 95th percentile over last <=100 points
    avg100 : mean over last <=100 points
    trend  : % change of avg(last<=100) vs avg(previous<=100) if we have >=200 points

    Returns zeros on empty input.
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

    return dict(last=last, p50=p50, p95=p95, avg100=avg100, trend=trend)


def _aggregate_worst_and_median(
    per_rank: Dict[int, List[Tuple[int, float]]],
    completed_step: int,
    max_points: int,
) -> Dict[str, Any]:
    """
    Aggregate a per-rank time series into:
      - worst series: per-step max across ranks (DDP gating)
      - median series: per-step median across ranks (typical rank)

    This function is intentionally compute-light:
    - only considers steps <= completed_step
    - only keeps last `max_points` steps
    - per-step cost is O(world_size log world_size) due to sorting (world_size is small)

    Returns
    -------
    {
      "steps": List[int],               # actual step ids aligned across ranks
      "worst":  { "y": np.ndarray, "stats": {...} },
      "median": { "y": np.ndarray, "stats": {...} },

      # Latest-step diagnostics (based on raw per-rank values at completed_step)
      "rank_skew_abs": float,           # max - min at latest step
      "rank_skew_pct": float,           # (max - min)/max at latest step
      "slowest_rank": Optional[int],    # argmax rank at latest step (worst rank)
    }
    """
    if not per_rank:
        return {}

    # step -> {rank: value}
    aligned: Dict[int, Dict[int, float]] = defaultdict(dict)

    # Align and bound by completed_step
    for rank, pairs in per_rank.items():
        for step, val in pairs:
            if step <= completed_step:
                aligned[step][rank] = val

    if not aligned:
        return {}

    steps = sorted(aligned.keys())[-max_points:]

    worst_y: List[float] = []
    median_y: List[float] = []

    slowest_rank: Optional[int] = None
    skew_abs = 0.0
    skew_pct = 0.0

    # Build both series in one pass
    for s in steps:
        d = aligned[s]
        if not d:
            continue

        vals = list(d.values())
        # Sorting is simple and stable for small world_size
        vals_sorted = sorted(vals)

        worst_val = float(vals_sorted[-1])
        median_val = float(np.median(vals_sorted))

        worst_y.append(worst_val)
        median_y.append(median_val)

        if s == completed_step:
            # Diagnostics only for the latest completed step
            slowest_rank = max(d, key=lambda r: d[r])
            min_v = float(vals_sorted[0])
            skew_abs = worst_val - min_v
            skew_pct = (skew_abs / worst_val) if worst_val > 0 else 0.0

    worst_arr = np.asarray(worst_y, dtype=float)
    median_arr = np.asarray(median_y, dtype=float)

    return dict(
        steps=steps,
        worst=dict(
            y=worst_arr,
            stats=_compute_stats(worst_arr),
        ),
        median=dict(
            y=median_arr,
            stats=_compute_stats(median_arr),
        ),
        rank_skew_abs=skew_abs,
        rank_skew_pct=skew_pct,
        slowest_rank=slowest_rank,
    )


class ModelCombinedRenderer(BaseRenderer):
    """
    Step-level model summary renderer (DDP-aware).

    Functionality:
    --------------
    For each metric, this renderer computes TWO per-step aggregations across ranks:
      1) WORST  : max across ranks per step (DDP gating / tail behavior)
      2) MEDIAN : median across ranks per step (typical rank behavior)

    Then it computes stats over time *separately* for each series:
      - last, p50, p95, avg100, trend

    Additional latest-step diagnostics (based on raw per-rank values at the latest completed step):
      - rank skew (abs and %)
      - slowest_rank (worst rank for the metric, at latest completed step)

    Important note
    --------------
    - "Latest completed step" is defined conservatively as:
        min(latest_step_per_rank) using step_memory (flushed at step end).
      This avoids partial steps and inconsistent cross-rank comparisons.
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
        # Cache is keyed by completed_step;
        # prevents recompute when dashboard refreshes faster than training steps.
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

        We use step_memory as the strictest completion signal (flushed at step end).
        """
        steps: List[int] = []

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

    def _collect_time_series(self, event_name: str) -> Dict[int, List[Tuple[int, float]]]:
        """
        Collect per-rank time series for a given event table.

        Returns:
          { rank: [(step, duration_ms), ...] }
        """
        out: Dict[int, List[Tuple[int, float]]] = {}

        # NOTE: sampler_name == event_name for time_db tables in this implementation.
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
        """
        Collect per-rank step memory time series.

        Returns:
          { rank: [(step, peak_allocated_mb), ...] }
        """
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

    def build_live_telemetry_payload(self) -> Dict[str, Any]:
        """
        Build a snapshot payload for the dashboard / CLI / notebook.

        Snapshot semantics
        ------------------
        For each metric key, returns:
          - steps: actual step ids aligned across ranks
          - worst:  series + stats over time
          - median: series + stats over time
          - rank_skew_abs / rank_skew_pct / slowest_rank (latest step diagnostics)
        """
        completed_step = self._compute_latest_completed_step()

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
            agg = _aggregate_worst_and_median(
                per_rank=per_rank,
                completed_step=completed_step,
                max_points=self.max_points,
            )
            if agg:
                payload[friendly] = agg

        # ---- step memory ----
        per_rank_mem = self._collect_step_memory()
        agg_mem = _aggregate_worst_and_median(
            per_rank=per_rank_mem,
            completed_step=completed_step,
            max_points=self.max_points,
        )
        if agg_mem:
            payload["step_gpu_memory"] = agg_mem

        # ---- update cache ----
        self._cached_completed_step = completed_step
        self._cached_payload = payload

        return payload

    def get_panel_renderable(self) -> Panel:
        """
        CLI panel renderable.

        Compact 3-metric layout (columns) with stats as rows.
        Each value cell is rendered as:  Worst/Median

        Row order (action-first):
          1) Worst rank
          2) Skew (last)
          3) Trend
          4) p50
          5) p95
          6) Last

        Styling:
          - Header stays "bold blue" like the original.
          - Row labels are colored (bold cyan).
          - Values remain default (white).
        """
        payload = self.build_live_telemetry_payload()

        if not payload and self._cached_panel:
            return self._cached_panel

        # Ensure stable column ordering
        col_order = [
            ("dataLoader_fetch", "Dataloader"),
            ("step_time", "Step time"),
            ("step_gpu_memory", "Step memory"),
        ]

        # 4-column table: Stat | Dataloader | Step time | Step memory
        table = Table(show_header=True, header_style="bold blue", box=None)
        table.add_column("[bold cyan]Stat[/bold cyan]")
        table.add_column("[bold blue]Dataload time[/bold blue]", justify="right")
        table.add_column("[bold blue]Step time[/bold blue]", justify="right")
        table.add_column("[bold blue]Step memory[/bold blue]", justify="right")

        def _fmt_cell(key: str, kind: str) -> str:
            """
            Format a single cell for a metric key, as Worst/Median.

            kind:
              - "worst_rank"
              - "skew"
              - "trend"
              - "p50"
              - "p95"
              - "last"
            """
            entry = payload.get(key)
            if not entry:
                return "—"

            # Metric-specific formatter
            fmt = fmt_mem_new if "memory" in key else fmt_time_run

            ws = entry["worst"]["stats"]
            ms = entry["median"]["stats"]

            if kind == "worst_rank":
                r = entry.get("slowest_rank")
                return f"r{r}" if r is not None else "—"

            if kind == "skew":
                skew_abs = entry.get("rank_skew_abs", 0.0)
                skew_pct = entry.get("rank_skew_pct", 0.0) * 100.0
                # Keep skew readable; abs uses mem formatter like your original code.
                return f"{fmt_mem_new(skew_abs)} ({skew_pct:.1f}%)"

            if kind == "trend":
                # Trend is computed only for worst series; median left as "—" to avoid noise.
                t = ws.get("trend") or "—"
                return f"{t}/—"

            if kind == "p50":
                return f"{fmt(ws['p50'])}/{fmt(ms['p50'])}"

            if kind == "p95":
                return f"{fmt(ws['p95'])}/{fmt(ms['p95'])}"

            if kind == "last":
                return f"{fmt(ws['last'])}/{fmt(ms['last'])}"

            return "—"

        # Rows (action-first)
        rows = [
            ("[bold cyan]Worst rank[/bold cyan]", "worst_rank"),
            ("[bold cyan]Skew (last)[/bold cyan]", "skew"),
            ("[bold cyan]Trend[/bold cyan]", "trend"),
            ("[bold cyan]p50[/bold cyan]", "p50"),
            ("[bold cyan]p95[/bold cyan]", "p95"),
            ("[bold cyan]Last[/bold cyan]", "last"),
        ]

        for label, kind in rows:
            table.add_row(
                label,
                _fmt_cell("dataLoader_fetch", kind),
                _fmt_cell("step_time", kind),
                _fmt_cell("step_gpu_memory", kind),
            )

        cols, _ = shutil.get_terminal_size()
        width = min(max(100, int(cols * 0.75)), 120)

        title = "[bold blue]Model Summary (worst/median)[/bold blue]"
        if self._cached_completed_step is not None:
            title += f" (Step {self._cached_completed_step})"

        panel = Panel(table, title=title, width=width)
        self._cached_panel = panel
        return panel


    def get_dashboard_renderable(self):
        """Dashboard renderable is the raw snapshot payload."""
        return self.build_live_telemetry_payload()

    def get_notebook_renderable(self) -> HTML:
        """
        Notebook HTML renderable.

        Shows WORST vs MEDIAN summary for each metric (p50/p95/avg/last/trend),
        plus rank skew and worst rank for latest completed step.
        """
        payload = self.build_live_telemetry_payload()

        # Cache fallback
        if not payload and self._cached_notebook:
            return self._cached_notebook

        def _trend_badge(trend: str) -> Tuple[str, str]:
            """Return (text, color) for a small trend badge."""
            if not isinstance(trend, str) or not trend:
                return "—", "#666"
            if trend.startswith("+"):
                return f"↑ {trend}", "#d32f2f"  # regression
            if trend.startswith("-"):
                return f"↓ {trend}", "#2e7d32"  # improvement
            if "≈" in trend:
                return trend, "#666"
            return trend, "#666"

        def metric_block(title: str, entry: Dict[str, Any], fmt) -> str:
            ws = entry["worst"]["stats"]
            ms = entry["median"]["stats"]

            trend_text, trend_color = _trend_badge(ws.get("trend", ""))

            skew_abs = entry.get("rank_skew_abs", 0.0)
            skew_pct = entry.get("rank_skew_pct", 0.0)
            slowest = entry.get("slowest_rank")

            skew_txt = f"{skew_abs:.2f} ({skew_pct * 100:.1f}%)"
            slowest_txt = str(slowest) if slowest is not None else "—"

            # Display both WORST and MEDIAN stats. Keep it compact.
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
                            <th align="left">Series</th>
                            <th align="right">Last</th>
                            <th align="right">p50</th>
                            <th align="right">p95</th>
                            <th align="right">Avg(100)</th>
                            <th align="center">Trend</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom:1px solid #f0f0f0;">
                            <td><b>Worst</b></td>
                            <td align="right">{fmt(ws["last"])}</td>
                            <td align="right">{fmt(ws["p50"])}</td>
                            <td align="right">{fmt(ws["p95"])}</td>
                            <td align="right">{fmt(ws["avg100"])}</td>
                            <td align="center" style="color:{trend_color}; font-weight:700;">
                                {trend_text}
                            </td>
                        </tr>
                        <tr>
                            <td><b>Median</b></td>
                            <td align="right">{fmt(ms["last"])}</td>
                            <td align="right">{fmt(ms["p50"])}</td>
                            <td align="right">{fmt(ms["p95"])}</td>
                            <td align="right">{fmt(ms["avg100"])}</td>
                            <td align="center" style="color:#666; font-weight:700;">—</td>
                        </tr>
                    </tbody>
                </table>

                <div style="margin-top:8px; font-size:12px; color:#444;">
                    <span><b>Rank skew:</b> {skew_txt}</span>
                    <span style="margin-left:10px;"><b>Worst rank:</b> {slowest_txt}</span>
                </div>
            </div>
            """

        blocks: List[str] = []

        # Use friendly ordering and be robust if a metric isn't present yet.
        if "dataLoader_fetch" in payload:
            blocks.append(metric_block("Dataloader Fetch Time", payload["dataLoader_fetch"], fmt_time_run))
        if "step_time" in payload:
            blocks.append(metric_block("Training Step Time", payload["step_time"], fmt_time_run))
        if "step_gpu_memory" in payload:
            blocks.append(metric_block("GPU Step Memory", payload["step_gpu_memory"], fmt_mem_new))

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
        """Simple helper for printing the CLI panel."""
        Console().print(self.get_panel_renderable())
