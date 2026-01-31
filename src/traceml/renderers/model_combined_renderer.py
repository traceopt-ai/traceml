"""
ModelCombinedRenderer (DDP-aware) — Worst vs Median per-step aggregation.

What this renderer does
-----------------------
For three core "model summary" metrics:

  1) dataLoader_fetch  (internal timer: _traceml_internal:dataloader_next)  [ms]
  2) step_time         (internal timer: _traceml_internal:step_time)        [ms]
  3) step_gpu_memory   (table: step_memory peak_allocated_mb)               [MB]

it produces a dashboard/CLI/notebook payload that contains TWO per-step
aggregations across ranks:

  - worst  : per-step max across ranks    (DDP gating / tail rank)
  - median : per-step median across ranks (typical rank)

Each metric has its own "step clock" and its own arrival semantics:
  - internal timers are recorded when their events fire
  - step memory is recorded when memory sampling happens

Therefore, we gate (clip) each metric independently using:
    completed_step(metric) = min_r latest_step(rank=r, table=metric_table)

This ensures we only aggregate steps for which *all ranks* have emitted data
for that metric table — without incorrectly forcing timers to be clipped by
the memory table (or vice-versa).

Compatibility guarantee
-----------------------
- Output naming remains unchanged:
    payload keys: "dataLoader_fetch", "step_time", "step_gpu_memory"
- Each entry remains shaped as:
    {
      "steps": List[int],
      "worst":  {"y": np.ndarray, "stats": {...}},
      "median": {"y": np.ndarray, "stats": {...}},
      "rank_skew_abs": float,
      "rank_skew_pct": float,   # 0..1
      "slowest_rank": Optional[int]
    }
- CLI / notebook formatting remains the same.
"""

import shutil
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Iterable

import numpy as np
from IPython.display import HTML
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from traceml.database.database import Database
from traceml.database.remote_database_store import RemoteDBStore
from traceml.distributed import get_ddp_info
from traceml.loggers.error_log import get_error_logger
from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.display.cli_display_manager import MODEL_COMBINED_LAYOUT
from traceml.renderers.utils import fmt_mem_new, fmt_time_run
from .utils import CARD_STYLE


# -------------------------
# Small utilities
# -------------------------


def _tail_rows(
    table: Any,
    step_key: str,
    value_key: str,
    limit: int,
) -> List[Tuple[int, float]]:
    """
    Read at most `limit` rows from the tail of a deque/list-like table.

    Parameters
    ----------
    table:
        Table-like object that supports iteration and reversed(table).
        In TraceML DB this is typically a deque of dict rows.
    step_key:
        Row field name containing the integer step.
    value_key:
        Row field name containing the metric value.
    limit:
        Maximum number of rows to return from the end.

    Returns
    -------
    List[(step:int, value:float)] in ascending step order.

    Notes
    -----
    - Ignores malformed rows gracefully.
    - Safe for missing keys / bad types.
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
    Compute basic telemetry stats over a 1D time series.

    Stats are computed over a trailing window of up to 100 samples (where relevant).

    Returns
    -------
    dict with:
      - last   : latest value
      - p50    : median over last <=100 points
      - p95    : 95th percentile over last <=100 points
      - avg100 : mean over last <=100 points
      - trend  : % change of avg(last<=100) vs avg(previous<=100) if we have >=200 points
                formatted as "+x.x%", "-x.x%", or "≈0%" or "".
    """
    if arr.size == 0:
        return dict(last=0.0, p50=0.0, p95=0.0, avg100=0.0, trend="")

    last = float(arr[-1])
    win100 = arr[-min(100, arr.size) :]
    win200 = arr[-min(200, arr.size) :]

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
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Aggregate a per-rank (step, value) series into WORST and MEDIAN per-step series.

    Parameters
    ----------
    per_rank:
        {rank: [(step, value), ...]}
    completed_step:
        Only consider steps <= completed_step.
        This should already be a per-metric "completed step" (see renderer).
    max_points:
        Keep only the last `max_points` aligned steps.
    debug:
        If True, emit useful debug info.

    Returns
    -------
    dict shaped as:
      {
        "steps": List[int],
        "worst":  {"y": np.ndarray, "stats": {...}},
        "median": {"y": np.ndarray, "stats": {...}},
        "rank_skew_abs": float,
        "rank_skew_pct": float,   # 0..1
        "slowest_rank": Optional[int],
      }

    Notes
    -----
    - We align by step id. If a step has missing ranks, it still exists in `aligned`,
      but the computed worst/median for that step will be based only on available ranks.
      In practice, `completed_step` should ensure all ranks have that step, so missing
      ranks should be rare — but we keep the function safe.
    """
    if not per_rank:
        return {}

    aligned: Dict[int, Dict[int, float]] = defaultdict(dict)

    # Align and bound by completed_step.
    for rank, pairs in per_rank.items():
        for step, val in pairs:
            if step <= completed_step:
                aligned[int(step)][int(rank)] = float(val)

    if not aligned:
        return {}

    steps = sorted(aligned.keys())[-max_points:]

    worst_y: List[float] = []
    median_y: List[float] = []

    slowest_rank: Optional[int] = None
    skew_abs = 0.0
    skew_pct = 0.0

    for s in steps:
        d = aligned[s]
        if not d:
            continue

        vals_sorted = sorted(d.values())
        worst_val = float(vals_sorted[-1])
        median_val = float(np.median(vals_sorted))

        worst_y.append(worst_val)
        median_y.append(median_val)

        # Latest-step diagnostics
        if s == completed_step:
            slowest_rank = max(d, key=lambda r: d[r])
            min_v = float(vals_sorted[0])
            skew_abs = worst_val - min_v
            skew_pct = (skew_abs / worst_val) if worst_val > 0 else 0.0

    worst_arr = np.asarray(worst_y, dtype=float)
    median_arr = np.asarray(median_y, dtype=float)

    if debug:
        # Useful for diagnosing "worst == median" cases.
        # Do not log entire aligned by default in production; it can be large.
        # Here we log only latest step composition.
        latest = steps[-1] if steps else None
        if latest is not None:
            _dbg.error(
                f"[agg debug] latest_step={latest} ranks={sorted(aligned[latest].keys())}"
            )

    return dict(
        steps=steps,
        worst=dict(y=worst_arr, stats=_compute_stats(worst_arr)),
        median=dict(y=median_arr, stats=_compute_stats(median_arr)),
        rank_skew_abs=skew_abs,
        rank_skew_pct=skew_pct,
        slowest_rank=slowest_rank,
    )


# -------------------------
# Renderer
# -------------------------


class ModelCombinedRenderer(BaseRenderer):
    """
    Step-level model summary renderer (DDP-aware).

    For each of the three summary metrics, compute:
      - per-step WORST and MEDIAN across ranks
      - rolling stats (last/p50/p95/avg100/trend) for each series
      - latest-step diagnostics: skew (abs + pct), slowest_rank

    Correctness: per-metric completion gating
    ----------------------------------------
    Each metric is gated with its own per-metric completed_step:
        completed_step(table) = min_r latest_step(rank=r, table=table)

    This avoids incorrectly using step_memory to clip timer tables.
    """

    # Internal table names -> output keys (must remain stable for compatibility).
    FRIENDLY_NAMES: Dict[str, str] = {
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
            name="Model Summary", layout_section_name=MODEL_COMBINED_LAYOUT
        )

        self.time_db = time_db
        self.memory_db = memory_db
        self.remote_store = remote_store
        self.window = int(window)
        self.max_points = int(max_points)

        # DDP info (rank-0 renderer typically runs with remote_store attached).
        self.is_ddp, _, self.world_size = get_ddp_info()

        # Cache keyed by a *composite signature* rather than one completed_step.
        # We keep it simple: store last computed payload and recompute if any metric advanced.
        self._cached_signature: Optional[Tuple[int, int, int]] = None
        self._cached_payload: Optional[dict] = None
        self._cached_panel: Optional[Panel] = None
        self._cached_notebook: Optional[HTML] = None

        self.logger = get_error_logger("ModelCombinedRenderer")

    # ---------
    # Rank DB iteration
    # ---------

    def _iter_rank_dbs(
        self, local_db: Database, sampler_name: str
    ) -> Iterable[Tuple[int, Database]]:
        """
        Yield (rank, Database) pairs.
        Rank 0 is always local.

        `sampler_name` is used as the lookup key in RemoteDBStore.
        In the time DB path we use event_name as sampler_name (as in your original code).
        In memory DB we use "StepMemorySampler".
        """
        yield 0, local_db

        if not self.remote_store:
            return

        for rank in self.remote_store.ranks():
            db = self.remote_store.get_db(rank, sampler_name)
            if db is not None:
                yield int(rank), db

    def _latest_step_for_db(self, db: Database, table_name: str) -> Optional[int]:
        """
        Return the last step recorded in `table_name` for this DB, else None.
        """
        table = db.create_or_get_table(table_name)
        if not table:
            return None
        try:
            return int(table[-1].get("step"))
        except Exception:
            return None

    # ---------
    # Per-metric completion gating (the fix)
    # ---------

    def _compute_latest_completed_step_for_table(
        self,
        *,
        table_name: str,
        local_db: Database,
        sampler_name: str,
    ) -> Optional[int]:
        """
        Compute "completed step" for a specific metric table.

        Definition:
          completed_step(table) = min_r latest_step(rank=r, table=table)

        That is: the latest step for which *all ranks* have emitted a record into that table.

        Parameters
        ----------
        table_name:
            DB table to check (e.g. "_traceml_internal:step_time", "step_memory")
        local_db:
            The local DB (rank 0)
        sampler_name:
            RemoteDBStore lookup key (often equals table name for timers, or sampler class name)

        Returns
        -------
        int step, or None if any rank has not produced any rows yet.
        """
        steps: List[int] = []

        local = self._latest_step_for_db(local_db, table_name)
        if local is None:
            return None
        steps.append(local)

        if self.remote_store:
            # remote_store.ranks() typically yields worker ranks only (1..ws-1).
            for rank in self.remote_store.ranks():
                rdb = self.remote_store.get_db(rank, sampler_name)
                if not rdb:
                    return None
                s = self._latest_step_for_db(rdb, table_name)
                if s is None:
                    return None
                steps.append(s)

        return min(steps) if steps else None

    def _collect_time_series(
        self, event_name: str
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Collect per-rank time series for a given time event table.

        Returns:
          { rank: [(step, duration_ms), ...] }
        """
        out: Dict[int, List[Tuple[int, float]]] = {}

        # In this implementation: sampler_name == event_name for time_db tables.
        for rank, db in self._iter_rank_dbs(self.time_db, "StepTimerSampler"):
            table = db.create_or_get_table(event_name)
            pairs = _tail_rows(
                table, step_key="step", value_key="duration_ms", limit=self.max_points
            )
            if pairs:
                out[rank] = pairs
        return out

    def _collect_step_memory(self) -> Dict[int, List[Tuple[int, float]]]:
        """
        Collect per-rank step memory series.

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
        Build a snapshot payload for dashboard / CLI / notebook.

        Output shape (compatible with existing UI code):
          payload = {
            "dataLoader_fetch": {...},
            "step_time": {...},
            "step_gpu_memory": {...},
          }

        Cache:
          We cache based on a signature tuple of the per-metric completed steps:
            (dl_completed, step_completed, mem_completed)
        """
        # Compute per-metric completed steps (correctness fix).
        dl_completed = self._compute_latest_completed_step_for_table(
            table_name="_traceml_internal:dataloader_next",
            local_db=self.time_db,
            sampler_name="StepTimerSampler",
        )
        st_completed = self._compute_latest_completed_step_for_table(
            table_name="_traceml_internal:step_time",
            local_db=self.time_db,
            sampler_name="StepTimerSampler",
        )
        mem_completed = self._compute_latest_completed_step_for_table(
            table_name="step_memory",
            local_db=self.memory_db,
            sampler_name="StepMemorySampler",
        )

        # If nothing is available yet, return cached payload or empty.
        if dl_completed is None and st_completed is None and mem_completed is None:
            return self._cached_payload or {}

        signature = (
            int(dl_completed or -1),
            int(st_completed or -1),
            int(mem_completed or -1),
        )

        # Cache hit: if nothing advanced, return cached.
        if self._cached_signature is not None and signature <= self._cached_signature:
            return self._cached_payload or {}

        payload: Dict[str, Any] = {}

        # ---- timers (each gated by its own completed step) ----
        for internal, friendly in self.FRIENDLY_NAMES.items():
            completed_step = (
                dl_completed if internal.endswith("dataloader_next") else st_completed
            )
            if completed_step is None:
                continue

            per_rank = self._collect_time_series(internal)
            agg = _aggregate_worst_and_median(
                per_rank=per_rank,
                completed_step=int(completed_step),
                max_points=self.max_points,
                debug=False,
            )
            if agg:
                payload[friendly] = agg

        # ---- step memory (gated by memory completed step) ----
        if mem_completed is not None:
            per_rank_mem = self._collect_step_memory()
            agg_mem = _aggregate_worst_and_median(
                per_rank=per_rank_mem,
                completed_step=int(mem_completed),
                max_points=self.max_points,
                debug=False,
            )
            if agg_mem:
                payload["step_gpu_memory"] = agg_mem

        self._cached_signature = signature
        self._cached_payload = payload

        return payload

    def get_panel_renderable(self) -> Panel:
        """
        CLI panel renderable.

        Compact 3-metric layout (columns) with stats as rows.
        Each value cell is rendered as: Worst/Median

        Row order (action-first):
          1) Worst rank
          2) Skew (last)
          3) Trend
          4) p50
          5) p95
          6) Last

        Styling:
          - Header stays "bold blue".
          - Row labels are colored (bold cyan).
          - Values remain default (white).
        """
        payload = self.build_live_telemetry_payload()
        if not payload and self._cached_panel:
            return self._cached_panel

        table = Table(show_header=True, header_style="bold blue", box=None)
        table.add_column("[bold cyan]Stat[/bold cyan]")
        table.add_column("[bold blue]Dataload time[/bold blue]", justify="right")
        table.add_column("[bold blue]Step time[/bold blue]", justify="right")
        table.add_column("[bold blue]Step memory[/bold blue]", justify="right")

        def _fmt_cell(key: str, kind: str) -> str:
            """
            Format a single cell for a metric key, as Worst/Median.

            kind in: worst_rank | skew | trend | p50 | p95 | last
            """
            entry = payload.get(key)
            if not entry:
                return "—"

            # Metric-specific formatting for values.
            val_fmt = fmt_mem_new if "memory" in key else fmt_time_run

            ws = entry["worst"]["stats"]
            ms = entry["median"]["stats"]

            if kind == "worst_rank":
                r = entry.get("slowest_rank")
                return f"r{r}" if r is not None else "—"

            if kind == "skew":
                skew_abs = float(entry.get("rank_skew_abs", 0.0))
                skew_pct = float(entry.get("rank_skew_pct", 0.0)) * 100.0
                # Keep skew readable; abs uses mem formatter in your original.
                return f"{fmt_mem_new(skew_abs)} ({skew_pct:.1f}%)"

            if kind == "trend":
                t = ws.get("trend") or "—"
                return f"{t}/—"

            if kind == "p50":
                return f"{val_fmt(ws['p50'])}/{val_fmt(ms['p50'])}"

            if kind == "p95":
                return f"{val_fmt(ws['p95'])}/{val_fmt(ms['p95'])}"

            if kind == "last":
                return f"{val_fmt(ws['last'])}/{val_fmt(ms['last'])}"

            return "—"

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
        if self._cached_signature is not None:
            # show the most conservative "global" step for display only
            step_for_title = max(
                self._cached_signature
            )  # signature stores latest completed per metric
            if step_for_title >= 0:
                title += f" (Step {step_for_title})"

        panel = Panel(table, title=title, width=width)
        self._cached_panel = panel
        return panel

    def get_dashboard_renderable(self) -> Dict[str, Any]:
        """Dashboard renderable is the raw snapshot payload."""
        return self.build_live_telemetry_payload()

    def get_notebook_renderable(self) -> HTML:
        """
        Notebook HTML renderable.

        Shows WORST vs MEDIAN summary for each metric,
        plus rank skew and worst rank for the latest completed step of that metric.
        """
        payload = self.build_live_telemetry_payload()

        if not payload and self._cached_notebook:
            return self._cached_notebook

        def _trend_badge(trend: str) -> Tuple[str, str]:
            if not isinstance(trend, str) or not trend:
                return "—", "#666"
            if trend.startswith("+"):
                return f"↑ {trend}", "#d32f2f"
            if trend.startswith("-"):
                return f"↓ {trend}", "#2e7d32"
            if "≈" in trend:
                return trend, "#666"
            return trend, "#666"

        def metric_block(title: str, entry: Dict[str, Any], fmt) -> str:
            ws = entry["worst"]["stats"]
            ms = entry["median"]["stats"]

            trend_text, trend_color = _trend_badge(ws.get("trend", ""))

            skew_abs = float(entry.get("rank_skew_abs", 0.0))
            skew_pct = float(entry.get("rank_skew_pct", 0.0))
            slowest = entry.get("slowest_rank")

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
        if "dataLoader_fetch" in payload:
            blocks.append(
                metric_block(
                    "Dataloader Fetch Time", payload["dataLoader_fetch"], fmt_time_run
                )
            )
        if "step_time" in payload:
            blocks.append(
                metric_block("Training Step Time", payload["step_time"], fmt_time_run)
            )
        if "step_gpu_memory" in payload:
            blocks.append(
                metric_block("GPU Step Memory", payload["step_gpu_memory"], fmt_mem_new)
            )

        html = HTML(
            f"""
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
        )

        self._cached_notebook = html
        return html

    def log_summary(self, path: Optional[str] = None) -> None:
        """Print the CLI panel."""
        Console().print(self.get_panel_renderable())
