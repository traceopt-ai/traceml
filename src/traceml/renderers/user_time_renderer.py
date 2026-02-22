import shutil
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from IPython.display import HTML
from rich.console import Group
from rich.panel import Panel
from rich.table import Table

from traceml.database.database import Database
from traceml.database.remote_database_store import RemoteDBStore
from traceml.renderers.base_renderer import BaseRenderer
from traceml.aggregator.display_drivers.layout import (
    STEPTIMER_LAYOUT,
)
from traceml.renderers.utils import fmt_time_run
from traceml.transport.distributed import get_ddp_info

from .utils import CARD_STYLE


@dataclass
class UserTimeRow:
    """
    One display row for a user-defined timer, aggregated across ranks.

    Semantics:
      - last / p50_100 / p95_100 / avg_100 are "worst-case across ranks" over a recent window.
      - Aggregation across ranks uses "worst wins" semantics (DDP bottleneck model)
      - No step alignment is performed
    """

    name: str
    last: float
    p50_100: float
    p95_100: float
    avg_100: float
    trend: str
    device: str
    worst_rank: str
    coverage: str
    min_samples: int


class UserTimeRenderer(BaseRenderer):
    """
    Renderer for **user-defined timing events** (`@trace_time`).

    DDP semantics (NO STEP alignment):
      - Each rank produces a sequence of timer durations per event name.
      - We compute stats per rank on the last N samples (default N=100).
      - We then aggregate across ranks using "worst wins" to reflect DDP bottleneck behavior.

    Why no step alignment:
      - Timers are event-based and may not occur at exactly the same steps on all ranks.
      - Step alignment would drop useful data and delay detection.
      - Instead we compute stable windowed stats per rank and pick the slowest rank.

    Output:
      - last / p50 / p95 / avg: worst-case across ranks (over window)
      - worst_rank: rank with highest p95 (default) or avg over window
      - coverage: ranks_with_data / world_size
      - min_samples: minimum #samples among ranks that contributed (best-effort)
    """

    SAMPLER_NAME = "TimeSampler"

    def __init__(
        self,
        top_n: int = 5,
        remote_store: Optional[RemoteDBStore] = None,
        window_size: int = 100,
        worst_metric: str = "p95",  # "p95" or "avg"
        show_internal: bool = False,
    ):
        super().__init__(
            name="User Times", layout_section_name=STEPTIMER_LAYOUT
        )
        self.top_n = int(top_n)
        self.remote_store = remote_store
        self.show_internal = show_internal

        # Window size used for stable aggregation and trend detection.
        self.window_size = max(int(window_size), 1)

        # Stable "worst rank" definition.
        self.worst_metric = worst_metric.lower().strip()
        if self.worst_metric not in ("p95", "avg"):
            self.worst_metric = "p95"

    def _infer_world_size(self) -> int:
        """
        Best-effort world size inference.
        In non-DDP runs, returns 1.
        """
        try:
            _, _, ws = get_ddp_info()
            return max(int(ws), 1)
        except Exception:
            return 1

    def _iter_rank_dbs(self) -> Iterable[Tuple[int, Database]]:
        """
        Yield (rank, db) for the local DB first, then remote rank DBs (if available).

        Important:
          - The local DB is always treated as rank 0 in this renderer, consistent with
            RemoteDBStore usage on rank 0.
          - In worker ranks, remote_store is typically None, so we only yield local.
        """
        if not self.remote_store:
            return

        ws = self._infer_world_size()

        # RemoteDBStore is expected to serve per-rank DBs by (rank, sampler_name)
        for rank in range(0, ws):
            db = (
                self.remote_store.get_db(rank, self.SAMPLER_NAME)
                if self.SAMPLER_NAME
                else None
            )
            if db is not None:
                yield rank, db

    def _is_internal(self, name: str) -> bool:
        return name.startswith("_traceml_internal:")

    def _collect_series_by_rank(
        self,
    ) -> Dict[str, Dict[int, Dict[str, List[float]]]]:
        """
        Collect per-event series for each rank.

        Returns:
          {
            event_name: {
              rank: {"cpu": [..], "gpu": [..]}   # exactly one of cpu/gpu is populated
            }
          }

        Notes:
          - We do not step-align events. We treat each rank's event stream independently.
          - We assume event tables store rows with 'duration_ms' and boolean 'is_gpu'.
        """
        out: Dict[str, Dict[int, Dict[str, List[float]]]] = {}

        for rank, db in self._iter_rank_dbs():
            for table_name, rows in db.all_tables().items():
                if not self.show_internal and self._is_internal(table_name):
                    continue
                if not rows:
                    continue

                # Extract durations for this rank/event.
                vals = [float(r.get("duration_ms", 0.0)) for r in rows]
                is_gpu = bool(rows[-1].get("is_gpu"))

                if table_name not in out:
                    out[table_name] = {}

                out[table_name][rank] = {
                    "cpu": [] if is_gpu else vals,
                    "gpu": vals if is_gpu else [],
                }
        return out

    @staticmethod
    def _safe_percentile(arr: np.ndarray, q: float) -> float:
        return float(np.percentile(arr, q)) if arr.size else 0.0

    def _window(self, vals: List[float]) -> np.ndarray:
        """
        Return last `window_size` samples as a float64 array.
        """
        if not vals:
            return np.asarray([], dtype=np.float64)
        n = min(self.window_size, len(vals))
        return np.asarray(vals[-n:], dtype=np.float64)

    def _rank_stats(
        self,
        cpu_vals: List[float],
        gpu_vals: List[float],
    ) -> Dict[str, float]:
        """
        Compute rank-local windowed stats (no cross-rank logic).

        Returns keys:
          - last, p50, p95, avg, nsamples
        """
        arr = self._window(gpu_vals or cpu_vals)
        if arr.size == 0:
            return {
                "last": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "avg": 0.0,
                "nsamples": 0.0,
            }

        return {
            "last": float(arr[-1]),
            "p50": self._safe_percentile(arr, 50),
            "p95": self._safe_percentile(arr, 95),
            "avg": float(arr.mean()),
            "nsamples": float(arr.size),
        }

    def _trend_from_history(self, arr_full: np.ndarray) -> str:
        """
        Trend indicator based on recent vs previous window (stable, best-effort).

        Uses the full series for the event on the chosen device for the aggregated view.
        (We apply this only after we decide the aggregated device stream.)
        """
        n = arr_full.size
        if n == 0:
            return ""

        # Early instability: if early p95 is much worse than later average.
        if 50 <= n <= 200:
            early = arr_full[:50]
            rest = arr_full[50:] if n > 50 else arr_full
            early_p95 = np.percentile(early, 95)
            later_avg = rest.mean() if rest.size else arr_full.mean()

            if later_avg > 1e-9 and early_p95 > 2.0 * later_avg:
                pct = (early_p95 - later_avg) / later_avg * 100.0
                if abs(pct) >= 5.0:
                    sign = "+" if pct > 0 else ""
                    return f"! {sign}{pct:.1f}%"
                return "!"

        # Recent vs previous: compare last window vs previous window (both 100).
        if n >= 200:
            recent_avg = arr_full[-100:].mean()
            prev_avg = arr_full[-200:-100].mean()

            if prev_avg > 1e-9:
                pct_change = (recent_avg - prev_avg) / prev_avg * 100.0
                if abs(pct_change) < 1.0:
                    return "≈0%"
                sign = "+" if pct_change > 0 else ""
                return f"{sign}{pct_change:.1f}%"

        return ""

    def _compute_aggregated_row(
        self,
        name: str,
        per_rank: Dict[int, Dict[str, List[float]]],
    ) -> UserTimeRow:
        """
        Compute a single aggregated row across ranks for one timer event.

        Aggregation rules:
          - Compute per-rank stats over last N samples.
          - Choose stable worst rank based on chosen metric ('p95' or 'avg').
          - Display values as worst-case across ranks (max of each stat).
          - coverage/min_samples describe data availability.
        """
        ws = self._infer_world_size()

        # Per-rank windowed stats.
        rank_stats: Dict[int, Dict[str, float]] = {}
        ranks_with_data: List[int] = []

        # Track device consistency: event should typically be CPU-only or GPU-only.
        # If some ranks disagree, we label it "MIXED" (rare, but safe).
        saw_cpu = False
        saw_gpu = False

        # For trend: we build an aggregated "history stream" (best-effort) by concatenating
        # all available samples across ranks. This is not step-aligned, but trend is heuristic.
        history_concat: List[float] = []

        for r in range(ws):
            vals = per_rank.get(r)
            if not vals:
                continue

            cpu_vals = vals.get("cpu", []) or []
            gpu_vals = vals.get("gpu", []) or []

            if gpu_vals:
                saw_gpu = True
                history_concat.extend(gpu_vals)
            else:
                saw_cpu = True
                history_concat.extend(cpu_vals)

            st = self._rank_stats(cpu_vals=cpu_vals, gpu_vals=gpu_vals)
            if st["nsamples"] > 0:
                rank_stats[r] = st
                ranks_with_data.append(r)

        # Coverage and minimum samples (among ranks that have any data).
        if ranks_with_data:
            min_samples = int(
                min(rank_stats[r]["nsamples"] for r in ranks_with_data)
            )
        else:
            min_samples = 0

        coverage = f"{len(ranks_with_data)}/{ws}" if ws > 1 else "1/1"

        device = (
            "MIXED" if (saw_cpu and saw_gpu) else ("GPU" if saw_gpu else "CPU")
        )

        if not rank_stats:
            return UserTimeRow(
                name=name,
                last=0.0,
                p50_100=0.0,
                p95_100=0.0,
                avg_100=0.0,
                trend="",
                device=device,
                worst_rank="—",
                coverage=coverage,
                min_samples=min_samples,
            )

        # Stable worst rank selection (windowed, not per-step).
        metric_key = "p95" if self.worst_metric == "p95" else "avg"
        worst_rank_id = max(
            rank_stats.keys(), key=lambda r: float(rank_stats[r][metric_key])
        )
        worst_rank = f"{worst_rank_id}"

        # Display worst-case values across ranks.
        last = max(float(s["last"]) for s in rank_stats.values())
        p50 = max(float(s["p50"]) for s in rank_stats.values())
        p95 = max(float(s["p95"]) for s in rank_stats.values())
        avg = max(float(s["avg"]) for s in rank_stats.values())

        # Trend heuristic based on concatenated history.
        trend = self._trend_from_history(
            np.asarray(history_concat, dtype=np.float64)
        )

        return UserTimeRow(
            name=name,
            last=last,
            p50_100=p50,
            p95_100=p95,
            avg_100=avg,
            trend=trend,
            device=device,
            worst_rank=worst_rank,
            coverage=coverage,
            min_samples=min_samples,
        )

    def _score_for_sort(self, row: UserTimeRow) -> float:
        """
        Sort score: worst-case avg_100 by default.
        (You can switch to p95_100 if you prefer.)
        """
        return float(row.avg_100)

    def _build_rows(self) -> List[UserTimeRow]:
        """
        Build StepTimerRow rows for display.

        We optionally keep top_n-1 rows plus an "Other" row, similar to v1.
        "Other" is aggregated as a single distribution across remaining events, with
        worst_rank shown as "—" because it does not represent a single timer.
        """
        series_by_rank = self._collect_series_by_rank()
        if not series_by_rank:
            return []

        # Compute a row per event.
        rows = [
            self._compute_aggregated_row(name, per_rank)
            for name, per_rank in series_by_rank.items()
        ]

        # Stable display order: slowest events first.
        rows.sort(key=self._score_for_sort, reverse=True)

        if self.top_n <= 0 or len(rows) <= self.top_n:
            return rows

        # Keep top_n-1, aggregate the rest into "Other".
        keep = rows[: self.top_n - 1]
        rest_names = [r.name for r in rows[self.top_n - 1 :]]

        # Aggregate "Other" by concatenating samples across remaining events and ranks.
        # This preserves "distributional" meaning but does not represent a single timer/rank.
        other_cpu: List[float] = []
        other_gpu: List[float] = []

        for name in rest_names:
            per_rank = series_by_rank.get(name, {})
            for _, vals in per_rank.items():
                other_cpu.extend(vals.get("cpu", []) or [])
                other_gpu.extend(vals.get("gpu", []) or [])

        other_row = self._compute_aggregated_row(
            name="Other",
            per_rank={0: {"cpu": other_cpu, "gpu": other_gpu}},
        )
        # Override fields that are not meaningful for "Other".
        other_row.worst_rank = "—"
        other_row.coverage = "—"
        other_row.min_samples = 0

        # Ensure "Other" is always last.
        return keep + [other_row]

    # Renderers: Rich / Notebook / Dashboard
    def get_panel_renderable(self) -> Panel:
        rows = self._build_rows()

        table = Table(show_header=True, header_style="bold blue", box=None)
        table.add_column("Event", justify="left", style="cyan")
        table.add_column("Last", justify="right")
        table.add_column("p50(100)", justify="right")
        table.add_column("p95(100)", justify="right")
        table.add_column("Avg(100)", justify="right")
        table.add_column("Trend", justify="center")
        table.add_column("Device", justify="center", style="magenta")

        # New DDP columns
        table.add_column("Worst", justify="center")
        # table.add_column("Cov", justify="center")
        # table.add_column("MinN", justify="right")

        if not rows:
            table.add_row(
                "[dim]No step timers recorded[/dim]",
                "—",
                "—",
                "—",
                "—",
                "",
                "—",
                "—",
                # "—", "—",
            )
        else:
            for r in rows:
                table.add_row(
                    f"[bold]{r.name}[/bold]",
                    fmt_time_run(r.last),
                    fmt_time_run(r.p50_100),
                    fmt_time_run(r.p95_100),
                    fmt_time_run(r.avg_100),
                    r.trend,
                    r.device,
                    r.worst_rank,
                    # r.coverage,
                    # str(r.min_samples) if r.min_samples else "—",
                )

        cols, _ = shutil.get_terminal_size()
        width = min(max(110, int(cols * 0.90)), 140)

        return Panel(
            Group(table),
            title="[bold blue]User Time (stats = max over DDP ranks)[/bold blue]",
            border_style="blue",
            width=width,
        )

    def get_notebook_renderable(self) -> HTML:
        rows = self._build_rows()

        if not rows:
            body = """
            <tr>
                <td colspan="10" style="text-align:center;color:gray;">
                    No user timers recorded
                </td>
            </tr>
            """
        else:
            body = ""
            for r in rows:
                trend_text = "—"
                trend_color = "#666"

                if isinstance(r.trend, str) and r.trend:
                    if r.trend.startswith("!"):
                        trend_text = r.trend
                        trend_color = "#f57c00"  # orange
                    elif r.trend.startswith("+"):
                        trend_text = f"↑ {r.trend}"
                        trend_color = "#d32f2f"  # red (regression)
                    elif r.trend.startswith("-"):
                        trend_text = f"↓ {r.trend}"
                        trend_color = "#2e7d32"  # green (improvement)
                    elif "≈" in r.trend:
                        trend_text = r.trend
                        trend_color = "#666"

                body += f"""
                <tr>
                    <td>{r.name}</td>
                    <td style="text-align:right;">{fmt_time_run(r.last)}</td>
                    <td style="text-align:right;">{fmt_time_run(r.p50_100)}</td>
                    <td style="text-align:right;">{fmt_time_run(r.p95_100)}</td>
                    <td style="text-align:right;">{fmt_time_run(r.avg_100)}</td>
                    <td style="
                        color:{trend_color};
                        font-weight:700;
                        text-align:center;
                    ">{trend_text}</td>
                    <td style="text-align:center;">{r.device}</td>
                    <td style="text-align:center;">{r.worst_rank}</td>
                    <td style="text-align:center;">{r.coverage}</td>
                    <td style="text-align:right;">{r.min_samples if r.min_samples else "—"}</td>
                </tr>
                """

        table_html = f"""
        <table style="
            width:100%;
            border-collapse:collapse;
            font-size:13px;
        ">
            <thead>
                <tr style="border-bottom:1px solid #e0e0e0;">
                    <th align="left">Event</th>
                    <th align="right">Last</th>
                    <th align="right">p50(100)</th>
                    <th align="right">p95(100)</th>
                    <th align="right">Avg(100)</th>
                    <th align="center">Trend</th>
                    <th align="center">Device</th>
                    <th align="center">Worst</th>
                    <th align="center">Cov</th>
                    <th align="right">MinN</th>
                </tr>
            </thead>
            <tbody>
                {body}
            </tbody>
        </table>
        """

        return HTML(
            f"""
        <div style="{CARD_STYLE}">
            <h4 style="color:#d47a00;margin:0 0 10px 0;">
                User Timings (DDP worst-rank aggregation)
            </h4>
            {table_html}
        </div>
        """
        )

    def get_dashboard_renderable(self):
        pass

    def log_summary(self, path: Optional[str] = None) -> None:
        pass
