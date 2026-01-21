import shutil
from typing import Dict, Any, Optional, Iterable, Mapping, List, Tuple
from collections.abc import Iterable as ABCIterable
import numpy as np

from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from IPython.display import HTML

from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.display.cli_display_manager import PROCESS_LAYOUT
from traceml.utils.formatting import fmt_percent, fmt_mem_new
from traceml.database.database import Database
from traceml.database.remote_database_store import RemoteDBStore
from .utils import CARD_STYLE


class ProcessRenderer(BaseRenderer):
    """
    Process metrics renderer (single-process + DDP with RemoteDBStore).

    This renderer reads rows produced by ProcessSampler (local + remote ranks)
    and exposes three views:
      1) CLI panel (rich)
      2) Notebook HTML card
      3) Dashboard payload (dict)

    Why a separate "Process" panel?
      - These are *process-local* metrics (CPU%, RSS RAM, torch CUDA allocator usage).
      - In DDP, every rank is its own process; rank-0 can optionally aggregate telemetry
        from other ranks via RemoteDBStore.

    Expected ProcessSampler row schema (flat fields):
      {
        "timestamp": float,
        "pid": int,
        "cpu_logical_core_count": int,
        "cpu_percent": float,            # process cpu%
        "ram_used": float,               # RSS bytes
        "ram_total": float,              # system total bytes (for context)
        "gpu_available": bool,
        "gpu_count": int,
        "gpu_device_index": int | None,  # device used by this process
        "gpu_mem_used": float,           # bytes (torch.cuda.memory_allocated)
        "gpu_mem_reserved": float,       # bytes (torch.cuda.memory_reserved)
        "gpu_mem_total": float,          # bytes (device total memory)
        # optionally: rank/local_rank/world_size ...
      }

    Aggregation semantics when remote_store is provided (rank-0 dashboard):
      - CPU/RAM: median across ranks (stable "typical" process)
      - GPU memory: max across ranks (OOM-risk signal)
      - GPU imbalance: max(gpu_used) - min(gpu_used) across ranks

    Summary semantics (compute_summary):
      - Pools samples across all ranks (local + remote) and computes p50/p95/peak.
      - This is "distribution across all process samples received", not time-aligned
        "max-across-ranks per timestep".
    """

    # RemoteDBStore keys must match sender payload:
    # message = {"rank": int, "sampler": str, "tables": {"process": [rows...]}}
    REMOTE_SAMPLER_NAME = "ProcessSampler"
    TABLE_NAME = "process"

    def __init__(self, database: Database, remote_store: Optional[RemoteDBStore] = None):
        super().__init__(name="Process", layout_section_name=PROCESS_LAYOUT)
        self.db = database
        self.remote_store = remote_store

        # local table reference (deque-like)
        self._table = self.db.create_or_get_table(self.TABLE_NAME)

    # Table/row utilities (robust to list, deque, or dict-of-deques)
    @staticmethod
    def _iter_rows(obj) -> Iterable[Dict[str, Any]]:
        """
        Yield row dicts from:
          - a deque/list of rows
          - a mapping {table_name -> deque/list of rows}

        This makes the renderer resilient to how Database stores tables internally.
        """
        if obj is None:
            return
        if isinstance(obj, Mapping):
            for _, table in obj.items():
                if table:
                    for row in table:
                        if isinstance(row, dict):
                            yield row
            return
        if isinstance(obj, ABCIterable):
            for row in obj:
                if isinstance(row, dict):
                    yield row

    @staticmethod
    def _safe_float(x, default: float = 0.0) -> float:
        try:
            if x is None:
                return float(default)
            return float(x)
        except Exception:
            return float(default)

    def _latest_row(self, db: Database, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Return the latest row dict from db.get_last_record(table_name), if available.
        """
        try:
            return db.get_last_record(table_name)
        except Exception:
            return None


    # Remote iteration pattern
    def _iter_rank_dbs(self, local_db: Database, sampler_name: str):
        """
        Yield (rank, Database) pairs:
          - rank 0 is always local_db
          - if remote_store is provided, yield remote rank DBs too
        """
        yield 0, local_db

        if not self.remote_store:
            return

        for rank in self.remote_store.ranks():
            r = int(rank)
            db = self.remote_store.get_db(r, sampler_name)
            if db is not None:
                yield r, db

    def _collect_latest_per_rank(self) -> Dict[int, Dict[str, Any]]:
        """
        Get the latest process row per rank (local + remote ranks).
        """
        out: Dict[int, Dict[str, Any]] = {}
        for rank, db in self._iter_rank_dbs(self.db, self.REMOTE_SAMPLER_NAME):
            row = self._latest_row(db, self.TABLE_NAME)
            if row:
                out[rank] = row
        return out


    # Snapshot computation (used by CLI + notebook + dashboard)
    def _compute_snapshot(self) -> Dict[str, Any]:
        """
        Compute the latest process snapshot.

        If remote telemetry exists, aggregates across ranks as described in class docstring.
        """
        per_rank = self._collect_latest_per_rank()
        if not per_rank:
            return self._empty_snapshot()

        cpu_vals: List[float] = []
        core_vals: List[float] = []
        ram_vals: List[float] = []
        ram_totals: List[float] = []

        gpu_used: List[float] = []
        gpu_reserved: List[float] = []
        gpu_total: List[float] = []
        gpu_dev: List[Any] = []

        for _, row in per_rank.items():
            cpu_vals.append(self._safe_float(row.get("cpu_percent", 0.0), 0.0))
            core_vals.append(self._safe_float(row.get("cpu_logical_core_count", 0.0), 0.0))
            ram_vals.append(self._safe_float(row.get("ram_used", 0.0), 0.0))
            ram_totals.append(self._safe_float(row.get("ram_total", 0.0), 0.0))

            if bool(row.get("gpu_available", False)) and row.get("gpu_mem_total", None) is not None:
                gpu_used.append(self._safe_float(row.get("gpu_mem_used", 0.0), 0.0))
                gpu_reserved.append(self._safe_float(row.get("gpu_mem_reserved", 0.0), 0.0))
                gpu_total.append(self._safe_float(row.get("gpu_mem_total", 0.0), 0.0))
                gpu_dev.append(row.get("gpu_device_index", None))

        n_ranks = len(per_rank)

        # CPU/RAM: "typical rank"
        cpu_med = float(np.median(cpu_vals)) if cpu_vals else 0.0
        cores = float(np.max(core_vals)) if core_vals else 0.0

        ram_med = float(np.median(ram_vals)) if ram_vals else 0.0
        ram_total = float(np.max(ram_totals)) if ram_totals else 0.0

        # GPU: "risk / worst rank"
        if gpu_total:
            used_max = float(np.max(gpu_used)) if gpu_used else 0.0
            reserved_max = float(np.max(gpu_reserved)) if gpu_reserved else 0.0
            total_max = float(np.max(gpu_total)) if gpu_total else 0.0

            used_min = float(np.min(gpu_used)) if gpu_used else 0.0
            used_imb = used_max - used_min

            dev_main = None
            if gpu_used and gpu_dev:
                try:
                    idx = int(np.argmax(np.asarray(gpu_used)))
                    dev_main = gpu_dev[idx] if idx < len(gpu_dev) else None
                except Exception:
                    dev_main = None
        else:
            used_max = reserved_max = total_max = None
            used_imb = None
            dev_main = None

        return {
            "n_ranks": n_ranks,
            "cpu_used": cpu_med,
            "cpu_logical_core_count": cores,
            "ram_used": ram_med,
            "ram_total": ram_total,
            "gpu_used": used_max,
            "gpu_reserved": reserved_max,
            "gpu_total": total_max,
            "gpu_device_index": dev_main,
            "gpu_used_imbalance": used_imb,
        }

    @staticmethod
    def _empty_snapshot() -> Dict[str, Any]:
        """
        Default snapshot when no records exist.
        """
        return {
            "n_ranks": 0,
            "cpu_used": 0.0,
            "cpu_logical_core_count": 0.0,
            "ram_used": 0.0,
            "ram_total": 0.0,
            "gpu_used": None,
            "gpu_reserved": None,
            "gpu_total": None,
            "gpu_device_index": None,
            "gpu_used_imbalance": None,
        }

    # -------------------------------------------------------------------------
    # CLI panel rendering (Rich)
    # -------------------------------------------------------------------------
    def get_panel_renderable(self) -> Panel:
        snap = self._compute_snapshot()
        n_ranks = int(snap.get("n_ranks", 1) or 1)

        table = Table.grid(padding=(0, 2))
        table.add_column(justify="left", style="white")
        table.add_column(justify="left", style="white")

        cpu_label = "CPU" if n_ranks <= 1 else f"CPU (median/{n_ranks}r)"
        ram_label = "RAM" if n_ranks <= 1 else f"RAM (median/{n_ranks}r)"

        table.add_row(
            f"[bold green]{cpu_label}[/bold green] {fmt_percent(snap.get('cpu_used', 0.0))}",
            f"[bold green]{ram_label}[/bold green] {fmt_mem_new(snap.get('ram_used', 0.0))}",
        )

        table.add_row(" ")

        if snap.get("gpu_total"):
            gpu_str = (
                f"{fmt_mem_new(snap['gpu_used'])}/"
                f"{fmt_mem_new(snap['gpu_reserved'])}/"
                f"{fmt_mem_new(snap['gpu_total'])}"
            )
            if n_ranks > 1:
                gpu_str += " [dim](max across ranks)[/dim]"
        else:
            gpu_str = "[red]Not available[/red]"

        table.add_row(f"[bold green]GPU MEM (used/reserved/total)[/bold green] {gpu_str}")

        # DDP-only: imbalance is a strong signal for skewed ranks
        if n_ranks > 1 and snap.get("gpu_used_imbalance") is not None:
            table.add_row(
                f"[bold green]GPU used imbalance[/bold green] "
                f"{fmt_mem_new(snap['gpu_used_imbalance'])} [dim](max-min across ranks)[/dim]"
            )

        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(100, int(cols * 0.75)), 100)

        return Panel(
            table,
            title="[bold cyan]Process Metrics[/bold cyan]",
            title_align="center",
            border_style="cyan",
            width=panel_width,
        )

    # -------------------------------------------------------------------------
    # Notebook HTML rendering
    # -------------------------------------------------------------------------
    def get_notebook_renderable(self) -> HTML:
        snap = self._compute_snapshot()
        n_ranks = int(snap.get("n_ranks", 1) or 1)

        if snap.get("gpu_total"):
            gpu_html = f"""
                <div>
                    <b>GPU MEM:</b>
                    {fmt_mem_new(snap['gpu_used'])} /
                    {fmt_mem_new(snap['gpu_reserved'])} /
                    {fmt_mem_new(snap['gpu_total'])}
                    {"<span style='color:#777;'>(max across ranks)</span>" if n_ranks > 1 else ""}
                </div>
            """
        else:
            gpu_html = """
                <div><b>GPU MEM:</b>
                    <span style="color:red;">Not available</span>
                </div>
            """

        imb_html = ""
        if n_ranks > 1 and snap.get("gpu_used_imbalance") is not None:
            imb_html = f"""
                <div><b>GPU used imbalance (max-min):</b> {fmt_mem_new(snap['gpu_used_imbalance'])}</div>
            """

        cpu_label = "CPU" if n_ranks <= 1 else f"CPU (median across {n_ranks} ranks)"
        ram_label = "RAM" if n_ranks <= 1 else f"RAM (median across {n_ranks} ranks)"

        html = f"""
        <div style="{CARD_STYLE}">
            <h4 style="color:#d47a00; margin-top:0;">Process Metrics</h4>
            <div><b>{cpu_label}:</b> {fmt_percent(snap.get('cpu_used', 0.0))}</div>
            <div><b>{ram_label}:</b> {fmt_mem_new(snap.get('ram_used', 0.0))}</div>
            {gpu_html}
            {imb_html}
        </div>
        """
        return HTML(html)

    # -------------------------------------------------------------------------
    # Dashboard payload
    # -------------------------------------------------------------------------
    def get_dashboard_renderable(self) -> Dict[str, Any]:
        """
        Payload for your NiceGUI/Plotly dashboard.

        Returns:
          - snapshot fields
          - local table reference (deque-like)
          - optionally, per-rank "last seen" timestamps if remote_store exists
        """
        snap = self._compute_snapshot()
        snap["table"] = self._table  # local process history (rank 0)

        if self.remote_store:
            snap["remote_last_seen"] = {
                int(r): float(self.remote_store.last_seen(int(r))) for r in self.remote_store.ranks()
            }
        return snap

    # -------------------------------------------------------------------------
    # Summary (pooled across ranks if remote_store exists)
    # -------------------------------------------------------------------------
    def _collect_all_rows_local_and_remote(self) -> List[Dict[str, Any]]:
        """
        Collect all process rows from:
          - local "process" table
          - remote ranks' "process" tables (if remote_store exists)
        """
        rows: List[Dict[str, Any]] = []

        # local
        rows.extend(list(self._iter_rows(self._table)))

        # remote
        if self.remote_store:
            for rank in self.remote_store.ranks():
                db = self.remote_store.get_db(int(rank), self.REMOTE_SAMPLER_NAME)
                if not db:
                    continue
                t = db.create_or_get_table(self.TABLE_NAME)
                rows.extend(list(self._iter_rows(t)))

        return rows

    def compute_summary(self) -> Dict[str, Any]:
        """
        Compute summary stats over pooled process samples.

        Returns keys compatible with your earlier summary style, but updated to
        flat GPU field names:
          - cpu_cores_p50/p95, cpu_logical_core_count
          - ram_used_p95/peak, ram_total
          - is_GPU_available
          - gpu_mem_used_p95_single / peak_single
          - gpu_mem_reserved_peak_single
          - gpu_mem_total_capacity
        """
        rows = self._collect_all_rows_local_and_remote()
        if not rows:
            return {"total_samples": 0}

        cpu_vals_pct: List[float] = []
        cpu_logical_cores: List[float] = []
        ram_vals: List[float] = []
        ram_total_vals: List[float] = []

        gpu_used_vals: List[float] = []
        gpu_reserved_vals: List[float] = []
        gpu_total_vals: List[float] = []

        for row in rows:
            cpu_vals_pct.append(self._safe_float(row.get("cpu_percent", 0.0), 0.0))
            cpu_logical_cores.append(self._safe_float(row.get("cpu_logical_core_count", 0.0), 0.0))

            ram_vals.append(self._safe_float(row.get("ram_used", 0.0), 0.0))
            ram_total_vals.append(self._safe_float(row.get("ram_total", 0.0), 0.0))

            if bool(row.get("gpu_available", False)) and row.get("gpu_mem_total", None) is not None:
                gpu_used_vals.append(self._safe_float(row.get("gpu_mem_used", 0.0), 0.0))
                gpu_reserved_vals.append(self._safe_float(row.get("gpu_mem_reserved", 0.0), 0.0))
                gpu_total_vals.append(self._safe_float(row.get("gpu_mem_total", 0.0), 0.0))

        # Keep legacy behavior: interpret cpu_percent as "cores fraction" by dividing by 100.
        cpu_cores = [v / 100.0 for v in cpu_vals_pct]

        summary: Dict[str, Any] = {
            "total_samples": len(rows),
            "cpu_cores_p50": round(float(np.median(cpu_cores)), 2) if cpu_cores else 0.0,
            "cpu_cores_p95": round(float(np.percentile(cpu_cores, 95)), 2) if cpu_cores else 0.0,
            "cpu_logical_core_count": float(np.max(cpu_logical_cores)) if cpu_logical_cores else 0.0,
            "ram_used_p95": round(float(np.percentile(ram_vals, 95)), 2) if ram_vals else 0.0,
            "ram_used_peak": round(float(np.max(ram_vals)), 2) if ram_vals else 0.0,
            "ram_total": float(np.max(ram_total_vals)) if ram_total_vals else 0.0,
            "is_GPU_available": bool(gpu_used_vals),
        }

        if gpu_used_vals:
            summary.update(
                {
                    "gpu_mem_used_p95_single": round(float(np.percentile(gpu_used_vals, 95)), 2),
                    "gpu_mem_used_peak_single": round(float(np.max(gpu_used_vals)), 2),
                    "gpu_mem_reserved_peak_single": round(float(np.max(gpu_reserved_vals)), 2),
                    "gpu_mem_total_capacity": float(np.max(gpu_total_vals)) if gpu_total_vals else 0.0,
                }
            )

        return summary

    # -------------------------------------------------------------------------
    # Summary printing (Rich)
    # -------------------------------------------------------------------------
    def _proc_cpu_summary(self, t: Table, block: dict) -> None:
        p50 = block.get("cpu_cores_p50", 0.0)
        p95 = block.get("cpu_cores_p95", 0.0)
        cores = block.get("cpu_logical_core_count", 0.0)

        t.add_row(
            "CPU (p50 / p95)",
            "[magenta]|[/magenta]",
            f"{p50:.2f} / {p95:.2f} cores (of {cores:.0f})",
        )

    def _proc_ram_summary(self, t: Table, block: dict) -> None:
        p95 = block.get("ram_used_p95", 0.0)
        peak = block.get("ram_used_peak", 0.0)
        total = block.get("ram_total", 0.0)

        pct_p95 = (p95 / total * 100.0) if total else 0.0
        pct_peak = (peak / total * 100.0) if total else 0.0

        t.add_row(
            "RAM (p95 / peak)",
            "[magenta]|[/magenta]",
            f"{fmt_mem_new(p95)} ({pct_p95:.0f}%) / {fmt_mem_new(peak)} ({pct_peak:.0f}%) "
            f"(total {fmt_mem_new(total)})",
        )

    def _proc_gpu_memory(self, t: Table, block: dict) -> None:
        if not block.get("is_GPU_available", False):
            t.add_row("GPU", "[magenta]|[/magenta]", "[red]Not available[/red]")
            return

        total = block.get("gpu_mem_total_capacity", 0.0)
        used_p95 = block.get("gpu_mem_used_p95_single", 0.0)
        used_peak = block.get("gpu_mem_used_peak_single", 0.0)
        reserved_peak = block.get("gpu_mem_reserved_peak_single", 0.0)

        t.add_row(
            "GPU MEM (p95 / peak)",
            "[magenta]|[/magenta]",
            f"{fmt_mem_new(used_p95)} / {fmt_mem_new(used_peak)} "
            f"(cap {fmt_mem_new(total)}) | reserved peak {fmt_mem_new(reserved_peak)}",
        )

    def log_summary(self, path=None) -> None:
        """
        Print the pooled summary to console. (path kept for API compatibility)
        """
        console = Console()
        summary = self.compute_summary()

        t = Table.grid(padding=(0, 1))
        t.add_column(justify="left", style="magenta")
        t.add_column(justify="center", style="dim", no_wrap=True)
        t.add_column(justify="right", style="white")

        t.add_row("TOTAL PROCESS SAMPLES", "[magenta]|[/magenta]", str(summary.get("total_samples", 0)))

        if summary.get("total_samples", 0) > 0:
            self._proc_cpu_summary(t, summary)
            self._proc_ram_summary(t, summary)
            self._proc_gpu_memory(t, summary)

        console.print(
            Panel(
                t,
                title="[bold magenta]Process - Summary[/bold magenta]",
                border_style="magenta",
            )
        )
