from collections import deque
from typing import Dict, Any, Optional, List

import numpy as np
import shutil

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
from traceml.loggers.error_log import get_error_logger


class ProcessRenderer(BaseRenderer):
    """
    Renderer for process-level telemetry (CPU, RAM, GPU memory).

    Scope
    -----
    - Rank-0 coordinator only
    - Supports single-process and DDP via RemoteDBStore

    High-level semantics
    --------------------
    1. Live views:
       - Stateless
       - Current-only
       - Pessimistic aggregation (worst rank)

    2. Dashboard history:
       - Incremental aggregation
       - Bounded rolling window
       - Seq alignment

    3. Summary:
       - Computed from rank-0 local data only
       - Remote summary intentionally deferred
    """

    REMOTE_SAMPLER_NAME = "ProcessSampler"
    TABLE_NAME = "process"

    # Dashboard history size (rolling window)
    DASHBOARD_MAX_ROWS = 200

    def __init__(
        self, database: Database, remote_store: Optional[RemoteDBStore] = None
    ):
        super().__init__(name="Process", layout_section_name=PROCESS_LAYOUT)

        self.db = database
        self.remote_store = remote_store

        # local table reference (deque-like)
        self._table = self.db.create_or_get_table(self.TABLE_NAME)

        # Rolling dashboard aggregation (rank-0 only)
        self._dashboard_rollup = deque(maxlen=self.DASHBOARD_MAX_ROWS)

        # Track how many samples we have consumed by all rank
        self._last_completed_seq: int = -1
        self.logger = get_error_logger("ProcessRenderer")

    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default


    def _compute_live_snapshot(self) -> Dict[str, Any]:
        """
        Compute a step-synchronized live snapshot.

        Semantics
        ---------
        - Snapshot is computed only for the latest seq completed by *all* ranks
        - CPU: max across ranks at that seq
        - GPU: rank with least headroom at that seq
        - GPU imbalance: max - min used at that seq
        """

        # ------------------------------------------------------------
        # 1. Collect databases for all active ranks (include local)
        # ------------------------------------------------------------
        rank_dbs: Dict[int, Database] = {0: self.db}

        if self.remote_store:
            for r in self.remote_store.ranks():
                db = self.remote_store.get_db(r, self.REMOTE_SAMPLER_NAME)
                if db:
                    rank_dbs[r] = db

        if not rank_dbs:
            return {}

        # ------------------------------------------------------------
        # 2. Determine last completed seq per rank
        # ------------------------------------------------------------
        last_seq_per_rank: Dict[int, int] = {}

        for rank, db in rank_dbs.items():
            table = db.create_or_get_table(self.TABLE_NAME)
            if not table:
                return {}
            last_seq_per_rank[rank] = table[-1].get("seq", -1)

        # Latest seq completed by *all* ranks
        committed_seq = min(last_seq_per_rank.values())
        if committed_seq < 0:
            return {}

        # ------------------------------------------------------------
        # 3. Fetch rows at committed seq for all ranks
        # ------------------------------------------------------------
        rows_per_rank: Dict[int, Dict[str, Any]] = {}

        for rank, db in rank_dbs.items():
            table = db.create_or_get_table(self.TABLE_NAME)
            row = next(
                (r for r in reversed(table) if r.get("seq") == committed_seq),
                None,
            )
            if row is None:
                return {}  # hard safety: missing data
            rows_per_rank[rank] = row

        # ------------------------------------------------------------
        # 4. Cross-rank aggregation (same logic as dashboard)
        # ------------------------------------------------------------
        cpu_vals: List[float] = []
        gpu_used: List[float] = []
        gpu_reserved: List[float] = []
        gpu_total: List[float] = []
        gpu_rank: List[int] = []

        for rank, row in rows_per_rank.items():
            cpu_vals.append(self._safe_float(row.get("cpu")))

            gpu = row.get("gpu")
            if row.get("gpu_available", False) and gpu:
                used = self._safe_float(gpu.get("mem_used"))
                reserved = self._safe_float(gpu.get("mem_reserved"))
                total = self._safe_float(gpu.get("mem_total"))

                gpu_used.append(used)
                gpu_reserved.append(reserved)
                gpu_total.append(total)
                gpu_rank.append(rank)

        snapshot = {
            "seq": committed_seq,
            "cpu_used": max(cpu_vals) if cpu_vals else 0.0,
        }

        if gpu_total:
            headrooms = [t - r for t, r in zip(gpu_total, gpu_reserved)]
            idx = int(np.argmin(headrooms))

            snapshot.update(
                {
                    "gpu_used": gpu_used[idx],
                    "gpu_reserved": gpu_reserved[idx],
                    "gpu_total": gpu_total[idx],
                    "gpu_rank": gpu_rank[idx],
                    "gpu_used_imbalance": (
                        max(gpu_used) - min(gpu_used)
                        if len(gpu_used) > 1 else 0.0
                    ),
                }
            )

        return snapshot

    def _update_dashboard_rollup(self) -> None:
        """
        Advance dashboard history using globally synchronized seq numbers.

        Invariant
        ---------
        One dashboard entry == one seq that *all active ranks* have completed.
        """

        # ------------------------------------------------------------
        # 1. Collect databases for all active ranks (include local)
        # ------------------------------------------------------------
        rank_dbs: Dict[int, Database] = {0: self.db}

        if self.remote_store:
            for r in self.remote_store.ranks():
                db = self.remote_store.get_db(r, self.REMOTE_SAMPLER_NAME)
                if db:
                    rank_dbs[r] = db

        if not rank_dbs:
            return

        # ------------------------------------------------------------
        # 2. Determine last completed seq per rank
        # ------------------------------------------------------------
        last_seq_per_rank: Dict[int, int] = {}

        for rank, db in rank_dbs.items():
            table = db.create_or_get_table(self.TABLE_NAME)
            if not table:
                return  # at least one rank has no data yet
            last_seq_per_rank[rank] = table[-1].get("seq", -1)

        # Global commit point: all ranks have completed up to this seq
        committed_upto = min(last_seq_per_rank.values())

        if committed_upto <= self._last_completed_seq:
            return

        # ------------------------------------------------------------
        # 3. Aggregate strictly synchronized seqs
        # ------------------------------------------------------------
        for seq in range(self._last_completed_seq + 1, committed_upto + 1):

            rows_per_rank: Dict[int, dict] = {}

            # --- hard synchronization check ---
            for rank, db in rank_dbs.items():
                table = db.create_or_get_table(self.TABLE_NAME)
                row = next(
                    (r for r in reversed(table) if r.get("seq") == seq),
                    None,
                )
                if row is None:
                    # Safety: skip this seq entirely if any rank is missing
                    rows_per_rank = {}
                    break
                rows_per_rank[rank] = row

            if not rows_per_rank:
                continue

            # 4. Cross-rank aggregation
            cpu_vals = []
            ram_used_vals = []
            gpu_used_vals = []
            gpu_candidates = []
            ram_total = 0.0

            for rank, row in rows_per_rank.items():
                cpu_vals.append(self._safe_float(row.get("cpu")))

                ram_used = self._safe_float(row.get("ram_used"))
                ram_used_vals.append(ram_used)
                ram_total = max(ram_total, self._safe_float(row.get("ram_total")))

                gpu = row.get("gpu")
                if row.get("gpu_available", False) and gpu:
                    used = self._safe_float(gpu.get("mem_used"))
                    reserved = self._safe_float(gpu.get("mem_reserved"))
                    total = self._safe_float(gpu.get("mem_total"))

                    headroom = total - reserved
                    gpu_used_vals.append(used)
                    gpu_candidates.append((headroom, rank, used, total))

            entry = {
                "seq": seq,
                "ranks_seen": len(rows_per_rank),
                "cpu_max": max(cpu_vals) if cpu_vals else 0.0,
                "ram_used_max": max(ram_used_vals) if ram_used_vals else 0.0,
                "ram_total": ram_total,
            }

            if gpu_candidates:
                headroom, worst_rank, used, total = min(gpu_candidates)
                entry.update(
                    {
                        "gpu_used": used,
                        "gpu_total": total,
                        "gpu_headroom": headroom,
                        "gpu_rank": worst_rank,
                        "gpu_used_imbalance": (
                            max(gpu_used_vals) - min(gpu_used_vals)
                            if len(gpu_used_vals) > 1
                            else 0.0
                        ),
                    }
                )
            else:
                entry.update(
                    {
                        "gpu_used": None,
                        "gpu_total": None,
                        "gpu_headroom": None,
                        "gpu_rank": None,
                        "gpu_used_imbalance": 0.0,
                    }
                )

            self._dashboard_rollup.append(entry)
        self._last_completed_seq = committed_upto

    # CLI panel rendering (Rich)
    def get_panel_renderable(self) -> Panel:
        snap = self._compute_live_snapshot()

        table = Table.grid(padding=(0, 2))
        table.add_column(justify="left", style="white")
        table.add_column(justify="left", style="white")

        # CPU: worst rank
        table.add_row(
            "[bold green]CPU (worst rank)[/bold green] "
            f"{fmt_percent(snap.get('cpu_used', 0.0))}",
            "",
        )

        table.add_row(" ")

        # GPU memory
        if snap.get("gpu_total") is not None:
            gpu_str = (
                f"{fmt_mem_new(snap['gpu_used'])}/"
                f"{fmt_mem_new(snap['gpu_reserved'])}/"
                f"{fmt_mem_new(snap['gpu_total'])}"
                f" [dim](rank {snap.get('gpu_rank')})[/dim]"
            )
        else:
            gpu_str = "[red]Not available[/red]"

        table.add_row(
            "[bold green]GPU MEM (used/reserved/total)[/bold green]",
            gpu_str,
        )

        # GPU imbalance (only if present)
        if snap.get("gpu_used_imbalance", 0.0) > 0.0:
            table.add_row(
                "[bold green]GPU used imbalance[/bold green]",
                f"{fmt_mem_new(snap['gpu_used_imbalance'])} "
                "[dim](max-min across ranks)[/dim]",
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

    # ------------------------------------------------------------------
    # Dashboard payload
    # ------------------------------------------------------------------
    def get_dashboard_renderable(self) -> Dict[str, Any]:
        self._update_dashboard_rollup()
        snap = self._compute_live_snapshot()
        snap["history"] = list(self._dashboard_rollup)
        return snap

    def get_notebook_renderable(self) -> HTML:
        snap = self._compute_live_snapshot()

        if snap.get("gpu_total") is not None:
            gpu_html = f"""
                <div>
                    <b>GPU MEM (worst rank {snap.get("gpu_rank")}):</b>
                    {fmt_mem_new(snap['gpu_used'])} /
                    {fmt_mem_new(snap['gpu_reserved'])} /
                    {fmt_mem_new(snap['gpu_total'])}
                </div>
            """
        else:
            gpu_html = """
                <div><b>GPU MEM:</b>
                    <span style="color:red;">Not available</span>
                </div>
            """

        imb_html = ""
        if snap.get("gpu_used_imbalance", 0.0) > 0.0:
            imb_html = f"""
                <div>
                    <b>GPU used imbalance (max-min):</b>
                    {fmt_mem_new(snap['gpu_used_imbalance'])}
                </div>
            """

        html = f"""
        <div style="{CARD_STYLE}">
            <h4 style="color:#d47a00; margin-top:0;">Process Metrics</h4>

            <div>
                <b>CPU (worst rank):</b>
                {fmt_percent(snap.get('cpu_used', 0.0))}
            </div>

            {gpu_html}
            {imb_html}
        </div>
        """
        return HTML(html)

    # ------------------------------------------------------------------
    # Summary (rank-0 only)
    # ------------------------------------------------------------------
    def compute_summary(self) -> Dict[str, Any]:
        """
        Compute summary statistics from rank-0 local samples only.
        """
        rows = list(self._table)
        if not rows:
            return {"total_samples": 0}

        cpu_vals = [self._safe_float(r.get("cpu")) / 100.0 for r in rows]
        ram_vals = [r["ram_used"] for r in rows]

        gpu_used = []
        gpu_reserved = []
        gpu_total = []

        for r in rows:
            gpu = r.get("gpu")
            if r.get("gpu_available", False) and gpu:
                gpu_used.append(self._safe_float(gpu.get("mem_used")))
                gpu_reserved.append(self._safe_float(gpu.get("mem_reserved")))
                gpu_total.append(self._safe_float(gpu.get("mem_total")))

        summary = {
            "total_samples": len(rows),
            "cpu_cores_p50": float(np.median(cpu_vals)),
            "cpu_cores_p95": float(np.percentile(cpu_vals, 95)),
            "ram_used_p95": float(np.percentile(ram_vals, 95)),
            "ram_used_peak": float(np.max(ram_vals)),
            "ram_total": float(max(r["ram_total"] for r in rows)),
            "is_GPU_available": bool(gpu_used),
        }

        if gpu_used:
            summary.update(
                {
                    "gpu_mem_used_p95_single": float(np.percentile(gpu_used, 95)),
                    "gpu_mem_used_peak_single": float(np.max(gpu_used)),
                    "gpu_mem_reserved_peak_single": float(np.max(gpu_reserved)),
                    "gpu_mem_total_capacity": float(np.max(gpu_total)),
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

        t.add_row(
            "TOTAL PROCESS SAMPLES",
            "[magenta]|[/magenta]",
            str(summary.get("total_samples", 0)),
        )

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
