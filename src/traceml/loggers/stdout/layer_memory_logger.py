from rich.panel import Panel
from rich.table import Table
from rich.console import Group, Console
from typing import Dict, Any, Optional
import shutil

from .base_logger import BaseStdoutLogger
from .display_manager import MODEL_SUMMARY_LAYOUT_NAME
from traceml.utils.formatting import fmt_mem


class LayerMemoryStdoutLogger(BaseStdoutLogger):
    """
    Compact layer memory logger:
      - Sorts layers by memory (desc)
      - Shows Memory (MB/GB) + % of total
    """

    def __init__(self, top_n: Optional[int] = None):
        """
        Args:
            top_n: If provided, show only top-N layers by memory (else show all).
        """
        super().__init__(name="Layer Memory", layout_section_name=MODEL_SUMMARY_LAYOUT_NAME)
        self._latest_snapshot: Dict[str, Any] = {}
        self.top_n = top_n


    def _truncate(self, s: str, max_len: int = 42) -> str:
        if not isinstance(s, str):
            return str(s)
        return s if len(s) <= max_len else s[: max_len - 1] + "…"


    def _get_panel_renderable(self) -> Panel:
        """
        Live snapshot of current model's memory usage.
        """
        layer_data: Dict[str, float] = self._latest_snapshot.get("layer_memory", {}) or {}
        total_mb = float(self._latest_snapshot.get("total_memory", 0.0) or 0.0)
        model_index = self._latest_snapshot.get("model_index", "—")

        if total_mb <= 0 and layer_data:
            total_mb = float(sum(layer_data.values()))

        # Sort by memory (desc), slice top-N if required
        items = sorted(layer_data.items(), key=lambda kv: float(kv[1]), reverse=True)
        if self.top_n is not None and self.top_n > 0:
            items = items[: self.top_n]

        table = Table(
            show_header=True,
            header_style="bold blue",
            box=None,
            pad_edge=False,
            padding=(0, 1),
        )
        table.add_column("Layer", justify="left", style="magenta")
        table.add_column("Memory", justify="right", style="white", no_wrap=True)
        table.add_column("% of total", justify="right", style="white", no_wrap=True)

        if items:
            for name, mem_mb in items:
                pct = (float(mem_mb) / total_mb * 100.0) if total_mb > 0 else 0.0
                table.add_row(
                    self._truncate(str(name)),
                    fmt_mem(mem_mb),
                    f"{pct:.1f}%",
                )
        else:
            table.add_row("[dim]No layers detected[/dim]", "—", "—")

        title_total = fmt_mem(total_mb)

        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(50, int(cols * 0.5)), 100)

        return Panel(
            Group(table),
            title=f"[bold blue]Model #{model_index}[/bold blue]  •  Total: [white]{title_total}[/white]",
            border_style="blue",
            width=panel_width,
        )

    # ---------- summary ----------
    def log_summary(self, summary: Dict[str, Any]):
        """
        Pretty-print final cumulative summary.
        """
        console = Console()

        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", style="blue")
        table.add_column(justify="center", style="dim", no_wrap=True)
        table.add_column(justify="right", style="white")

        def fmt_val(key: str, value: Any) -> str:
            if value is None:
                return "N/A"
            if "memory" in key:
                try:
                    v = float(value)
                    return fmt_mem(v)
                except Exception:
                    return "N/A"
            try:
                return str(int(value))
            except Exception:
                return str(value)

        keys_to_display = [
            "total_models_seen",
            "total_samples_taken",
            "average_model_memory_mb",
            "peak_model_memory_mb",
        ]

        for key in keys_to_display:
            val = summary.get(key, 0)
            table.add_row(key.replace("_", " ").upper(), "[blue]|[/blue]", fmt_val(key, val))

        panel = Panel(
            table,
            title=f"[bold blue]{self.name} - Summary[/bold blue]",
            border_style="blue",
        )
        console.print(panel)
