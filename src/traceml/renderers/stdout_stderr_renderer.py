from __future__ import annotations

from itertools import islice
from typing import Any, List, Dict

from IPython.display import HTML
from rich.panel import Panel
from rich.text import Text

from traceml.database.remote_database_store import RemoteDBStore
from traceml.renderers.base_renderer import BaseRenderer
from traceml.aggregator.display_drivers.layout import (
    STDOUT_STDERR_LAYOUT,
)


class StdoutStderrRenderer(BaseRenderer):
    """
    Renderer for stdout/stderr lines stored by StdoutStderrSampler.

    Rank-0 display only:
    - Reads ONLY from RemoteDBStore (which itself only lives on rank0).
    - Shows only rank 0 logs by default.
    """

    def __init__(
        self,
        remote_store: RemoteDBStore,
        display_lines: int = 50,
        rank: int = 0,
        sampler_name: str = "Stdout/Stderr",
        table_name: str = "stdout_stderr",
    ) -> None:
        super().__init__(
            name="Stdout/Stderr",
            layout_section_name=STDOUT_STDERR_LAYOUT,
        )
        self._store = remote_store
        self._display_lines = int(display_lines)
        self._rank = int(rank)
        self._sampler_name = str(sampler_name)
        self._table_name = str(table_name)

    @staticmethod
    def _tail_rows(rows: Any, n: int) -> List[Dict[str, Any]]:
        """
        Tail helper that works with your Database table (likely a deque/list).
        """
        if not rows or n <= 0:
            return []
        # rows is expected to be sized (deque/list). islice works fine.
        start = max(0, len(rows) - n)
        return list(islice(rows, start, len(rows)))

    def get_panel_renderable(self) -> Panel:
        db = self._store.get_db(
            rank=self._rank, sampler_name=self._sampler_name
        )
        table = (
            None if db is None else db.create_or_get_table(self._table_name)
        )

        tail = (
            self._tail_rows(table, self._display_lines)
            if table is not None
            else []
        )

        if not tail:
            content = Text(
                "Waiting for stdout/stderr...", style="dim", justify="center"
            )
        else:
            # render as one block for speed + simpler wrapping behavior
            text_block = "\n".join(
                str(r.get("line", "")) for r in tail if r.get("line")
            )
            content = Text(text_block)

        return Panel(
            content,
            title=f"[bold cyan]STDOUT / STDERR (RANK {self._rank})[/bold cyan]",
            border_style="cyan",
        )

    def get_notebook_renderable(self) -> HTML:
        return HTML(
            "<pre>Stdout/Stderr renderer disabled in notebook mode.</pre>"
        )

    def log_summary(self, path: str) -> None:
        pass
