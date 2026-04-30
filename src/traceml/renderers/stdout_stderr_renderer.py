from __future__ import annotations

from typing import List

from rich.panel import Panel
from rich.text import Text

from traceml.aggregator.display_drivers.layout import STDOUT_STDERR_LAYOUT
from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.stdout_stderr.common import (
    StdoutStderrDB,
    StdoutStderrLine,
)


class StdoutStderrRenderer(BaseRenderer):
    """
    Renderer for stdout/stderr lines stored by StdoutStderrSampler.

    Current behavior
    ----------------
    - Displays only rank 0 logs by default.
    - Reads from SQLite-backed history instead of RemoteDBStore.
    - Preserves the same user-visible CLI behavior as before.
    """

    def __init__(
        self,
        db_path: str,
        display_lines: int = 50,
        rank: int = 0,
    ) -> None:
        super().__init__(
            name="Stdout/Stderr",
            layout_section_name=STDOUT_STDERR_LAYOUT,
        )
        self._db = StdoutStderrDB(db_path=db_path)
        self._display_lines = int(display_lines)
        self._rank = int(rank)

    @staticmethod
    def _lines_to_text(lines: List[StdoutStderrLine]) -> Text:
        """
        Convert line records into a Rich Text block.
        """
        if not lines:
            return Text(
                "Waiting for stdout/stderr...",
                style="dim",
                justify="center",
            )

        text_block = "\n".join(line.line for line in lines if line.line)
        if not text_block:
            return Text(
                "Waiting for stdout/stderr...",
                style="dim",
                justify="center",
            )

        return Text(text_block)

    def get_panel_renderable(self) -> Panel:
        """
        Render the latest stdout/stderr lines as a Rich panel.
        """
        lines: List[StdoutStderrLine] = []
        try:
            conn = self._db.connect()
            try:
                lines = self._db.fetch_latest_lines(
                    conn,
                    rank=self._rank,
                    limit=self._display_lines,
                )
            finally:
                conn.close()
        except Exception:
            # Best-effort only: renderer must never crash the display loop.
            lines = []

        content = self._lines_to_text(lines)

        return Panel(
            content,
            title=f"[bold cyan]STDOUT / STDERR (RANK {self._rank})[/bold cyan]",
            border_style="cyan",
        )

    def log_summary(self, path: str) -> None:
        """
        Reserved for future summary export support.
        """
        return None
