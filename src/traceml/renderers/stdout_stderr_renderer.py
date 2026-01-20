from rich.panel import Panel
from rich.text import Text
from IPython.display import HTML
from itertools import islice

from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.display.cli_display_manager import STDOUT_STDERR_LAYOUT
from traceml.database.database import Database


class StdoutStderrRenderer(BaseRenderer):
    """
    Renderer for stdout/stderr lines stored by StdoutStderrSampler.
    Rank-0 only.
    """

    def __init__(
        self,
        database: Database,
        display_lines: int = 5,
    ):
        super().__init__(
            name="Stdout/Stderr",
            layout_section_name=STDOUT_STDERR_LAYOUT,
        )
        self.db = database
        self.display_lines = display_lines
        self._table = self.db.create_or_get_table("stdout_stderr")


    @staticmethod
    def _tail(dq, n):
        if not dq:
            return []
        return list(islice(dq, max(0, len(dq) - n), len(dq)))


    def get_panel_renderable(self) -> Panel:
        rows = self._tail(self._table, self.display_lines)

        if not rows:
            content = Text(
                "Waiting for stdout/stderr...", style="dim", justify="center"
            )
        else:
            lines = [Text(r["line"], style="white") for r in rows]
            content = Text("\n").join(lines)

        return Panel(
            content,
            title="[bold cyan]STDOUT / STDERR (RANK 0)[/bold cyan]",
            border_style="cyan",
        )


    def get_notebook_renderable(self) -> HTML:
        rows = self._tail(self._table, self.display_lines)

        if not rows:
            html = "<div style='opacity:0.6;'>Waiting for stdout/stderr…</div>"
        else:
            escaped = "<br>".join(r["line"] for r in rows)
            html = f"""
                <div style="
                    font-family:monospace;
                    white-space:pre-wrap;
                    background:#111;
                    color:#eee;
                    padding:8px;
                    border-radius:6px;
                ">
                    {escaped}
                </div>
            """

        return HTML(html)

    def log_summary(self, path) -> None:
        pass
