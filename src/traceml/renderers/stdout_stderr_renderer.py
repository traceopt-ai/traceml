from rich.panel import Panel
from rich.text import Text
from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.display.cli_display_manager import STDOUT_STDERR_LAYOUT
from traceml.renderers.display.stdout_stderr_capture import StreamCapture
import os


class StdoutStderrRenderer(BaseRenderer):
    """
    Simple renderer for captured stdout/stderr.
    Keeps small rolling text buffers in memory and can persist them to a log file.
    """

    def __init__(
        self,
        max_cache_lines: int = 20_000,
        display_lines: int = 5,
        log_dir: str = "./logs",
        log_filename: str = "stdout_stderr.log",
    ):
        super().__init__(
            name="Stdout/Stderr",
            layout_section_name=STDOUT_STDERR_LAYOUT,
        )
        self.max_cache_lines = max_cache_lines
        self.display_lines = display_lines
        self.stdout_stderr_cache = []

        # file log path setup
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_filename)

        # truncate previous run
        with open(self.log_path, "w") as f:
            f.write("[TraceML] New run started\n\n")

    def _update_cache_and_log(self, capture, cache):
        """Update cache and append *only new* lines to log."""
        if not capture:
            return

        text = capture.read_buffer() or ""
        if not text:
            return

        # Split for cache update
        new_lines = [ln for ln in text.splitlines() if ln.strip()]
        if not new_lines:
            return

        # Update rolling cache
        cache += new_lines
        cache[:] = cache[-self.max_cache_lines :]

        # Append only new lines to the log file
        with open(self.log_path, "a", encoding="utf-8") as f:
            for ln in new_lines:
                f.write(ln + "\n")

    def get_panel_renderable(self) -> Panel:
        # Refresh caches from the global captures
        self._update_cache_and_log(
            StreamCapture._stdout_stderr_capture, self.stdout_stderr_cache
        )

        # Prepare visible lines
        visible_lines = self.stdout_stderr_cache[-self.display_lines :]
        if not visible_lines:
            content = Text(
                "Waiting for stdout/stderr...", style="dim", justify="center"
            )
        else:
            lines = [Text(ln, style="white") for ln in visible_lines]
            content = Text("\n").join(lines)

        return Panel(
            content,
            title="[bold cyan]STDOUT / STDERR[/bold cyan]",
            border_style="cyan",
        )

    def log_summary(self) -> None:
        """Persist cached lines at the end of the run."""
        self._update_cache_and_log(
            StreamCapture._stdout_stderr_capture, self.stdout_stderr_cache
        )
        print(f"[TraceML] Stdout/Stderr logs saved to: {self.log_path}")
