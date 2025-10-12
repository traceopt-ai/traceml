from rich.panel import Panel
from rich.text import Text
from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.display.cli_display_manager import STDOUT_STDERR_LAYOUT_NAME
from traceml.renderers.display.stdout_stderr_capture import StreamCapture


class StdoutStderrRenderer(BaseRenderer):
    """
    Simple renderer for captured stdout/stderr.
    Keeps small rolling text buffers in memory.
    """

    def __init__(self, max_cache_lines: int = 200, display_lines: int = 3):
        super().__init__(
            name="Stdout/Stderr",
            layout_section_name=STDOUT_STDERR_LAYOUT_NAME,
        )
        self.max_cache_lines = max_cache_lines
        self.display_lines = display_lines
        self.stdout_cache = []
        self.stderr_cache = []

    def _update_cache(self, capture, cache):
        text = capture.read_buffer() if capture else ""
        lines = text.splitlines()
        cache += lines[-self.max_cache_lines :]

    def get_panel_renderable(self) -> Panel:
        # Refresh caches from the global captures
        self._update_cache(StreamCapture._stdout_capture, self.stdout_cache)
        self._update_cache(StreamCapture._stderr_capture, self.stderr_cache)

        # Combine and colorize
        lines = [
            Text(ln, style="white") for ln in self.stdout_cache[-self.display_lines :]
        ] + [
            Text(ln, style="bold red") for ln in self.stderr_cache[-self.display_lines :]
        ]

        if not lines:
            content = Text("Waiting for stdout/stderr...", style="dim", justify="center")
        else:
            content = Text("\n").join(lines)

        return Panel(
            content,
            title="[bold cyan]STDOUT / STDERR[/bold cyan]",
            border_style="cyan",
        )

    def log_summary(self, summary):
        pass
