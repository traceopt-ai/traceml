from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from typing import Dict, Any, Callable, Optional
from traceml.loggers.error_log import get_error_logger
from traceml.renderers.display.stdout_stderr_capture import StreamCapture
from traceml.renderers.display.layout import (
    ROOT_LAYOUT, SYSTEM_LAYOUT, PROCESS_LAYOUT,
    LAYER_COMBINED_MEMORY_LAYOUT, ACTIVATION_GRADIENT_LAYOUT,
    STEPTIMER_LAYOUT, STDOUT_STDERR_LAYOUT,
    LAYER_COMBINED_TIMER_LAYOUT
)

class CLIDisplayManager:
    """
    Manages a single shared Rich Live display and a dynamic Layout for all stdout loggers.
    """

    _console: Console = Console()
    _live_display: Optional[Live] = None
    _layout: Layout = Layout(name=ROOT_LAYOUT)

    # Key: layout_panel_name (e.g., "live_metrics")
    # Value: Callable[[], Renderable] - a function that returns the latest renderable panel
    _layout_content_fns: Dict[str, Callable[[], Any]] = {}
    _active_logger_count: int = 0

    logger = get_error_logger("CLIDisplayManager")

    @classmethod
    def _create_initial_layout(cls):
        """
        Defines the improved structure of the Rich Layout with flexible ratios.
        """
        cls._layout.split_column(
            Layout(name="dashboard", ratio=4),
            Layout(name=STDOUT_STDERR_LAYOUT, ratio=1),
        )
        dashboard = cls._layout["dashboard"]
        dashboard.split_column(
            Layout(name="upper_row", ratio=1),
            Layout(name="model_row", ratio=4),
            Layout(name="bottom_row", ratio=1),
        )
        dashboard["upper_row"].split_row(
            Layout(name=SYSTEM_LAYOUT, ratio=1),
            Layout(name=PROCESS_LAYOUT, ratio=1),
        )
        dashboard["model_row"].split_row(
            Layout(name=LAYER_COMBINED_MEMORY_LAYOUT, ratio=15),
            Layout(name=LAYER_COMBINED_TIMER_LAYOUT, ratio=14),
        )
        dashboard["bottom_row"].split_row(
            Layout(name=ACTIVATION_GRADIENT_LAYOUT, ratio=1),
            Layout(name=STEPTIMER_LAYOUT, ratio=1),
        )

        # Initialize panels with placeholder text
        dashboard[SYSTEM_LAYOUT].update(
            Panel(Text("Waiting for System Metrics...", justify="center"))
        )
        dashboard[PROCESS_LAYOUT].update(
            Panel(Text("Waiting for Process Metrics...", justify="center"))
        )
        dashboard[LAYER_COMBINED_MEMORY_LAYOUT].update(
            Panel(Text("Waiting for Layer Memory...", justify="center"))
        )
        dashboard[LAYER_COMBINED_TIMER_LAYOUT].update(
            Panel(Text("Waiting for Layer Timing...", justify="center"))
        )
        dashboard[ACTIVATION_GRADIENT_LAYOUT].update(
            Panel(Text("Waiting for Activation + Gradient...", justify="center"))
        )
        dashboard[STEPTIMER_LAYOUT].update(
            Panel(Text("Waiting for Step Timers...", justify="center"))
        )
        cls._layout[STDOUT_STDERR_LAYOUT].update(
            Panel(
                Text("Waiting for stdout/stderr...", justify="center"),
                title="Logs",
                border_style="cyan",
            )
        )

    @classmethod
    def start_display(cls):
        """Starts the shared display if not already running."""
        if cls._active_logger_count == 0:
            cls._create_initial_layout()
            cls._live_display = Live(
                cls._layout,
                console=cls._console,
                auto_refresh=False,
                transient=False,
                screen=True,
            )
            try:
                cls._live_display.start()
                StreamCapture.redirect_to_capture()
            except Exception as e:
                cls.logger.error(f"[TraceML] Failed to start shared live display: {e}")
                cls._live_display = None

        cls._active_logger_count += 1

    @classmethod
    def stop_display(cls):
        """Stops the shared Rich Live display."""
        if cls._live_display:
            try:
                cls._live_display.stop()
                StreamCapture.redirect_to_original()
            except Exception as e:
                cls.logger.error(f"[TraceML] Error stopping live display: {e}")
            finally:
                cls._live_display = None
                cls._layout_content_fns.clear()
                # Re-initialize layout to reset state for next run
                cls._layout = Layout(name=ROOT_LAYOUT_NAME)

    @classmethod
    def release_display(cls):
        """Decrements active logger count and stops display if none remain."""
        cls._active_logger_count = max(cls._active_logger_count - 1, 0)
        if cls._active_logger_count == 0:
            cls.stop_display()

    @classmethod
    def register_layout_content(
        cls, layout_section: str, content_fn: Callable[[], Any]
    ):
        """
        Registers a function that provides content for a specific layout panel.
        """
        if cls._layout.get(layout_section) is None:
            cls.logger.error(
                f"[TraceML] WARNING: Layout panel '{layout_section}' not found. Cannot register content."
            )
            return
        cls._layout_content_fns.setdefault(layout_section, content_fn)

    @classmethod
    def update_display(cls):
        """
        Triggers an update of the entire live display by calling all registered
        content functions and updating the layout.
        """
        pass
        if cls._live_display is None:
            return

        try:
            for section_name, content_fn in cls._layout_content_fns.items():
                try:
                    renderable = content_fn()
                    if renderable is not None:
                        cls._layout[section_name].update(renderable)
                except Exception as e:
                    error_panel = Panel(
                        f"[red]Error rendering {section_name}: {e}[/red]",
                        title=f"[bold red]Render Error: {section_name}[/bold red]",
                        border_style="red",
                    )
                    cls._layout[section_name].update(error_panel)
                    cls.logger.error(
                        f"[TraceML] Error in rendering content for panel {section_name}: {e}"
                    )

            StreamCapture.redirect_to_original()
            cls._live_display.refresh()
            StreamCapture.redirect_to_capture()
        except Exception as e:
            cls.logger.error(f"[TraceML] Error updating live display: {e}")
