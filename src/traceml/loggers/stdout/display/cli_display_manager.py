from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from typing import Dict, Any, Callable, Optional
from traceml.loggers.error_log import get_error_logger, setup_error_logger
import threading

ROOT_LAYOUT_NAME = "root"
LIVE_METRICS_LAYOUT_NAME = "live_metrics_section"
SYSTEM_PROCESS_LAYOUT_NAME = "system_process_section"
LAYER_COMBINED_SUMMARY_LAYOUT_NAME = "layer_combined_summary_section"
ACTIVATION_GRADIENT_SUMMARY_LAYOUT_NAME = "activation_gradient_summary_section"


class CLIDisplayManager:
    """
    Manages a single shared Rich Live display and a dynamic Layout for all stdout loggers.
    """

    _console: Console = Console()
    _live_display: Optional[Live] = None
    _layout: Layout = Layout(name=ROOT_LAYOUT_NAME)

    # Key: layout_panel_name (e.g., "live_metrics")
    # Value: Callable[[], Renderable] - a function that returns the latest renderable panel
    _layout_content_fns: Dict[str, Callable[[], Any]] = {}
    _active_logger_count: int = 0

    setup_error_logger()
    logger = get_error_logger("CLIDisplayManager")

    @classmethod
    def _create_initial_layout(cls):
        """
        Defines the improved structure of the Rich Layout with flexible ratios.
        """
        cls._layout.split_column(
            Layout(name=SYSTEM_PROCESS_LAYOUT_NAME, ratio=1),
            Layout(name=LAYER_COMBINED_SUMMARY_LAYOUT_NAME, ratio=3),
            Layout(name=ACTIVATION_GRADIENT_SUMMARY_LAYOUT_NAME, ratio=1),
        )

        # Initialize panels with placeholder text
        cls._layout[SYSTEM_PROCESS_LAYOUT_NAME].update(
            Panel(Text("Waiting for System Metrics...", justify="center"))
        )
        cls._layout[LAYER_COMBINED_SUMMARY_LAYOUT_NAME].update(
            Panel(Text("Waiting for Current Model...", justify="center"))
        )
        cls._layout[ACTIVATION_GRADIENT_SUMMARY_LAYOUT_NAME].update(
            Panel(Text("Waiting for Activation + Gradient...", justify="center"))
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
            except Exception as e:
                cls.logger.error(
                    f"[TraceML] Failed to start shared live display: {e}"
                )
                cls._live_display = None

        cls._active_logger_count += 1

    @classmethod
    def stop_display(cls):
        """Stops the shared Rich Live display."""
        if cls._live_display:
            try:
                cls._live_display.stop()
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

            cls._live_display.refresh()
        except Exception as e:
            cls.logger.error(
                f"[TraceML] Error updating live display: {e}",
            )
