from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from typing import Dict, Any, Callable, Optional
import threading
import sys


# Layout section names
ROOT_LAYOUT_NAME = "root"
LIVE_METRICS_LAYOUT_NAME = "live_metrics_section"
SYSTEM_LAYOUT_NAME = "system_metrics_section"
PROCESS_LAYOUT_NAME = "process_metrics_section"
MODEL_SUMMARY_LAYOUT_NAME = "current_model_summary_section"
ACTIVATION_SUMMARY_LAYOUT_NAME = "activation_memory_summary_section"


class StdoutDisplayManager:
    """
    Manages a single shared Rich Live display and a dynamic Layout for all stdout loggers.
    """

    _console: Console = Console()
    _live_display: Optional[Live] = None
    _layout: Layout = Layout(name=ROOT_LAYOUT_NAME)

    # Registry for functions that generate content for specific layout sections
    # Key: layout_panel_name (e.g., "live_metrics")
    # Value: Callable[[], Renderable] - a function that returns the latest renderable for that panel
    _layout_content_fns: Dict[str, Callable[[], Any]] = {}

    # For thread safety if multiple threads update _panel_content_fns
    _lock = threading.Lock()
    _active_logger_count: int = 0

    @classmethod
    def _create_initial_layout(cls):
        """
        Defines the improved structure of the Rich Layout with flexible ratios.
        """
        cls._layout.split_column(
            Layout(name=SYSTEM_LAYOUT_NAME),
            Layout(name=PROCESS_LAYOUT_NAME),
            Layout(name=MODEL_SUMMARY_LAYOUT_NAME),
            Layout(name=ACTIVATION_SUMMARY_LAYOUT_NAME),
        )

        # Initialize panels with placeholder text
        cls._layout[SYSTEM_LAYOUT_NAME].update(
            Panel(Text("Waiting for System Metrics...", justify="center"))
        )
        cls._layout[PROCESS_LAYOUT_NAME].update(
            Panel(Text("Waiting for Process Metrics...", justify="center"))
        )
        cls._layout[MODEL_SUMMARY_LAYOUT_NAME].update(
            Panel(Text("Waiting for Current Model...", justify="center"))
        )
        cls._layout[ACTIVATION_SUMMARY_LAYOUT_NAME].update(
            Panel(Text("Waiting for Activation Memory...", justify="center"))
        )

    @classmethod
    def start_display(cls):
        """Starts the shared Rich Live display if not already running."""
        with cls._lock:
            if cls._active_logger_count == 0:
                cls._create_initial_layout()
                cls._live_display = Live(
                    cls._layout,
                    console=cls._console,
                    auto_refresh=False,  # We'll manage refresh manually
                    transient=False,  # Keep output after stop
                    screen=True,  # Use full screen if possible for better experience
                )
                try:
                    cls._live_display.start()
                    cls._live_display.refresh()
                except Exception as e:
                    print(
                        f"[TraceML] Failed to start shared live display: {e}",
                        file=sys.stderr,
                    )
                    cls._live_display = None  # Reset if failed to start

            cls._active_logger_count += 1

    @classmethod
    def stop_display(cls):
        """Stops the shared Rich Live display."""
        if cls._live_display:
            try:
                cls._live_display.stop()
            except Exception as e:
                print(f"[TraceML] Error stopping live display: {e}", file=sys.stderr)
            finally:
                cls._live_display = None
                cls._layout_content_fns.clear()
                # Re-initialize layout to reset state for next run
                cls._layout = Layout(name=ROOT_LAYOUT_NAME)
                print("[TraceML] Rich live display stopped.", file=sys.stderr)

    @classmethod
    def release_display(cls):
        """Decrements active logger count and stops display if none remain."""
        with cls._lock:
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
        with cls._lock:
            if cls._layout.get(layout_section) is None:
                print(
                    f"[TraceML] WARNING: Layout panel '{layout_section}' not found. Cannot register content.",
                    file=sys.stderr,
                )
                return
            cls._layout_content_fns[layout_section] = content_fn

    @classmethod
    def update_display(cls):
        """
        Triggers an update of the entire live display by calling all registered
        content functions and updating the layout.
        """
        with cls._lock:
            if cls._live_display is None:
                # If display fails to start log to console directly
                # print("Live display not active. Logging directly to console.", file=sys.stderr)
                # Fallback to direct print for each registered panel (simplified)
                # This fallback needs a better design to show *all* current state
                return  # For now, just exit if live display isn't running

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
                        print(
                            f"[TraceML] Error in rendering content for panel {section_name}: {e}",
                            file=sys.stderr,
                        )

                cls._live_display.refresh()  # Only refresh once per update cycle
            except Exception as e:
                print(f"[TraceML] Error updating live display: {e}", file=sys.stderr)
