"""
CLI display driver for TraceML (Rich Live dashboard).

- Keep the aggregator API minimal: start() / tick() / stop()
- Make CLI layout + renderer wiring a single, cohesive unit
"""


from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from traceml.aggregator.display_drivers.base import BaseDisplayDriver
from traceml.database.remote_database_store import RemoteDBStore
from traceml.runtime.settings import TraceMLSettings
from traceml.runtime.stdout_stderr_capture import StreamCapture
from traceml.renderers.base_renderer import BaseRenderer

from traceml.aggregator.display_drivers.layout import (
    ROOT_LAYOUT,
    SYSTEM_LAYOUT,
    PROCESS_LAYOUT,
    LAYER_COMBINED_MEMORY_LAYOUT,
    LAYER_COMBINED_TIMER_LAYOUT,
    MODEL_COMBINED_LAYOUT,
    MODEL_MEMORY_LAYOUT,
    STDOUT_STDERR_LAYOUT,
)

from traceml.renderers.system.renderer import SystemRenderer
from traceml.renderers.process.renderer import ProcessRenderer
from traceml.renderers.layer_combined_memory.renderer import LayerCombinedMemoryRenderer
from traceml.renderers.layer_combined_time.renderer import LayerCombinedTimeRenderer
from traceml.renderers.step_combined.renderer import StepCombinedRenderer
from traceml.renderers.step_memory.renderer import StepMemoryRenderer
from traceml.renderers.stdout_stderr_renderer import StdoutStderrRenderer


def _safe(logger: Any, label: str, fn: Callable[[], Any]) -> Any:
    """Best-effort execution helper; log and continue."""
    try:
        return fn()
    except Exception as e:
        logger.error(f"[TraceML] {label}: {e}")
        return None


@dataclass(frozen=True)
class _SectionBinding:
    """
    A binding from a CLI layout section name -> a callable returning a Rich renderable.
    """
    section: str
    render_fn: Callable[[], Any]


class CLIDisplayDriver(BaseDisplayDriver):
    """
    Rich Live CLI driver.

    Contract used by TraceMLAggregator:
      - start(): start Rich Live display
      - tick(): update all panels
      - stop(): stop display / cleanup

    Renderers:
      - Renderers MUST read only from RemoteDBStore.
      - CLI driver expects renderers to implement get_panel_renderable().
    """

    def __init__(self, logger: Any, store: RemoteDBStore, settings: TraceMLSettings) -> None:
        self._logger = logger
        self._store = store
        self._settings = settings

        self._console = Console()
        self._layout = Layout(name=ROOT_LAYOUT)
        self._live: Optional[Live] = None

        self._registered = False
        self._bindings: List[_SectionBinding] = []

        # CLI chooses its renderer set (can differ from dashboard)
        self._renderers: List[BaseRenderer] = [
            SystemRenderer(remote_store=store),
            ProcessRenderer(remote_store=store),
            LayerCombinedMemoryRenderer(remote_store=store, top_n_layers=settings.num_display_layers),
            LayerCombinedTimeRenderer(remote_store=store, top_n_layers=settings.num_display_layers),
            StepCombinedRenderer(remote_store=store),
            StepMemoryRenderer(remote_store=store),
            StdoutStderrRenderer(remote_store=store),  # CLI-only
        ]

    # -------------------------
    # Lifecycle
    # -------------------------

    def start(self) -> None:
        """Initialize layout and start Rich Live."""
        self._create_initial_layout()
        self._live = Live(
            self._layout,
            console=self._console,
            auto_refresh=False,
            transient=False,
            screen=False,
        )
        _safe(self._logger, "CLI Live.start failed", self._live.start)

    def stop(self) -> None:
        """Stop Rich Live and restore stdout/stderr redirection."""
        if self._live is None:
            return
        try:
            self._live.stop()
            StreamCapture.redirect_to_original()
        except Exception as e:
            self._logger.error(f"[TraceML] CLI Live.stop failed: {e}")
        finally:
            self._live = None
            self._bindings.clear()
            self._registered = False

    def tick(self) -> None:
        """
        Periodic update called by the aggregator loop.
        Updates all bound sections and refreshes the Live display.
        """
        if self._live is None:
            return

        if not self._registered:
            self._register_once()

        self._update_all_sections()
        self._refresh()

    # -------------------------
    # Layout + Wiring
    # -------------------------

    def _create_layout(self) -> Layout:
        """
        Create the Rich layout tree.
        Section names are CLI-specific and defined here.
        """
        self._layout.split_column(
            Layout(name="dashboard", ratio=3),
            Layout(name=STDOUT_STDERR_LAYOUT, ratio=1),
        )

        dashboard = self._layout["dashboard"]
        dashboard.split_column(
            Layout(name="upper_row", ratio=3),
            Layout(name="middle_row", ratio=6),
            Layout(name="layer_row", ratio=5),
        )

        dashboard["upper_row"].split_row(
            Layout(name=SYSTEM_LAYOUT, ratio=4),
            Layout(name=PROCESS_LAYOUT, ratio=5),
        )

        dashboard["layer_row"].split_row(
            Layout(name=LAYER_COMBINED_MEMORY_LAYOUT, ratio=8),
            Layout(name=LAYER_COMBINED_TIMER_LAYOUT, ratio=7),
        )

        dashboard["middle_row"].split_row(
            Layout(name=MODEL_COMBINED_LAYOUT, ratio=3),
            Layout(name=MODEL_MEMORY_LAYOUT, ratio=2),
        )

        return dashboard

    def _create_initial_layout(self) -> None:
        """Initialize the layout with placeholders so the UI is stable before data arrives."""
        dashboard = self._create_layout()

        dashboard[SYSTEM_LAYOUT].update(Panel(Text("Waiting for System Metrics...", justify="center")))
        dashboard[PROCESS_LAYOUT].update(Panel(Text("Waiting for Process Metrics...", justify="center")))
        dashboard[LAYER_COMBINED_MEMORY_LAYOUT].update(Panel(Text("Waiting for Layer Memory...", justify="center")))
        dashboard[LAYER_COMBINED_TIMER_LAYOUT].update(Panel(Text("Waiting for Layer Timing...", justify="center")))
        dashboard[MODEL_MEMORY_LAYOUT].update(Panel(Text("Waiting for Step Memory...", justify="center")))

        self._layout[STDOUT_STDERR_LAYOUT].update(
            Panel(
                Text("Waiting for Stdout/Stderr...", justify="center"),
                title="Logs",
                border_style="cyan",
            )
        )

    def _register_once(self) -> None:
        """
        Create a binding list (section -> render_fn).
        This replaces CLIDisplayManager.register_layout_content.
        """
        bindings: List[_SectionBinding] = []

        # Default behavior: each renderer writes into its declared layout_section_name.
        for r in self._renderers:
            # Explicitly require CLI method
            render_fn = getattr(r, "get_panel_renderable", None)
            if render_fn is None:
                self._logger.error(
                    f"[TraceML] CLI renderer missing get_panel_renderable(): {r.__class__.__name__}"
                )
                continue

            section = getattr(r, "layout_section_name", None)
            if not section:
                self._logger.error(
                    f"[TraceML] CLI renderer missing layout_section_name: {r.__class__.__name__}"
                )
                continue

            # Validate layout contains the section
            if self._layout.get(section) is None:
                self._logger.error(
                    f"[TraceML] CLI layout section not found: {section!r} "
                    f"(renderer={r.__class__.__name__})"
                )
                continue

            bindings.append(_SectionBinding(section=section, render_fn=render_fn))

        self._bindings = bindings
        self._registered = True

    # -------------------------
    # Rendering / Refresh
    # -------------------------

    def _update_all_sections(self) -> None:
        """Call each section render_fn and update the Rich layout panel."""
        for b in self._bindings:
            try:
                renderable = b.render_fn()
                if renderable is not None:
                    self._layout[b.section].update(renderable)
            except Exception as e:
                self._layout[b.section].update(
                    Panel(
                        f"[red]Error rendering {b.section}: {e}[/red]",
                        title=f"[bold red]Render Error: {b.section}[/bold red]",
                        border_style="red",
                    )
                )
                self._logger.error(f"[TraceML] CLI render error in {b.section}: {e}")

    def _refresh(self) -> None:
        """
        Refresh Live display safely w.r.t. stdout/stderr capture.
        """
        if self._live is None:
            return
        try:
            StreamCapture.redirect_to_original()
            self._live.refresh()
        finally:
            StreamCapture.redirect_to_capture()