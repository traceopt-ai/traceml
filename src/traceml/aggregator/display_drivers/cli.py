"""
Display drivers for TraceML Aggregator.

A display driver is the ONLY component that knows:
- which renderers exist for a given display mode
- how those renderers map into UI sections/layout
- whether it needs a register-once phase
- how to "tick" updates

Aggregator stays UI-agnostic and only calls:
- driver.start()
- driver.tick()
- driver.stop()
"""

from typing import Any, Callable, List

from traceml.database.remote_database_store import RemoteDBStore
from traceml.runtime.settings import TraceMLSettings
from traceml.renderers.base_renderer import BaseRenderer

from traceml.renderers.display.managers.cli_display_manager import CLIDisplayManager

# Import renderers here so driver controls which ones exist per mode.
from traceml.renderers.layer_combined_memory.renderer import LayerCombinedMemoryRenderer
from traceml.renderers.layer_combined_time.renderer import LayerCombinedTimeRenderer
from traceml.renderers.process.renderer import ProcessRenderer
from traceml.renderers.step_combined.renderer import StepCombinedRenderer
from traceml.renderers.step_memory.renderer import StepMemoryRenderer
from traceml.renderers.stdout_stderr_renderer import StdoutStderrRenderer
from traceml.renderers.system.renderer import SystemRenderer


def _safe(logger: Any, label: str, fn: Callable[[], Any]) -> Any:
    """Best-effort execution helper; log and continue."""
    try:
        return fn()
    except Exception as e:
        logger.error(f"[TraceML] {label}: {e}")
        return None


class CLIDisplayDriver:
    """
    CLI display driver (Rich Live).

    Strategy (matches your current CLIDisplayManager design):
    - Register renderer content callbacks ONCE (section -> fn returning Rich renderable)
    - Each tick: call CLIDisplayManager.update_display() which refreshes the Rich layout
    """

    def __init__(
        self,
        logger: Any,
        store: RemoteDBStore,
        settings: TraceMLSettings,
    ) -> None:
        self._logger = logger
        self._store = store
        self._settings = settings

        # Your CLIDisplayManager is classmethod-based; instantiation is harmless.
        self._ui = CLIDisplayManager()
        self._registered = False

        # CLI can pick its own renderer set (including Stdout/Stderr).
        self._renderers: List[BaseRenderer] = [
            SystemRenderer(remote_store=store),
            ProcessRenderer(remote_store=store),
            LayerCombinedMemoryRenderer(remote_store=store, top_n_layers=settings.num_display_layers),
            LayerCombinedTimeRenderer(remote_store=store, top_n_layers=settings.num_display_layers),
            StepCombinedRenderer(remote_store=store),
            StepMemoryRenderer(remote_store=store),
            StdoutStderrRenderer(remote_store=store),  # CLI-only by default
        ]

    def start(self) -> None:
        _safe(self._logger, "CLI display start failed", self._ui.start_display)

    def stop(self) -> None:
        _safe(self._logger, "CLI display release failed", self._ui.release_display)

    def tick(self) -> None:
        self._register_once()
        _safe(self._logger, "CLI display update failed", self._ui.update_display)

    def _register_once(self) -> None:
        """
        Register renderer callbacks into the CLI layout.

        Note: section names are CLI-specific, because the CLI layout is defined by CLIDisplayManager.
        If you want CLI to use a different section naming / grouping, change it here (not in aggregator).
        """
        if self._registered:
            return

        for r in self._renderers:
            # Renderers must expose get_panel_renderable for CLI
            def register(rr: BaseRenderer = r) -> None:
                self._ui.register_layout_content(rr.layout_section_name, rr.get_panel_renderable)

            _safe(self._logger, f"{r.__class__.__name__}.cli register failed", register)

        self._registered = True
