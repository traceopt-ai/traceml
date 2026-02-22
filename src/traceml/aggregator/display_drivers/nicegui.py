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

from __future__ import annotations

from typing import Any, Callable, List

from traceml.database.remote_database_store import RemoteDBStore
from traceml.runtime.settings import TraceMLSettings
from traceml.renderers.base_renderer import BaseRenderer

from traceml.renderers.display.managers.nicegui_display_manager import NiceGUIDisplayManager

# Import renderers here so driver controls which ones exist per mode.
from traceml.renderers.layer_combined_memory.renderer import LayerCombinedMemoryRenderer
from traceml.renderers.layer_combined_time.renderer import LayerCombinedTimeRenderer
from traceml.renderers.process.renderer import ProcessRenderer
from traceml.renderers.step_combined.renderer import StepCombinedRenderer
from traceml.renderers.step_memory.renderer import StepMemoryRenderer
from traceml.renderers.system.renderer import SystemRenderer


def _safe(logger: Any, label: str, fn: Callable[[], Any]) -> Any:
    """Best-effort execution helper; log and continue."""
    try:
        return fn()
    except Exception as e:
        logger.error(f"[TraceML] {label}: {e}")
        return None


class NiceGUIDisplayDriver:
    """
    NiceGUI dashboard display driver.

    Strategy (matches your current NiceGUIDisplayManager design):
    - Register content callbacks ONCE (layout -> fn returning "data" for that card)
    - Each tick: call NiceGUIDisplayManager.update_display(), which stores latest_data thread-safely.
      The UI thread should periodically apply latest_data via update functions.

    Important:
    - Ensure define_pages(cls) schedules _ui_update_loop() (e.g., ui.timer) on the UI side.
      Otherwise values will be stored but not displayed.
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

        self._ui = NiceGUIDisplayManager()
        self._registered = False

        # Dashboard can choose a DIFFERENT renderer set from CLI.
        # For example, omit Stdout/Stderr or add dashboard-specific renderers.
        self._renderers: List[BaseRenderer] = [
            SystemRenderer(remote_store=store),
            ProcessRenderer(remote_store=store),
            LayerCombinedMemoryRenderer(remote_store=store, top_n_layers=settings.num_display_layers),
            LayerCombinedTimeRenderer(remote_store=store, top_n_layers=settings.num_display_layers),
            StepCombinedRenderer(remote_store=store),
            StepMemoryRenderer(remote_store=store),
        ]

    def start(self) -> None:
        _safe(self._logger, "NiceGUI display start failed", self._ui.start_display)

    def stop(self) -> None:
        _safe(self._logger, "NiceGUI display release failed", self._ui.release_display)

    def tick(self) -> None:
        self._register_once()
        _safe(self._logger, "NiceGUI display update failed", self._ui.update_display)

    def _register_once(self) -> None:
        """
        Register renderer callbacks into NiceGUI manager.

        Each renderer used here must implement get_dashboard_renderable() OR you can
        swap to dashboard-specific renderers that do.
        """
        if self._registered:
            return

        for r in self._renderers:
            def register(rr: BaseRenderer = r) -> None:
                fn = getattr(rr, "get_dashboard_renderable", None)
                if fn is None:
                    raise AttributeError(
                        f"{rr.__class__.__name__} missing get_dashboard_renderable() "
                        f"(needed for dashboard display driver)"
                    )
                self._ui.register_layout_content(rr.layout_section_name, fn)

            _safe(self._logger, f"{r.__class__.__name__}.dashboard register failed", register)

        self._registered = True