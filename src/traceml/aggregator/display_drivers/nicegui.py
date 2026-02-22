"""
NiceGUI display driver for TraceML Aggregator (merged driver + manager).

This keeps backward-compatible attribute/method names that define_pages(...)
likely expects from the old NiceGUIDisplayManager:

- _ui_started, _ui_ready
- _latest_data_lock
- cards, update_funcs, latest_data
- _layout_content_fns
- _ui_update_loop(), register_layout_content(), update_display(), release_display()

Aggregator contract (required by trace_aggregator):
- start()
- tick()
- stop()
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Callable, Dict, List

from nicegui import ui

from traceml.database.remote_database_store import RemoteDBStore
from traceml.runtime.settings import TraceMLSettings
from traceml.renderers.base_renderer import BaseRenderer

from traceml.aggregator.display_drivers.nicegui_sections.pages import define_pages

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

    Design:
    - UI runs in a background thread via ui.run(...)
    - Aggregator thread calls tick():
        - registers renderer callbacks once (layout -> fn)
        - update_display() stores latest_data under lock
    - UI thread periodically calls _ui_update_loop() (expected to be wired in define_pages)
      to apply latest_data to widgets via update_funcs.

    IMPORTANT:
    - define_pages(self) should set self._ui_ready = True
    - define_pages(self) should schedule ui.timer(..., self._ui_update_loop)
      If it does not, the dashboard may start but never update.
    """

    def __init__(self, logger: Any, store: RemoteDBStore, settings: TraceMLSettings) -> None:
        self._logger = logger
        self._store = store
        self._settings = settings

        # ---- Backward-compatible "manager" state ----
        self._layout_content_fns: Dict[str, Callable[[], Any]] = {}
        self._ui_started: bool = False
        self._ui_ready: bool = False

        self._latest_data_lock = threading.Lock()
        self.cards: Dict[str, Dict[str, Any]] = {}
        self.update_funcs: Dict[str, Callable[[Dict[str, Any], Any], None]] = {}
        self.latest_data: Dict[str, Any] = {}

        # ---- Driver-only state ----
        self._registered: bool = False
        self._renderers: List[BaseRenderer] = [
            SystemRenderer(remote_store=store),
            ProcessRenderer(remote_store=store),
            LayerCombinedMemoryRenderer(
                remote_store=store, top_n_layers=settings.num_display_layers
            ),
            LayerCombinedTimeRenderer(
                remote_store=store, top_n_layers=settings.num_display_layers
            ),
            StepCombinedRenderer(remote_store=store),
            StepMemoryRenderer(remote_store=store),
        ]

        self._port: int = 8765
        self._show: bool = True

    # -------------------------
    # Aggregator contract
    # -------------------------

    def start(self) -> None:
        """Start the NiceGUI server in a background thread (best effort)."""
        if self._ui_started:
            return

        self._ui_started = True
        threading.Thread(target=self._start_ui_server, daemon=True).start()

        # Keep original behavior: wait until define_pages sets _ui_ready.
        while not self._ui_ready:
            time.sleep(0.05)

    def tick(self) -> None:
        """
        Called from aggregator thread.
        SAFE: only computes data + stores it under lock (no UI touches).
        """
        if not self._ui_ready:
            return

        self._register_once()
        self.update_display()

    def stop(self) -> None:
        """Best-effort cleanup of state (server shutdown is not guaranteed)."""
        self.release_display()

    # -------------------------
    # UI server thread
    # -------------------------

    def _start_ui_server(self) -> None:
        """
        Runs in background thread.

        The old logic relied on define_pages(...) to:
        - populate cards/update_funcs
        - set _ui_ready = True
        - schedule ui.timer(..., _ui_update_loop)
        """
        try:
            define_pages(self)
            ui.run(
                port=self._port,
                reload=False,
                show=self._show,
                title="TraceML Dashboard",
            )
        except Exception as e:
            self._logger.error(f"[TraceML] NiceGUI server failed: {e}")
            self._logger.error(traceback.format_exc())
            self._ui_ready = False

    # -------------------------
    # Old "manager" API kept
    # -------------------------

    def _ui_update_loop(self) -> None:
        """
        UI-thread update loop.
        This must be scheduled on the UI thread (usually inside define_pages) via ui.timer.
        """
        if not self._ui_ready:
            return

        with self._latest_data_lock:
            for layout, data in self.latest_data.items():
                try:
                    update_fn = self.update_funcs.get(layout)
                    if update_fn:
                        update_fn(self.cards.get(layout, {}), data)
                except Exception:
                    for label in self.cards.get(layout, {}).values():
                        if hasattr(label, "text"):
                            label.text = "⚠️ Could not update"

    def register_layout_content(self, layout_section: str, content_fn: Callable[[], Any]) -> None:
        self._layout_content_fns[layout_section] = content_fn

    def update_display(self) -> None:
        """Called from aggregator thread → SAFE: stores data only."""
        if not self._ui_ready:
            return
        with self._latest_data_lock:
            for layout, fn in self._layout_content_fns.items():
                try:
                    self.latest_data[layout] = fn()
                except Exception as e:
                    self.latest_data[layout] = f"[Error] {e}"

    def release_display(self) -> None:
        """Clear state; safe to call multiple times."""
        if not self._ui_started:
            return

        self._ui_started = False
        self._ui_ready = False

        with self._latest_data_lock:
            self.latest_data.clear()
            self.cards.clear()
            self.update_funcs.clear()
            self._layout_content_fns.clear()

        self._registered = False

    # -------------------------
    # Driver renderer wiring
    # -------------------------

    def _register_once(self) -> None:
        """Register layout -> renderer callback exactly once."""
        if self._registered:
            return

        for r in self._renderers:
            def register(rr: BaseRenderer = r) -> None:
                fn = getattr(rr, "get_dashboard_renderable", None)
                if fn is None:
                    raise AttributeError(
                        f"{rr.__class__.__name__} missing get_dashboard_renderable()"
                    )
                self.register_layout_content(rr.layout_section_name, fn)

            _safe(self._logger, f"{r.__class__.__name__}.dashboard register failed", register)

        self._registered = True