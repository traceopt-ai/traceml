"""
NiceGUI display driver for TraceML Aggregator.

Threading model
---------------
- Aggregator thread:
  - calls tick()
  - computes renderer payloads (NO UI touching)
  - stores latest payloads under a lock

- UI thread:
  - periodically runs _ui_update_loop() via ui.timer(...)
  - snapshots latest payloads under lock
  - applies payloads to UI widgets via registered update functions

Multi-subscriber support
------------------------
A single layout section (e.g. MODEL_COMBINED_LAYOUT) may be displayed on multiple
pages or multiple UI regions.

To support this without recomputing:
- Each layout registers exactly ONE compute callback: layout -> content_fn
- Each layout may have MANY UI subscribers: layout -> [(cards, update_fn), ...]

Compute still happens once per tick per layout.

Important reliability notes
---------------------------
- NiceGUI timers can stop if their callback raises; _ui_update_loop is guarded.
- Timers are typically client/session-bound; we therefore schedule the update
  timer once per client to avoid "stops after reconnect".
- Locks are held only for snapshots / swapping payload dicts:
  - We do NOT compute while holding the lock.
  - We do NOT update UI while holding the lock.

Aggregator contract (required by trace_aggregator)
--------------------------------------------------
- start()
- tick()
- stop()
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

from nicegui import ui

from traceml.aggregator.display_drivers.base import BaseDisplayDriver
from traceml.aggregator.display_drivers.nicegui_sections.pages import define_pages
from traceml.database.remote_database_store import RemoteDBStore
from traceml.renderers.base_renderer import BaseRenderer
from traceml.runtime.settings import TraceMLSettings

from traceml.renderers.layer_combined_memory.renderer import LayerCombinedMemoryRenderer
from traceml.renderers.layer_combined_time.renderer import LayerCombinedTimeRenderer
from traceml.renderers.process.renderer import ProcessRenderer
from traceml.renderers.step_combined.renderer import StepCombinedRenderer
from traceml.renderers.step_memory.renderer import StepMemoryRenderer
from traceml.renderers.system.renderer import SystemRenderer


def _safe(logger: Any, label: str, fn: Callable[[], Any]) -> Any:
    """Best-effort execution helper; logs exceptions and continues."""
    try:
        return fn()
    except Exception as e:
        logger.error(f"[TraceML] {label}: {e}")
        logger.error(traceback.format_exc())
        return None


# (client_id, cards, update_fn)
Subscriber = Tuple[Optional[str], Dict[str, Any], Callable[[Dict[str, Any], Any], None]]


class NiceGUIDisplayDriver(BaseDisplayDriver):
    """
    NiceGUI dashboard display driver.

    This class is intentionally small and table-driven:
    - renderers define "what to compute" per layout via get_dashboard_renderable()
    - pages define "how to display" per layout via subscribe_layout(...)
    """

    def __init__(self, logger: Any, store: RemoteDBStore, settings: TraceMLSettings) -> None:
        self._logger = logger
        self._store = store
        self._settings = settings

        # ---- UI lifecycle ----
        self._ui_started: bool = False
        self._ui_ready: bool = False

        # ---- Compute callbacks (one per layout) ----
        self._layout_content_fns: Dict[str, Callable[[], Any]] = {}

        # ---- UI subscribers (many per layout) ----
        self._layout_subscribers: Dict[str, List[Subscriber]] = {}

        # ---- Shared payload storage ----
        self._latest_data_lock = threading.Lock()
        self.latest_data: Dict[str, Any] = {}

        # ---- Timer tracking (per client/session) ----
        self._timer_clients: set[str] = set()

        # ---- Registration ----
        self._registered: bool = False

        # ---- Renderers ----
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

        # ---- UI server config ----
        self._port: int = 8765
        self._show: bool = True

    # ---------------------------------------------------------------------
    # Aggregator contract
    # ---------------------------------------------------------------------

    def start(self) -> None:
        """
        Start the NiceGUI server in a background thread (best effort).

        This blocks until define_pages(...) sets self._ui_ready = True.
        """
        if self._ui_started:
            return

        self._ui_started = True
        threading.Thread(target=self._start_ui_server, daemon=True).start()

        while not self._ui_ready:
            time.sleep(0.05)

    def tick(self) -> None:
        """
        Called from aggregator thread.

        SAFE:
        - registers compute callbacks once
        - computes payloads and swaps latest_data (no UI calls)
        """
        if not self._ui_ready:
            return

        self._register_once()
        self.update_display()

    def stop(self) -> None:
        """Best-effort cleanup of in-memory state."""
        self.release_display()

    # ---------------------------------------------------------------------
    # UI server thread
    # ---------------------------------------------------------------------

    def _start_ui_server(self) -> None:
        """
        Runs inside a background thread.

        define_pages(self) should:
        - build UI
        - subscribe layouts via subscribe_layout(...)
        - schedule timer via ensure_ui_timer(...)
        - set self._ui_ready = True
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

    def ensure_ui_timer(self, interval_s: float = 0.75) -> None:
        """
        Ensure the UI update loop is scheduled once per NiceGUI client.

        Why per-client:
        - Timers created in page contexts are often bound to the client/session.
        - If a session reloads/reconnects, its timer may stop.
        - A global "started" flag can prevent scheduling a new timer and cause
          the dashboard to stop updating.
        """
        client = getattr(ui.context, "client", None)
        client_id = getattr(client, "id", None)
        if not client_id:
            # Called outside a client context; skip scheduling.
            return

        if client_id in self._timer_clients:
            return

        ui.timer(interval_s, self._ui_update_loop)
        self._timer_clients.add(client_id)

    def _ui_update_loop(self) -> None:
        """
        UI-thread update loop.

        - Must never throw (some NiceGUI versions stop timers on exceptions).
        - Must not hold shared locks while touching UI widgets.
        """
        if not self._ui_ready:
            return

        try:
            # Snapshot shared state quickly under lock
            with self._latest_data_lock:
                data_snapshot = dict(self.latest_data)
                subs_snapshot = {
                    layout: list(subs) for layout, subs in self._layout_subscribers.items()
                }

            # Apply updates WITHOUT holding the lock
            for layout, data in data_snapshot.items():
                for _client_id, cards, update_fn in subs_snapshot.get(layout, []):
                    try:
                        update_fn(cards, data)
                    except Exception:
                        # Best-effort: do not crash UI updates if one section fails.
                        for w in cards.values():
                            if hasattr(w, "text"):
                                w.text = "⚠️ Could not update"

        except Exception as e:
            self._logger.error(f"[TraceML] UI update loop failed: {e}")
            self._logger.error(traceback.format_exc())

    # ---------------------------------------------------------------------
    # Registration & subscription
    # ---------------------------------------------------------------------

    def register_layout_content(self, layout_section: str, content_fn: Callable[[], Any]) -> None:
        """
        Register a compute callback for a layout section.

        Notes:
        - Exactly one compute callback per layout is expected.
        - Re-registering overwrites the previous callback.
        """
        self._layout_content_fns[layout_section] = content_fn

    def subscribe_layout(
        self,
        layout_section: str,
        cards: Dict[str, Any],
        update_fn: Callable[[Dict[str, Any], Any], None],
        *,
        replace_for_client: bool = True,
    ) -> None:
        """
        Subscribe a UI region to a layout payload.

        Parameters
        ----------
        layout_section:
            Layout key (e.g. MODEL_COMBINED_LAYOUT).
        cards:
            Dict of UI handles returned by build_*_section().
        update_fn:
            Function that updates those handles from the payload.
        replace_for_client:
            If True, replaces any existing subscriber for (layout, client_id).
            This prevents duplicate subscribers when the page reloads.
        """
        client = getattr(ui.context, "client", None)
        client_id = getattr(client, "id", None)

        subs = self._layout_subscribers.setdefault(layout_section, [])

        if replace_for_client and client_id:
            subs[:] = [s for s in subs if s[0] != client_id]

        subs.append((client_id, cards, update_fn))

    # ---------------------------------------------------------------------
    # Aggregator thread: compute & store latest payloads
    # ---------------------------------------------------------------------

    def update_display(self) -> None:
        """
        Compute latest payloads and store them in latest_data.

        Critical:
        - Do not hold the lock while calling content_fns (they can be slow).
        - Swap under lock in one shot.
        """
        if not self._ui_ready:
            return

        new_data: Dict[str, Any] = {}
        for layout, fn in self._layout_content_fns.items():
            try:
                new_data[layout] = fn()
            except Exception as e:
                new_data[layout] = f"[Error] {e}"

        with self._latest_data_lock:
            self.latest_data = new_data

    def release_display(self) -> None:
        """Clear internal state; safe to call multiple times."""
        if not self._ui_started:
            return

        self._ui_started = False
        self._ui_ready = False

        with self._latest_data_lock:
            self.latest_data.clear()

        self._layout_content_fns.clear()
        self._layout_subscribers.clear()
        self._timer_clients.clear()
        self._registered = False

    def _register_once(self) -> None:
        """Register layout -> renderer compute callback exactly once."""
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