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
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from nicegui import app, ui

from traceml_ai.aggregator.display_drivers.base import BaseDisplayDriver
from traceml_ai.aggregator.display_drivers.nicegui_sections.pages import (
    define_pages,
)
from traceml_ai.aggregator.display_drivers.server_readiness import (
    ServerReadiness,
    socket_is_listening,
    wait_for_server_ready,
)
from traceml_ai.aggregator.display_drivers.staleness import format_staleness
from traceml_ai.database.remote_database_store import RemoteDBStore
from traceml_ai.renderers.base_renderer import DashboardRenderer
from traceml_ai.renderers.layer_combined_memory.renderer import (
    LayerCombinedMemoryRenderer,
)
from traceml_ai.renderers.layer_combined_time.renderer import (
    LayerCombinedTimeRenderer,
)
from traceml_ai.renderers.model_diagnostics.renderer import (
    ModelDiagnosticsRenderer,
)
from traceml_ai.renderers.process.renderer import ProcessRenderer
from traceml_ai.renderers.step_memory.renderer import StepMemoryRenderer
from traceml_ai.renderers.step_time.renderer import StepCombinedRenderer
from traceml_ai.renderers.system.renderer import SystemRenderer
from traceml_ai.runtime.settings import TraceMLSettings

# Max seconds start() waits to confirm the dashboard server is listening before
# returning anyway. Training must never block on the dashboard (TRA-68).
_SERVER_STARTUP_TIMEOUT_SEC = 10.0


@dataclass
class LayoutError:
    """A section compute error captured as a payload so the UI can show it."""

    message: str


def render_error(cards: Dict[str, Any], message: str) -> None:
    """Write a short error message into every text-bearing card widget."""
    text = f"⚠️ {message}"
    for widget in cards.values():
        if hasattr(widget, "text"):
            widget.text = text


def _safe(logger: Any, label: str, fn: Callable[[], Any]) -> Any:
    """Best-effort execution helper; logs exceptions and continues."""
    try:
        return fn()
    except Exception as e:
        logger.error(f"[TraceML] {label}: {e}")
        logger.error(traceback.format_exc())
        return None


# (client_id, cards, update_fn)
Subscriber = Tuple[
    Optional[str], Dict[str, Any], Callable[[Dict[str, Any], Any], None]
]


class NiceGUIDisplayDriver(BaseDisplayDriver):
    """
    NiceGUI dashboard display driver.

    This class is intentionally small and table-driven:
    - renderers define "what to compute" per layout via get_dashboard_renderable()
    - pages define "how to display" per layout via subscribe_layout(...)
    """

    def __init__(
        self, logger: Any, store: RemoteDBStore, settings: TraceMLSettings
    ) -> None:
        self._logger = logger
        self._store = store
        self._settings = settings
        self._deep_profile = (
            getattr(self._settings, "profile", "run") == "deep"
        )

        # ---- UI lifecycle ----
        self._ui_started: bool = False
        self._ui_ready: bool = False  # a browser has rendered a page

        # ---- Server readiness (TRA-68): start() waits for the socket, never
        # for a browser, and always returns within the timeout. ----
        self._lifespan_started = threading.Event()
        self._server_thread: Optional[threading.Thread] = None
        self._startup_timeout_sec: float = _SERVER_STARTUP_TIMEOUT_SEC

        # ---- Staleness (TRA-68): seconds since payloads last refreshed,
        # computed UI-side so a wedged display loop stops looking fresh. ----
        self._last_data_monotonic: Optional[float] = None
        self._staleness_threshold_sec: float = 5.0
        self._staleness_labels: Dict[Optional[str], Any] = {}

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
        self._renderers: List[DashboardRenderer] = [
            SystemRenderer(db_path=self._settings.db_path),
            ProcessRenderer(db_path=self._settings.db_path),
            StepCombinedRenderer(db_path=self._settings.db_path),
            StepMemoryRenderer(db_path=self._settings.db_path),
            ModelDiagnosticsRenderer(db_path=self._settings.db_path),
        ]

        if self._deep_profile:
            self._renderers += [
                LayerCombinedMemoryRenderer(
                    remote_store=store,
                    top_n_layers=settings.num_display_layers,
                ),
                LayerCombinedTimeRenderer(
                    remote_store=store,
                    top_n_layers=settings.num_display_layers,
                ),
            ]

        # ---- UI server config (resolved via traceml.yaml/env/CLI) ----
        self._port: int = settings.dashboard_port
        self._show: bool = settings.dashboard_auto_open

    # ---------------------------------------------------------------------
    # Aggregator contract
    # ---------------------------------------------------------------------

    def start(self) -> None:
        """
        Start the NiceGUI server in a background daemon thread (best effort).

        Waits for the server socket to start listening (NOT for a browser to
        connect) and always returns within _startup_timeout_sec, so a headless
        or remote training run is never blocked (TRA-68). _ui_ready stays a
        pure compute gate: tick() computes only once a browser has rendered.
        """
        if self._ui_started:
            return

        self._ui_started = True

        # Deterministic pre-check: if the port is already held, our bind is
        # doomed -- report it clearly instead of starting a server that will
        # die, and avoid mistaking the foreign listener for our own.
        if socket_is_listening("127.0.0.1", self._port):
            self._log_startup_result(ServerReadiness.FAILED)
            return

        self._lifespan_started.clear()
        self._server_thread = threading.Thread(
            target=self._start_ui_server, daemon=True
        )
        self._server_thread.start()

        result = wait_for_server_ready(
            is_listening=lambda: socket_is_listening("127.0.0.1", self._port),
            is_alive=self._server_thread.is_alive,
            lifespan_started=self._lifespan_started.is_set,
            timeout=self._startup_timeout_sec,
        )
        self._log_startup_result(result)

    def _log_startup_result(self, result: ServerReadiness) -> None:
        url = f"http://localhost:{self._port}"
        if result is ServerReadiness.READY:
            self._logger.info(f"[TraceML] Dashboard ready at {url}")
        elif result is ServerReadiness.FAILED:
            self._logger.error(
                f"[TraceML] Dashboard server failed to start on port "
                f"{self._port} (is the port already in use?). Training "
                f"continues without the dashboard."
            )
        else:  # TIMEOUT
            self._logger.warning(
                f"[TraceML] Dashboard not confirmed within "
                f"{self._startup_timeout_sec:.0f}s; continuing. It may still "
                f"come up at {url}."
            )

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
            # Fires when our ASGI lifespan starts (just before the socket is
            # bound); gates the readiness probe in start().
            app.on_startup(self._lifespan_started.set)
            # Reclaim a client's subscribers + timer when its browser leaves,
            # so reconnects don't leak (TRA-68).
            app.on_disconnect(self._handle_disconnect)
            ui.run(
                port=self._port,
                reload=False,
                show=self._show,
                title="TraceML Dashboard",
            )
        except BaseException as e:  # uvicorn calls sys.exit() on bind failure
            # SystemExit is not an Exception; catching it here turns a port
            # conflict into a logged failure instead of a wedged daemon thread.
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
                    layout: list(subs)
                    for layout, subs in self._layout_subscribers.items()
                }

            # Apply updates WITHOUT holding the lock
            for layout, data in data_snapshot.items():
                for _client_id, cards, update_fn in subs_snapshot.get(
                    layout, []
                ):
                    if isinstance(data, LayoutError):
                        render_error(cards, data.message)
                        continue
                    try:
                        update_fn(cards, data)
                    except Exception:
                        # Best-effort: keep the UI alive if a section fails.
                        render_error(cards, "could not update")

            # Refresh the staleness indicator(s) on every UI tick so a wedged
            # display loop stops looking fresh (computed UI-side, not at swap).
            staleness = self.staleness_text()
            for label in list(self._staleness_labels.values()):
                if hasattr(label, "text"):
                    label.text = staleness

        except Exception as e:
            self._logger.error(f"[TraceML] UI update loop failed: {e}")
            self._logger.error(traceback.format_exc())

    # ---------------------------------------------------------------------
    # Registration & subscription
    # ---------------------------------------------------------------------

    def register_layout_content(
        self, layout_section: str, content_fn: Callable[[], Any]
    ) -> None:
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

    def _handle_disconnect(self, client: Any) -> None:
        """app.on_disconnect handler: prune the disconnected client's state."""
        client_id = getattr(client, "id", None)
        if client_id is not None:
            self._prune_client(client_id)

    def _prune_client(self, client_id: str) -> None:
        """
        Remove a disconnected client's subscribers and timer registration.

        Without this, every browser session leaks subscribers and a ui.timer
        that are never reclaimed, so per-tick work grows without bound across
        reconnects (TRA-68).
        """
        for layout in list(self._layout_subscribers):
            remaining = [
                sub
                for sub in self._layout_subscribers[layout]
                if sub[0] != client_id
            ]
            if remaining:
                self._layout_subscribers[layout] = remaining
            else:
                del self._layout_subscribers[layout]
        self._timer_clients.discard(client_id)
        self._staleness_labels.pop(client_id, None)

    def register_staleness_label(self, label: Any) -> None:
        """Register a per-client widget that shows the staleness indicator."""
        client = getattr(ui.context, "client", None)
        client_id = getattr(client, "id", None)
        self._staleness_labels[client_id] = label

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
                new_data[layout] = LayoutError(str(e))

        with self._latest_data_lock:
            self.latest_data = new_data

        # Mark "fresh" only when real data flowed; all-error ticks leave the
        # timestamp untouched so the staleness indicator advances.
        if any(
            not isinstance(payload, LayoutError)
            for payload in new_data.values()
        ):
            self._last_data_monotonic = time.monotonic()

    def staleness_text(self, now: Optional[float] = None) -> str:
        """Short 'stale Ns' label for the UI, or '' when data is fresh."""
        if self._last_data_monotonic is None:
            return ""
        current = time.monotonic() if now is None else now
        age = current - self._last_data_monotonic
        return format_staleness(age, self._staleness_threshold_sec)

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
        self._staleness_labels.clear()
        self._last_data_monotonic = None
        self._registered = False

    def _register_once(self) -> None:
        """Register layout -> renderer compute callback exactly once."""
        if self._registered:
            return

        for r in self._renderers:

            def register(rr: DashboardRenderer = r) -> None:
                fn = getattr(rr, "get_dashboard_renderable", None)
                if fn is None:
                    raise AttributeError(
                        f"{rr.__class__.__name__} missing get_dashboard_renderable()"
                    )
                self.register_layout_content(rr.layout_section_name, fn)

            _safe(
                self._logger,
                f"{r.__class__.__name__}.dashboard register failed",
                register,
            )

        self._registered = True
