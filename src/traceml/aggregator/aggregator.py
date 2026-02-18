"""
TraceML aggregator (out-of-process telemetry server + renderer driver).

The aggregator is a standalone component intended to run as a separate process
from training. It is responsible for:

- Receiving telemetry rows from all ranks over TCP
- Ingesting rows into a unified RemoteDBStore
- Building renderers and driving UI updates (CLI / dashboard)

Renderers MUST read ONLY from RemoteDBStore.
"""

import threading
from typing import Any, Callable, Dict, List, Tuple, Type

from traceml.database.remote_database_store import RemoteDBStore
from traceml.renderers.base_renderer import BaseRenderer

from traceml.renderers.display.managers.cli_display_manager import CLIDisplayManager
from traceml.renderers.display.managers.nicegui_display_manager import NiceGUIDisplayManager

from traceml.renderers.layer_combined_memory.renderer import LayerCombinedMemoryRenderer
from traceml.renderers.layer_combined_time.renderer import LayerCombinedTimeRenderer
from traceml.renderers.process.renderer import ProcessRenderer
from traceml.renderers.step_combined.renderer import StepCombinedRenderer
from traceml.renderers.step_memory.renderer import StepMemoryRenderer
from traceml.renderers.stdout_stderr_renderer import StdoutStderrRenderer
from traceml.renderers.system.renderer import SystemRenderer

from traceml.transport.tcp_transport import TCPConfig, TCPServer

from traceml.runtime.runtime import TraceMLSettings  # reuse settings dataclass



_DISPLAY_BACKENDS: Dict[str, Tuple[Type[Any], str]] = {
    "cli": (CLIDisplayManager, "get_panel_renderable"),
    "dashboard": (NiceGUIDisplayManager, "get_dashboard_renderable"),
}


def _safe(logger, label: str, fn: Callable[[], Any]) -> Any:
    """Execute `fn()` and log exceptions; never raise."""
    try:
        return fn()
    except Exception as e:
        logger.error(f"[TraceML] {label}: {e}")
        return None


class TraceMLAggregator:
    """
    Telemetry aggregator + renderer driver.

    Owns:
    - TCPServer (listens for telemetry from all ranks)
    - RemoteDBStore (unified telemetry store; renderers read ONLY this)
    - Display manager + renderers
    """

    def __init__(
        self,
        logger,
        stop_event: threading.Event,
        settings: TraceMLSettings,
    ) -> None:
        self._logger = logger
        self._stop_event = stop_event
        self._settings = settings
        self.mode = settings.mode

        # Unified telemetry store
        self._store = RemoteDBStore(max_rows=int(settings.remote_max_rows))

        # Transport: rank0 TCP server
        self._tcp_server = TCPServer(
            TCPConfig(host=settings.tcp.host, port=int(settings.tcp.port))
        )

        # Display backend
        display_cls, self._render_attr = _DISPLAY_BACKENDS[settings.mode]
        self._display_manager = display_cls()

        # Renderers: MUST read ONLY from RemoteDBStore
        self._renderers = self._build_renderers(
            num_display_layers=settings.num_display_layers,
            remote_store=self._store,
        )

        self._registered = False
        self._thread = threading.Thread(
            target=self._loop,
            name="TraceMLAggregator",
            daemon=True,
        )

    @property
    def store(self) -> RemoteDBStore:
        """Expose store for testing (read-only usage)."""
        return self._store

    def start(self) -> None:
        """
        Start server and aggregator loop.

        Start order matters: the server must be listening before sampler threads
        attempt to flush senders (including rank0 -> rank0).
        """
        _safe(self._logger, "TCPServer.start failed", self._tcp_server.start)
        _safe(
            self._logger,
            "Display start failed",
            self._display_manager.start_display,
        )
        _safe(
            self._logger, "Aggregator thread start failed", self._thread.start
        )

    def stop(self, timeout_sec: float) -> None:
        """Stop aggregator loop and release UI/server resources (best effort)."""
        self._thread.join(timeout=float(timeout_sec))
        if self._thread.is_alive():
            self._logger.error(
                "[TraceML] WARNING: aggregator thread did not terminate"
            )

        _safe(
            self._logger,
            "Display release failed",
            self._display_manager.release_display,
        )
        _safe(self._logger, "TCPServer.stop failed", self._tcp_server.stop)

    def _build_renderers(
        self,
        num_display_layers: int,
        remote_store: RemoteDBStore,
    ) -> List[BaseRenderer]:
        """
        Build renderers that read ONLY from `remote_store`.
        """
        renderers: List[BaseRenderer] = [
            SystemRenderer(remote_store=remote_store),
            ProcessRenderer(remote_store=remote_store),
            LayerCombinedMemoryRenderer(
                remote_store=remote_store, top_n_layers=num_display_layers
            ),
            LayerCombinedTimeRenderer(
                remote_store=remote_store, top_n_layers=num_display_layers
            ),
            # UserTimeRenderer(remote_store=remote_store),
            StepCombinedRenderer(remote_store=remote_store),
            StepMemoryRenderer(remote_store=remote_store),
        ]
        if self.mode == "cli":
            renderers.append(StdoutStderrRenderer(remote_store=remote_store))
        return renderers

    def _register_renderers_once(self) -> None:
        """Register renderer layouts into the display manager once."""
        if self._registered:
            return
        for r in self._renderers:

            def register(rr=r):
                render_fn = getattr(rr, self._render_attr)
                self._display_manager.register_layout_content(
                    rr.layout_section_name,
                    render_fn,
                )

            _safe(
                self._logger,
                f"{r.__class__.__name__}.register failed",
                register,
            )
        self._registered = True

    def _drain_tcp(self) -> None:
        """Drain server messages and ingest them into the store."""
        for msg in self._tcp_server.poll():
            _safe(
                self._logger,
                "RemoteDBStore.ingest failed",
                lambda m=msg: self._store.ingest(m),
            )

    def _loop(self) -> None:
        """Periodic drain + render loop."""
        self._register_renderers_once()

        while not self._stop_event.is_set():
            self._drain_tcp()
            _safe(
                self._logger,
                "Display update failed",
                self._display_manager.update_display,
            )
            self._stop_event.wait(float(self._settings.render_interval_sec))

        # final update
        self._drain_tcp()
        _safe(
            self._logger,
            "Display update failed",
            self._display_manager.update_display,
        )
