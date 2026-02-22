"""
TraceML Aggregator (out-of-process telemetry server + UI driver).

Responsibilities:
- Receive telemetry rows from all ranks over TCP
- Ingest rows into a unified RemoteDBStore
- Drive a display driver (CLI / NiceGUI) that owns renderers + view layout

Key invariants:
- Renderers MUST read ONLY from RemoteDBStore.
- Aggregator MUST NOT know about UI sections, layouts, or renderer methods.
"""

import threading
from typing import Any, Callable, Dict, Type

from traceml.aggregator.display_drivers.base import BaseDisplayDriver
from traceml.database.remote_database_store import RemoteDBStore
from traceml.runtime.settings import TraceMLSettings
from traceml.transport.tcp_transport import TCPConfig, TCPServer

from traceml.aggregator.display_drivers.cli import CLIDisplayDriver
from traceml.aggregator.display_drivers.nicegui import NiceGUIDisplayDriver



def _safe(logger: Any, label: str, fn: Callable[[], Any]) -> Any:
    """
    Execute fn() and never raise.
    Intended for cleanup paths and UI rendering where errors should be logged
    but not crash training/aggregation.
    """
    try:
        return fn()
    except Exception as e:
        logger.error(f"[TraceML] {label}: {e}")
        return None


_DISPLAY_DRIVERS: Dict[str, Type[BaseDisplayDriver]] = {
    "cli": CLIDisplayDriver,
    "dashboard": NiceGUIDisplayDriver,
}


class TraceMLAggregator:
    """
    Telemetry aggregator process.

    Owns:
    - TCPServer: receives messages from training ranks
    - RemoteDBStore: unified telemetry store (single source of truth)
    - Display driver: backend-specific driver that owns renderers + UI updates
    """

    def __init__(
        self,
        logger: Any,
        stop_event: threading.Event,
        settings: TraceMLSettings,
    ) -> None:
        self._logger = logger
        self._stop_event = stop_event
        self._settings = settings

        # Unified telemetry store: renderers read ONLY from here
        self._store = RemoteDBStore(max_rows=int(settings.remote_max_rows))

        # TCP server: aggregator listens on rank0
        self._tcp_server = TCPServer(
            TCPConfig(host=settings.tcp.host, port=int(settings.tcp.port))
        )

        # UI driver (CLI / dashboard). Driver owns renderer selection and layout mapping.
        driver_cls = _DISPLAY_DRIVERS.get(settings.mode)
        if driver_cls is None:
            raise ValueError(
                f"[TraceML] Unknown display mode: {settings.mode!r}. "
                f"Supported: {sorted(_DISPLAY_DRIVERS.keys())}"
            )
        self._display_driver = driver_cls(
            logger=self._logger,
            store=self._store,
            settings=self._settings,
        )

        self._thread = threading.Thread(
            target=self._loop,
            name="TraceMLAggregator",
            daemon=True,
        )

    @property
    def store(self) -> RemoteDBStore:
        """Expose store for tests (read-only usage)."""
        return self._store

    def start(self) -> None:
        """
        Start server + UI driver + aggregator loop.

        Start order matters: the server must be listening before training ranks
        attempt to connect/flush.
        """
        _safe(self._logger, "TCPServer.start failed", self._tcp_server.start)
        _safe(self._logger, "Display driver start failed", self._display_driver.start)
        _safe(self._logger, "Aggregator thread start failed", self._thread.start)

    def stop(self, timeout_sec: float) -> None:
        """
        Stop aggregator loop and release UI/server resources (best effort).

        Notes:
        - Aggregator loop exits when stop_event is set.
        - We join with timeout; if it doesn't stop, log a warning and continue cleanup.
        """
        self._thread.join(timeout=float(timeout_sec))
        if self._thread.is_alive():
            self._logger.error("[TraceML] WARNING: aggregator thread did not terminate")

        _safe(self._logger, "Display driver stop failed", self._display_driver.stop)
        _safe(self._logger, "TCPServer.stop failed", self._tcp_server.stop)

    def _drain_tcp(self) -> None:
        """
        Drain server messages and ingest them into the store.

        Each msg is expected to be a telemetry row (or batch) compatible with RemoteDBStore.ingest().
        """
        for msg in self._tcp_server.poll():
            _safe(
                self._logger,
                "RemoteDBStore.ingest failed",
                lambda m=msg: self._store.ingest(m),
            )

    def _loop(self) -> None:
        """
        Periodic drain + UI tick loop.

        Aggregator does not render; it delegates to the display driver.
        """
        interval_sec = float(self._settings.render_interval_sec)

        while not self._stop_event.is_set():
            self._drain_tcp()
            _safe(self._logger, "Display driver tick failed", self._display_driver.tick)
            self._stop_event.wait(interval_sec)

        # Final flush + final render tick
        self._drain_tcp()
        _safe(self._logger, "Display driver tick failed", self._display_driver.tick)