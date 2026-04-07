"""
TraceML Aggregator (out-of-process telemetry server + UI driver).

Responsibilities
----------------
- Receive telemetry rows from all ranks over TCP
- Ingest rows into a unified RemoteDBStore
- Optionally persist telemetry to SQLite history
- Drive a display driver (CLI / NiceGUI) that owns renderers and view layout


Key invariants
--------------
- Renderers currently read from RemoteDBStore and, in some cases, SQLite-backed
  history during the storage transition.
- Over time, SQLite-backed history may replace parts of the RemoteDBStore read path.
- The aggregator MUST NOT know about UI sections, layouts, or renderer methods.
"""

import threading
import time
from typing import Any, Callable, Dict, Type

from traceml.aggregator.display_drivers.base import BaseDisplayDriver
from traceml.aggregator.display_drivers.cli import CLIDisplayDriver
from traceml.aggregator.display_drivers.nicegui import NiceGUIDisplayDriver
from traceml.aggregator.sqlite_writer import (
    SQLiteWriterConfig,
    SQLiteWriterSimple,
)
from traceml.database.remote_database_store import RemoteDBStore
from traceml.runtime.settings import TraceMLSettings
from traceml.transport.tcp_transport import TCPConfig, TCPServer

from .final_summary import generate_summary


def _safe(logger: Any, label: str, fn: Callable[[], Any]) -> Any:
    """
    Execute ``fn()`` and never raise.

    This helper is intended for cleanup paths, UI ticking, and telemetry
    ingestion paths where failures should be logged but should not crash the
    already-running aggregator process.
    """
    try:
        return fn()
    except Exception:
        logger.exception(f"[TraceML] {label}")
        return None


_DISPLAY_DRIVERS: Dict[str, Type[BaseDisplayDriver]] = {
    "cli": CLIDisplayDriver,
    "dashboard": NiceGUIDisplayDriver,
}


class TraceMLAggregator:
    """
    Telemetry aggregator process.

    Owns
    ----
    - TCPServer: receives messages from training ranks
    - RemoteDBStore: unified telemetry store (single source of truth)
    - SQLiteWriterSimple: optional history persistence
    - Display driver: backend-specific driver that owns renderers and UI updates
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

        # Unified telemetry store: renderers read only from here.
        self._store = RemoteDBStore(max_rows=int(settings.remote_max_rows))

        # TCP server: aggregator listens for rank-local agents.
        self._tcp_server = TCPServer(
            TCPConfig(
                host=str(settings.tcp.host),
                port=int(settings.tcp.port),
            )
        )

        db_path = getattr(settings, "db_path", None)
        if not db_path:
            db_path = f"traceml_session_{time.time_ns()}.db"

        self._sqlite_writer = SQLiteWriterSimple(
            SQLiteWriterConfig(
                path=str(db_path),
                enabled=bool(settings.history_enabled),
                max_queue=50_000,
                flush_interval_sec=0.5,
                max_flush_items=20_000,
                synchronous="NORMAL",
            ),
        )

        # Display driver owns renderer selection and layout mapping.
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
        """Expose the store for tests and read-only inspection."""
        return self._store

    def start(self) -> None:
        """
        Start the aggregator server, persistence layer, UI driver, and loop.

        Start order matters:
        1. TCP server must start first so workers can connect.
        2. SQLite writer starts before ingestion begins.
        3. Display driver starts before periodic ticks.
        4. Aggregator loop thread starts last.

        Startup failures should propagate so the launcher can fail fast rather
        than running in a partially initialized state.
        """
        self._tcp_server.start()
        self._sqlite_writer.start()
        self._display_driver.start()

        try:
            self._thread.start()
        except Exception:
            self._logger.exception("[TraceML] Aggregator thread start failed")
            raise

    def stop(self, timeout_sec: float) -> None:
        """
        Stop the aggregator loop and release resources.

        Notes
        -----
        - The loop exits when ``stop_event`` is set by the process entrypoint.
        - Cleanup is best-effort once shutdown begins.
        - Final summary generation runs only when history is enabled and a
          database path is configured.
        """
        if self._thread.is_alive():
            self._thread.join(timeout=float(timeout_sec))

        if self._thread.is_alive():
            self._logger.error(
                "[TraceML] WARNING: aggregator thread did not terminate"
            )

        _safe(
            self._logger,
            "Display driver stop failed",
            self._display_driver.stop,
        )
        _safe(
            self._logger,
            "TCPServer.stop failed",
            self._tcp_server.stop,
        )
        _safe(
            self._logger,
            "SQLiteWriter.stop failed",
            self._sqlite_writer.stop,
        )

        if self._settings.history_enabled and self._settings.db_path:
            _safe(
                self._logger,
                "Final summary failed",
                lambda: generate_summary(str(self._settings.db_path)),
            )

    def _drain_tcp(self) -> None:
        """
        Drain pending TCP messages and ingest them into the store and history.

        Each message is expected to be a telemetry row or batch compatible with
        ``RemoteDBStore.ingest()`` and ``SQLiteWriterSimple.ingest()``.
        """
        for msg in self._tcp_server.poll():
            _safe(
                self._logger,
                "RemoteDBStore.ingest failed",
                lambda m=msg: self._store.ingest(m),
            )
            _safe(
                self._logger,
                "SQLiteWriter.ingest failed",
                lambda m=msg: self._sqlite_writer.ingest(m),
            )

    def _loop(self) -> None:
        """
        Run the periodic drain and display tick loop.

        The aggregator does not render directly. Rendering is delegated to the
        configured display driver.
        """
        interval_sec = float(self._settings.render_interval_sec)

        while not self._stop_event.is_set():
            self._drain_tcp()
            _safe(
                self._logger,
                "Display driver tick failed",
                self._display_driver.tick,
            )
            self._stop_event.wait(interval_sec)

        # Final drain and final display tick on shutdown.
        self._drain_tcp()
        _safe(
            self._logger,
            "Display driver tick failed",
            self._display_driver.tick,
        )
