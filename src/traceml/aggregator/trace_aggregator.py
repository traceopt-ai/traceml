"""Out-of-process telemetry server and display driver host."""

import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

from traceml.aggregator.display_drivers.base import BaseDisplayDriver
from traceml.aggregator.display_drivers.cli import CLIDisplayDriver
from traceml.aggregator.display_drivers.nicegui import NiceGUIDisplayDriver
from traceml.aggregator.display_drivers.summary import SummaryDisplayDriver
from traceml.aggregator.sqlite_writer import (
    SQLiteWriterConfig,
    SQLiteWriterSimple,
)
from traceml.aggregator.summary_service import FinalSummaryService
from traceml.database.remote_database_store import RemoteDBStore
from traceml.reporting.final import generate_summary
from traceml.runtime.settings import TraceMLSettings
from traceml.transport.tcp_transport import TCPConfig, TCPServer


def _safe(logger: Any, label: str, fn: Callable[[], Any]) -> Any:
    """Execute ``fn()`` and log failures without raising."""
    try:
        return fn()
    except Exception:
        logger.exception(f"[TraceML] {label}")
        return None


_DISPLAY_DRIVERS: Dict[str, Type[BaseDisplayDriver]] = {
    "cli": CLIDisplayDriver,
    "dashboard": NiceGUIDisplayDriver,
    "summary": SummaryDisplayDriver,
}


class TraceMLAggregator:
    """Telemetry aggregator process."""

    _REMOTE_STORE_SAMPLERS = frozenset(
        {
            "LayerMemorySampler",
            "LayerForwardMemorySampler",
            "LayerBackwardMemorySampler",
            "LayerForwardTimeSampler",
            "LayerBackwardTimeSampler",
        }
    )

    def __init__(
        self,
        logger: Any,
        stop_event: threading.Event,
        settings: TraceMLSettings,
    ) -> None:
        self._logger = logger
        self._stop_event = stop_event
        self._settings = settings

        # Transitional live store for the remaining renderer paths that have
        # not yet moved to SQLite-backed history.
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

        session_root = Path(str(settings.logs_dir)).resolve() / str(
            settings.session_id or "default"
        )

        self._summary_service = FinalSummaryService(
            logger=self._logger,
            session_root=session_root,
            db_path=str(db_path),
            flush_history=self._sqlite_writer.flush_now,
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
                lambda: generate_summary(
                    str(self._settings.db_path),
                    session_root=str(
                        Path(str(self._settings.logs_dir)).resolve()
                        / str(self._settings.session_id or "default")
                    ),
                    print_to_stdout=True,
                ),
            )

    def _drain_tcp(self) -> None:
        """
        Drain pending TCP messages and ingest them into the store and history.

        Each message is expected to be a telemetry row or batch compatible with
        ``SQLiteWriterSimple.ingest()``. A legacy subset is also mirrored into
        ``RemoteDBStore`` for renderers that still depend on the in-memory path.

        Design note
        -----------
        Errors in each sink are isolated: a failure in one does not prevent the
        other from receiving the message.  Direct try/except is used instead of
        ``_safe(lambda ...)`` to avoid allocating two closure objects per message
        in the hot path.
        """
        for msg in self._tcp_server.poll():
            remote_msg = self._filter_remote_store_message(msg)
            if remote_msg is not None:
                try:
                    self._store.ingest(remote_msg)
                except Exception:
                    self._logger.exception(
                        "[TraceML] RemoteDBStore.ingest failed"
                    )
            try:
                self._sqlite_writer.ingest(msg)
            except Exception:
                self._logger.exception("[TraceML] SQLiteWriter.ingest failed")

    def _message_sampler_name(self, msg: Any) -> Optional[str]:
        """
        Return the sampler name carried by one logical telemetry payload.

        Returns ``None`` when the payload is malformed or does not follow the
        expected envelope shape.
        """
        if not isinstance(msg, dict):
            return None

        sampler = msg.get("sampler")
        if sampler is None:
            return None

        try:
            return str(sampler)
        except Exception:
            return None

    def _filter_remote_store_message(self, msg: Any) -> Any:
        """
        Return the subset of a telemetry message still needed by RemoteDBStore.

        Notes
        -----
        - SQLite remains the full history sink for every payload.
        - RemoteDBStore is now treated as a transitional cache for the few
          legacy renderers that still depend on it.
        - Batch envelopes preserve their original list shape so
          `RemoteDBStore.ingest()` can continue to process them normally.
        """
        if isinstance(msg, list):
            filtered = [
                item
                for item in msg
                if self._message_sampler_name(item)
                in self._REMOTE_STORE_SAMPLERS
            ]
            return filtered if filtered else None

        sampler_name = self._message_sampler_name(msg)
        if sampler_name in self._REMOTE_STORE_SAMPLERS:
            return msg

        return None

    def _loop(self) -> None:
        """
        Run the event-driven drain and display tick loop.

        The loop blocks on ``TCPServer.wait_for_data()`` rather than sleeping
        for a fixed interval.  This means the aggregator drains new messages
        as soon as they arrive over TCP — reducing end-to-end ingestion latency
        from up to ``render_interval_sec`` down to near-zero.

        The display driver tick is still rate-limited to at most once per
        ``render_interval_sec`` so the UI cadence is unchanged.
        """
        interval_sec = float(self._settings.render_interval_sec)
        last_tick_ts = 0.0

        while not self._stop_event.is_set():
            # Wake immediately when data arrives, or after interval_sec at most.
            self._tcp_server.wait_for_data(timeout=interval_sec)
            self._drain_tcp()

            # Rate-limit the UI tick to interval_sec cadence.
            now = time.monotonic()
            if now - last_tick_ts >= interval_sec:
                _safe(
                    self._logger,
                    "Final summary service poll failed",
                    self._summary_service.poll,
                )
                _safe(
                    self._logger,
                    "Display driver tick failed",
                    self._display_driver.tick,
                )
                last_tick_ts = now

        # Final drain and final display tick on shutdown.
        self._drain_tcp()
        _safe(
            self._logger,
            "Final summary service poll failed",
            self._summary_service.poll,
        )
        _safe(
            self._logger,
            "Display driver tick failed",
            self._display_driver.tick,
        )
