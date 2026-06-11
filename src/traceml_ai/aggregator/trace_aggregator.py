# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Out-of-process telemetry server and display driver host."""

import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

from traceml_ai.aggregator.display_drivers.base import BaseDisplayDriver
from traceml_ai.aggregator.display_drivers.cli import CLIDisplayDriver
from traceml_ai.aggregator.display_drivers.summary import SummaryDisplayDriver
from traceml_ai.aggregator.sqlite_writer import (
    SQLiteWriterConfig,
    SQLiteWriterSimple,
)
from traceml_ai.aggregator.summary_service import FinalSummaryService
from traceml_ai.database.remote_database_store import RemoteDBStore
from traceml_ai.reporting.final import generate_summary
from traceml_ai.runtime.settings import AggregatorEndpoint, TraceMLSettings
from traceml_ai.sdk.protocol import get_final_summary_json_path
from traceml_ai.telemetry.envelope import normalize_telemetry_envelope
from traceml_ai.transport.tcp_transport import TCPConfig, TCPServer

DASHBOARD_EXTRA_INSTALL_HINT = (
    "Dashboard mode requires optional dependencies. Install them with "
    "`pip install 'traceml-ai[dashboard]'`."
)


def _safe(logger: Any, label: str, fn: Callable[[], Any]) -> Any:
    """Execute ``fn()`` and log failures without raising."""
    try:
        return fn()
    except Exception:
        logger.exception(f"[TraceML] {label}")
        return None


_DISPLAY_DRIVERS: Dict[str, Type[BaseDisplayDriver]] = {
    "cli": CLIDisplayDriver,
    "summary": SummaryDisplayDriver,
}


def _resolve_display_driver(mode: str) -> Type[BaseDisplayDriver]:
    if mode == "dashboard":
        try:
            from traceml_ai.aggregator.display_drivers.nicegui import (
                NiceGUIDisplayDriver,
            )
        except ModuleNotFoundError as exc:
            if exc.name in {"nicegui", "plotly"}:
                raise RuntimeError(
                    f"[TraceML] {DASHBOARD_EXTRA_INSTALL_HINT}"
                ) from exc
            raise

        return NiceGUIDisplayDriver

    driver_cls = _DISPLAY_DRIVERS.get(mode)
    if driver_cls is None:
        supported = sorted([*_DISPLAY_DRIVERS.keys(), "dashboard"])
        raise ValueError(
            f"[TraceML] Unknown display mode: {mode!r}. Supported: {supported}"
        )
    return driver_cls


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
                host=str(settings.aggregator.bind_host),
                port=int(settings.aggregator.port),
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
                summary_window_rows=int(settings.summary_window_rows),
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
            settle_telemetry=self._settle_telemetry,
            summary_window_rows=int(settings.summary_window_rows),
            write_html=bool(settings.html_report),
        )

        # Display driver owns renderer selection and layout mapping.
        driver_cls = _resolve_display_driver(settings.mode)

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

    @property
    def endpoint(self) -> AggregatorEndpoint:
        """Return the reachable endpoint after the TCP server has started."""
        return AggregatorEndpoint(
            host=str(self._settings.aggregator.connect_host),
            port=int(self._tcp_server.port),
            session_id=str(self._settings.session_id or "default"),
        )

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

        session_root = Path(str(self._settings.logs_dir)).resolve() / str(
            self._settings.session_id or "default"
        )
        if (
            self._settings.history_enabled
            and self._settings.db_path
            and not get_final_summary_json_path(session_root).is_file()
        ):
            _safe(
                self._logger,
                "Final summary failed",
                lambda: generate_summary(
                    str(self._settings.db_path),
                    session_root=str(session_root),
                    print_to_stdout=True,
                    summary_window_rows=int(
                        self._settings.summary_window_rows
                    ),
                    write_html=bool(self._settings.html_report),
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

    def _settle_telemetry(self, timeout_sec: float) -> bool:
        """
        Best-effort drain of in-flight telemetry before first final summary.

        This waits until TCP input is quiet for a short window or until the
        timeout expires, then flushes SQLite history.
        """
        deadline = time.monotonic() + float(timeout_sec)
        quiet_sec = min(0.5, max(0.0, float(timeout_sec)))

        while time.monotonic() < deadline:
            self._drain_tcp()
            remaining = max(0.0, deadline - time.monotonic())
            if not self._tcp_server.wait_for_data(
                timeout=min(quiet_sec, remaining)
            ):
                self._drain_tcp()
                return bool(self._sqlite_writer.flush_now(remaining))

        self._drain_tcp()
        return bool(self._sqlite_writer.flush_now(0.0))

    def _message_sampler_name(self, msg: Any) -> Optional[str]:
        """
        Return the sampler name carried by one logical telemetry payload.

        Returns ``None`` when the payload is malformed or does not follow the
        expected envelope shape.
        """
        envelope = normalize_telemetry_envelope(msg)
        if envelope is None:
            return None

        return envelope.meta.sampler

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
