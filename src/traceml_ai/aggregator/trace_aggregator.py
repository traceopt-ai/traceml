# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Out-of-process telemetry server and display driver host."""

import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from traceml_ai.aggregator.display_drivers.base import BaseDisplayDriver
from traceml_ai.aggregator.display_drivers.cli import CLIDisplayDriver
from traceml_ai.aggregator.display_drivers.summary import SummaryDisplayDriver
from traceml_ai.aggregator.sqlite_writer import (
    SQLiteWriterConfig,
    SQLiteWriterSimple,
)
from traceml_ai.aggregator.summary_service import FinalSummaryService
from traceml_ai.reporting.final import generate_summary
from traceml_ai.runtime.settings import AggregatorEndpoint, TraceMLSettings
from traceml_ai.sdk.protocol import get_final_summary_json_path, utc_now_iso
from traceml_ai.telemetry.control import (
    RankFinishedControl,
    parse_rank_finished,
)
from traceml_ai.transport.tcp_transport import TCPConfig, TCPServer
from traceml_ai.utils.atomic_io import write_json_atomic

DASHBOARD_EXTRA_INSTALL_HINT = (
    "Dashboard mode requires optional dependencies. Install them with "
    "`pip install 'traceml-ai[dashboard]'`."
)
_SQLITE_FINALIZE_BUDGET_FRACTION = 0.25
_SQLITE_FINALIZE_BUDGET_MIN_SEC = 5.0
_SQLITE_FINALIZE_BUDGET_MAX_SEC = 60.0
_SQLITE_FINALIZE_TINY_FLOOR_SEC = 0.001


def _safe(logger: Any, label: str, fn: Callable[[], Any]) -> Any:
    """Execute ``fn()`` and log failures without raising."""
    try:
        return fn()
    except Exception:
        logger.exception(f"[TraceML] {label}")
        return None


class TraceMLFinalizationError(RuntimeError):
    """Raised when summary-mode end-of-run finalization cannot complete."""


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

    def __init__(
        self,
        logger: Any,
        stop_event: threading.Event,
        settings: TraceMLSettings,
    ) -> None:
        self._logger = logger
        self._stop_event = stop_event
        self._settings = settings
        self._expected_world_size = max(
            1, int(getattr(settings, "expected_world_size", 1) or 1)
        )
        self._finished_ranks: dict[int, RankFinishedControl] = {}
        self._started = False
        self._drain_lock = threading.Lock()

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
            flush_history=self._sqlite_writer.force_flush,
            settle_telemetry=self._settle_telemetry,
            summary_window_rows=int(settings.summary_window_rows),
            write_html=bool(settings.html_report),
        )

        # Display driver owns renderer selection and layout mapping.
        driver_cls = _resolve_display_driver(settings.mode)

        self._display_driver = driver_cls(
            logger=self._logger,
            settings=self._settings,
        )

        self._thread = threading.Thread(
            target=self._loop,
            name="TraceMLAggregator",
            daemon=True,
        )

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
            self._started = True
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
        Stop the aggregator and deterministically finalize end-of-run artifacts.

        Notes
        -----
        End-of-run finalization keeps the TCP server open briefly so late
        multi-node telemetry can arrive, closes SQLite before summary generation,
        and treats a missing summary as an error in summary mode.
        """
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        session_root = Path(str(self._settings.logs_dir)).resolve() / str(
            self._settings.session_id or "default"
        )

        def remaining() -> float:
            return max(0.0, deadline - time.monotonic())

        warning_payload: Optional[dict[str, Any]] = None
        finalize_payload: Optional[dict[str, Any]] = None

        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=min(5.0, remaining()))

        if self._thread.is_alive():
            self._logger.error(
                "[TraceML] WARNING: aggregator thread did not terminate"
            )

        _safe(
            self._logger,
            "Display driver stop failed",
            self._display_driver.stop,
        )

        try:
            sqlite_finalize_budget = self._sqlite_finalize_budget(remaining())
            settle_budget = max(0.0, remaining() - sqlite_finalize_budget)
            if self._settings.history_enabled:
                warning_payload = self._settle_end_of_run_telemetry(
                    settle_budget
                )
            _safe(
                self._logger,
                "TCPServer.stop failed",
                self._tcp_server.stop,
            )
            finalize_result = self._sqlite_writer.finalize(
                max(sqlite_finalize_budget, remaining())
            )
            finalize_payload = finalize_result.to_dict()
            if not finalize_result.ok:
                raise TraceMLFinalizationError(
                    "SQLite history did not finalize cleanly: "
                    f"{finalize_result.error or 'unknown error'}"
                )
            warning_payload = self._add_sqlite_finalize_warning(
                warning_payload,
                finalize_payload,
            )

            if (
                self._started
                and self._settings.history_enabled
                and self._settings.db_path
                and not get_final_summary_json_path(session_root).is_file()
            ):
                generate_summary(
                    str(self._settings.db_path),
                    session_root=str(session_root),
                    print_to_stdout=True,
                    summary_window_rows=int(
                        self._settings.summary_window_rows
                    ),
                    write_html=bool(self._settings.html_report),
                )

            if (
                self._started
                and self._settings.mode == "summary"
                and self._settings.history_enabled
                and not get_final_summary_json_path(session_root).is_file()
            ):
                raise TraceMLFinalizationError(
                    "Summary mode finished without final_summary.json."
                )

            if warning_payload is not None:
                self._write_finalization_artifact(
                    session_root,
                    "finalization_warning.json",
                    warning_payload,
                )

        except Exception as exc:
            self._write_finalization_artifact(
                session_root,
                "finalization_error.json",
                {
                    "status": "error",
                    "completed_at": utc_now_iso(),
                    "error": str(exc),
                    "finished_ranks": sorted(self._finished_ranks),
                    "expected_world_size": self._expected_world_size,
                    "sqlite_finalize": finalize_payload,
                    "writer": self._sqlite_writer.stats(),
                },
            )
            if self._settings.mode == "summary":
                raise
            self._logger.exception("[TraceML] Finalization failed")

    def _drain_tcp(self) -> None:
        """
        Drain pending TCP messages and ingest them into SQLite history.

        Each message is expected to be a telemetry row or batch compatible with
        ``SQLiteWriterSimple.ingest()``.
        """
        with self._drain_lock:
            for msg in self._tcp_server.poll():
                for payload in self._split_telemetry_payloads(msg):
                    try:
                        self._sqlite_writer.ingest(payload)
                    except Exception:
                        self._logger.exception(
                            "[TraceML] SQLiteWriter.ingest failed"
                        )

    @staticmethod
    def _sqlite_finalize_budget(timeout_sec: float) -> float:
        """
        Reserve part of the end-of-run timeout for SQLite close/checkpoint.

        Large runs need most of the timeout for late rank telemetry, but SQLite
        still needs a guaranteed slice so a missing rank marker cannot consume
        the whole deadline and turn a clean writer close into a false timeout.
        """
        total = max(0.0, float(timeout_sec))
        if total <= 0.0:
            return 0.0
        if total < _SQLITE_FINALIZE_BUDGET_MIN_SEC:
            return max(
                total * _SQLITE_FINALIZE_BUDGET_FRACTION,
                min(_SQLITE_FINALIZE_TINY_FLOOR_SEC, total),
            )
        return min(
            _SQLITE_FINALIZE_BUDGET_MAX_SEC,
            max(
                _SQLITE_FINALIZE_BUDGET_MIN_SEC,
                total * _SQLITE_FINALIZE_BUDGET_FRACTION,
            ),
        )

    @staticmethod
    def _add_sqlite_finalize_warning(
        warning_payload: Optional[dict[str, Any]],
        finalize_payload: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """Attach nonfatal SQLite finalize warnings to the warning artifact."""
        if not finalize_payload or finalize_payload.get("prune_ok", True):
            return warning_payload

        if warning_payload is None:
            warning_payload = {
                "status": "warning",
                "completed_at": utc_now_iso(),
                "message": (
                    "SQLite final retention prune failed; summary generation "
                    "continued because queued telemetry was written and "
                    "SQLite checkpoint/close succeeded."
                ),
            }
        warning_payload["sqlite_finalize"] = finalize_payload
        return warning_payload

    def _split_telemetry_payloads(self, msg: Any) -> List[Any]:
        """
        Consume control messages and return sampler telemetry payloads.

        Rank-finished markers share the TCP transport with telemetry so workers
        do not need a second control channel. They are consumed here and never
        enter SQLite projection storage.
        """
        if isinstance(msg, list):
            telemetry: List[Any] = []
            for item in msg:
                control = parse_rank_finished(item)
                if control is not None:
                    self._finished_ranks[control.global_rank] = control
                else:
                    telemetry.append(item)
            return [telemetry] if telemetry else []

        control = parse_rank_finished(msg)
        if control is not None:
            self._finished_ranks[control.global_rank] = control
            return []
        return [msg]

    def _settle_end_of_run_telemetry(
        self, timeout_sec: float
    ) -> Optional[dict[str, Any]]:
        """
        Drain late telemetry before final SQLite close.

        The aggregator keeps TCP open during this phase because worker ranks can
        finish at slightly different times on multi-node jobs. Finalization
        proceeds once all expected ranks sent a rank-finished marker or the
        caller's deadline expires.
        """
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        quiet_sec = 0.5

        while time.monotonic() < deadline:
            self._drain_tcp()
            if len(self._finished_ranks) >= self._expected_world_size:
                remaining = max(0.0, deadline - time.monotonic())
                if not self._tcp_server.wait_for_data(
                    timeout=min(quiet_sec, remaining)
                ):
                    self._drain_tcp()
                    return None
                continue

            remaining = max(0.0, deadline - time.monotonic())
            if remaining <= 0.0:
                break
            self._tcp_server.wait_for_data(timeout=min(quiet_sec, remaining))

        self._drain_tcp()
        if len(self._finished_ranks) >= self._expected_world_size:
            return None

        missing = [
            rank
            for rank in range(self._expected_world_size)
            if rank not in self._finished_ranks
        ]
        warning = {
            "status": "warning",
            "completed_at": utc_now_iso(),
            "message": (
                "Timed out waiting for all ranks to report finished before "
                "end-of-run finalization."
            ),
            "expected_world_size": self._expected_world_size,
            "finished_ranks": sorted(self._finished_ranks),
            "missing_ranks": missing,
        }
        try:
            self._logger.warning("[TraceML] %s", warning["message"])
        except Exception:
            pass
        return warning

    @staticmethod
    def _write_finalization_artifact(
        session_root: Path,
        filename: str,
        payload: dict[str, Any],
    ) -> None:
        """Write a finalization diagnostic artifact under ``aggregator/``."""
        try:
            path = Path(session_root).resolve() / "aggregator" / filename
            write_json_atomic(path, payload)
        except Exception:
            pass

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
                return bool(self._sqlite_writer.force_flush(remaining))

        self._drain_tcp()
        return bool(self._sqlite_writer.force_flush(0.0))

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
