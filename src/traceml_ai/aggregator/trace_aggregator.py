# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Out-of-process telemetry server and display driver host."""

import logging
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

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
from traceml_ai.sdk.protocol import (
    get_final_summary_json_path,
    load_json_or_none,
    utc_now_iso,
)
from traceml_ai.telemetry.control import (
    RankFinishedControl,
    parse_rank_finished,
)
from traceml_ai.telemetry.envelope import TelemetryMeta
from traceml_ai.transport.tcp_transport import TCPConfig, TCPServer
from traceml_ai.utils.atomic_io import write_json_atomic

DASHBOARD_DEPENDENCY_INSTALL_HINT = (
    "Dashboard mode requires nicegui. It is included in the "
    "default TraceML install; if it is missing, run "
    "`pip install -U traceml-ai` or `pip install nicegui`."
)
_SQLITE_FINALIZE_BUDGET_FRACTION = 0.25
_SQLITE_FINALIZE_BUDGET_MIN_SEC = 5.0
_SQLITE_FINALIZE_BUDGET_MAX_SEC = 60.0
_SQLITE_FINALIZE_TINY_FLOOR_SEC = 0.001
# Longest wait between flush attempts before telling the display driver
# that a run finished, while the writer keeps failing to flush.
_RUN_FINISHED_RETRY_MAX_SEC = 30.0
# The first retry waits at least this long, so a zero render interval
# cannot turn the retry into a flush per loop iteration.
_RUN_FINISHED_RETRY_MIN_SEC = 1.0
# Each flush attempt waits at most this long (it can block the loop for
# twice this), however long the render interval is.
_RUN_FINISHED_FLUSH_TIMEOUT_MAX_SEC = 2.0
_RUN_FINISHED_FLUSH_TIMEOUT_MIN_SEC = 0.05

_LOGGER = logging.getLogger(__name__)

# (hostname, pid, session_id) of a sender whose payloads were not admitted,
# as strings so a malformed stamp cannot make the key unhashable.
_ForeignSender = Tuple[Optional[str], Optional[str], Optional[str]]


def _safe(logger: Any, label: str, fn: Callable[[], Any]) -> Any:
    """Execute ``fn()`` and log failures without raising."""
    try:
        return fn()
    except Exception:
        logger.exception(f"[TraceML] {label}")
        return None


def _stamp_of(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """The sender stamp: envelope ``meta``, else the top level (control
    and legacy flat payloads)."""
    meta = payload.get("meta")
    return meta if isinstance(meta, Mapping) else payload


class _CurrentRun:
    """Which ranks of the run now reporting have sent ``rank_finished``.

    Kept only to tell the display driver, once per run, that the run
    finished; the end-of-run settle reads ``_finished_ranks`` instead.
    ``traceml serve`` hosts one run after another on one aggregator, and
    the run stamps cannot tell them apart there: a generated session id
    names one process, and an explicit one is shared by every run. What
    can is order: a rank sends its marker after its last telemetry, so a
    rank that reports again after its marker has begun the next run.

    Every rank of a run has finished once the distinct finished ranks
    reach the configured world size or the largest one a marker named,
    whichever is larger. Changed and read under the aggregator's
    ``_drain_lock``.

    Known limits: two runs reporting to one aggregator at the same time
    are not told apart, and ranks that never send a marker (releases
    before it, or a rank that never started) keep their run from being
    told at all.
    """

    def __init__(self, expected_world_size: int) -> None:
        self._expected_world_size = expected_world_size
        self._finished: set[int] = set()
        self._world_size = 0
        self._notified = False
        # Bumped per run, so a notification settles only the run it began.
        self._generation = 0
        self._retry_at: Optional[float] = None
        self._retry_delay_s: Optional[float] = None

    def rank_finished(self, control: RankFinishedControl) -> None:
        """Count one marker toward this run."""
        self._finished.add(control.global_rank)
        self._world_size = max(self._world_size, control.world_size)

    def rank_reported(self, rank: Optional[int]) -> None:
        """Telemetry from ``rank``: past its marker, a new run began."""
        if rank is None or rank not in self._finished:
            return
        self._finished.clear()
        self._world_size = 0
        self._notified = False
        self._generation += 1
        self._retry_at = None
        self._retry_delay_s = None

    def due(self, now_s: float) -> Optional[int]:
        """This run's generation when the driver should be told now."""
        if self._notified:
            return None
        world_size = max(self._expected_world_size, self._world_size)
        if len(self._finished) < world_size:
            return None
        if self._retry_at is not None and now_s < self._retry_at:
            return None
        return self._generation

    def notified(self, generation: int) -> bool:
        """Mark the run told; False if a newer run began meanwhile."""
        if generation != self._generation:
            return False
        self._notified = True
        return True

    def flush_failed(
        self, generation: int, *, now_s: float, first_delay_s: float
    ) -> None:
        """Try this run again later: after ``first_delay_s``, doubling
        per failure, never more than ``_RUN_FINISHED_RETRY_MAX_SEC``."""
        if generation != self._generation:
            return
        delay = (
            first_delay_s
            if self._retry_delay_s is None
            else self._retry_delay_s * 2.0
        )
        self._retry_delay_s = min(delay, _RUN_FINISHED_RETRY_MAX_SEC)
        self._retry_at = now_s + self._retry_delay_s


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
            if exc.name == "nicegui":
                raise RuntimeError(
                    f"[TraceML] {DASHBOARD_DEPENDENCY_INSTALL_HINT}"
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
        # The run now reporting, to tell the display driver it finished.
        self._current_run = _CurrentRun(self._expected_world_size)
        # Times the flush retries; injectable so a test can step it.
        self._retry_clock: Callable[[], float] = time.monotonic
        self._foreign_senders: dict[_ForeignSender, int] = {}
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
                history_retention_s=float(settings.history_retention_s),
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
            write_html=bool(settings.html_report),
            profile=str(settings.profile),
            history_retention_s=float(settings.history_retention_s),
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

        self._thread.start()
        self._started = True

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
            try:
                if self._settings.history_enabled:
                    warning_payload = self._settle_end_of_run_telemetry(
                        settle_budget
                    )
                _safe(
                    self._logger,
                    "TCPServer.stop failed",
                    self._tcp_server.stop,
                )
            finally:
                # Report dropped senders even when the final drain raised.
                self._warn_foreign_senders()
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
            ):
                summary_path = get_final_summary_json_path(session_root)
                previous_generation = self._summary_generation_id(summary_path)
                summary_error: Optional[Exception] = None
                try:
                    generate_summary(
                        str(self._settings.db_path),
                        session_root=str(session_root),
                        print_to_stdout=True,
                        history_retention_s=float(
                            self._settings.history_retention_s
                        ),
                        write_html=bool(self._settings.html_report),
                        profile=str(self._settings.profile),
                    )
                except Exception as summary_exc:
                    summary_error = summary_exc

                if self._summary_generation_id(summary_path) in (
                    None,
                    previous_generation,
                ):
                    refresh_error = TraceMLFinalizationError(
                        "Final summary was not refreshed at end of run."
                    )
                    if summary_error is not None:
                        raise refresh_error from summary_error
                    raise refresh_error

                if summary_error is not None:
                    # A refreshed artifact makes this error nonfatal, so this
                    # component remains its sole structured-log owner.
                    self._logger.error(
                        "[TraceML] generate_summary raised",
                        exc_info=(
                            type(summary_error),
                            summary_error,
                            summary_error.__traceback__,
                        ),
                    )
                    warning_payload = self._add_generate_summary_warning(
                        warning_payload,
                        summary_error,
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
                    "finished_ranks": self._finished_ranks_snapshot(),
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

    @staticmethod
    def _add_generate_summary_warning(
        warning_payload: Optional[dict[str, Any]],
        exc: Exception,
    ) -> Optional[dict[str, Any]]:
        """Record a nonfatal post-write summary-generation error.

        Reached only after verifying that ``final_summary.json`` has a new
        generation marker, so a later render/print error did not cost the
        artifact and must not fail the run.
        """
        if warning_payload is None:
            warning_payload = {
                "status": "warning",
                "completed_at": utc_now_iso(),
                "message": (
                    "Summary artifact was written, but summary generation "
                    "raised after the write; continuing without failing the "
                    "run."
                ),
            }
        warning_payload["generate_summary_error"] = str(exc)
        return warning_payload

    @staticmethod
    def _summary_generation_id(path: Path) -> Optional[str]:
        """Return the generation marker from a valid summary artifact."""
        payload = load_json_or_none(path)
        if not isinstance(payload, dict):
            return None
        generated_at = payload.get("generated_at")
        if not isinstance(generated_at, str) or not generated_at.strip():
            return None
        return generated_at

    def _finished_ranks_snapshot(self) -> List[int]:
        """Return a sorted snapshot of finished global ranks (lock-safe).

        ``_finished_ranks`` is mutated under ``_drain_lock`` by the drain
        path; snapshotting under the same lock avoids a ``RuntimeError`` if a
        late/zombie drain mutates the dict mid-iteration.
        """
        with self._drain_lock:
            return sorted(self._finished_ranks)

    def _split_telemetry_payloads(self, msg: Any) -> List[Any]:
        """
        Admit payloads, consume control messages, return sampler telemetry.

        Rank-finished markers share the TCP transport with telemetry so workers
        do not need a second control channel. They are consumed here and never
        enter SQLite projection storage.

        This is the single admission point between the socket and SQLite:
        every payload, telemetry or control, passes ``_admit`` first, so a
        rank left over from another run can neither add rows nor mark one of
        this run's ranks finished.

        Callers hold ``_drain_lock``, which also guards ``_current_run``.
        """
        items = msg if isinstance(msg, list) else [msg]
        telemetry: List[Any] = []
        for item in items:
            if not self._admit(item):
                continue
            control = parse_rank_finished(item)
            if control is not None:
                self._finished_ranks[control.global_rank] = control
                self._current_run.rank_finished(control)
            else:
                if isinstance(item, Mapping):
                    self._current_run.rank_reported(
                        TelemetryMeta.from_mapping(_stamp_of(item)).rank
                    )
                telemetry.append(item)

        if isinstance(msg, list):
            return [telemetry] if telemetry else []
        return telemetry

    def _admit(self, payload: Any) -> bool:
        """
        Return False for a payload stamped for a different run.

        The stamp is read from envelope ``meta`` or, for control and legacy
        flat payloads, from the top level. Each key is checked only when this
        aggregator enforces it and the payload carries it: ranks that predate
        the stamp send neither key and are always admitted. ``session_id`` is
        enforced when the launch path shares one run id with its ranks, and
        ``run_nonce`` when a single launcher started both sides. With
        ``admit_generated_session_id`` only explicit ids are enforced: a
        stamped id without a ``session_source`` counts as explicit.
        """
        if not isinstance(payload, Mapping):
            return True
        stamp = _stamp_of(payload)

        settings = self._settings
        enforce_session = settings.enforce_session_id and not (
            settings.admit_generated_session_id
            and stamp.get("session_source") == "generated"
        )
        expected = (
            (
                "session_id",
                str(settings.session_id or "") if enforce_session else "",
            ),
            ("run_nonce", str(settings.run_nonce or "")),
        )
        for key, want in expected:
            got = stamp.get(key)
            if want and got not in (None, "") and str(got) != want:
                sender: _ForeignSender = tuple(  # type: ignore[assignment]
                    None if stamp.get(field) is None else str(stamp[field])
                    for field in ("hostname", "pid", "session_id")
                )
                self._foreign_senders[sender] = (
                    self._foreign_senders.get(sender, 0) + 1
                )
                return False
        return True

    def _warn_foreign_senders(self) -> None:
        """Print one stderr warning naming every sender that was dropped."""
        with self._drain_lock:
            senders = dict(self._foreign_senders)
        if not senders:
            return

        def fmt(value: Any) -> str:
            return "?" if value in (None, "") else str(value)

        named = "; ".join(
            f"host={fmt(host)} pid={fmt(pid)} session={fmt(session)} "
            f"({count})"
            for (host, pid, session), count in sorted(
                senders.items(), key=lambda item: str(item[0])
            )
        )
        try:
            print(
                f"[TraceML] WARNING: ignored {sum(senders.values())} "
                f"payload(s) from another TraceML run: {named}. A training "
                "process from an earlier run may still be running; stop it "
                "so it no longer sends telemetry to this run.",
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            # Best-effort diagnostics must not interrupt finalization.
            pass

    def _settle_end_of_run_telemetry(
        self, timeout_sec: float
    ) -> Optional[dict[str, Any]]:
        """
        Drain late telemetry before final SQLite close.

        The aggregator keeps TCP open during this phase because worker ranks can
        finish at slightly different times on multi-node jobs. Finalization
        proceeds once all expected ranks sent a rank-finished marker, once no
        rank connection is still open, or when the caller's deadline expires.

        A rank that was killed never sends its marker, and a rank whose
        connection has closed cannot send anything more, so waiting on it would
        only hold the aggregator port for the rest of the budget.
        """
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        quiet_sec = 0.5
        reason = "timeout"
        # Back-compat: workers from older releases do not send a rank_finished
        # marker, so _finished_ranks never reaches expected_world_size. They
        # still end this loop once their connections close, and then emit a
        # nonfatal "missing ranks" warning.
        while time.monotonic() < deadline:
            self._drain_tcp()
            all_finished = (
                len(self._finished_ranks) >= self._expected_world_size
            )
            if all_finished or self._tcp_server.open_connections() == 0:
                # Wait one quiet window so frames still in flight are taken.
                remaining = max(0.0, deadline - time.monotonic())
                if not self._tcp_server.wait_for_data(
                    timeout=min(quiet_sec, remaining)
                ):
                    self._drain_tcp()
                    if all_finished:
                        return None
                    reason = "ranks_disconnected"
                    break
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
        if reason == "ranks_disconnected":
            message = (
                "Every rank connection closed before all ranks reported "
                "finished; finalizing without the missing ranks."
            )
        else:
            message = (
                "Timed out waiting for all ranks to report finished before "
                "end-of-run finalization."
            )
        warning = {
            "status": "warning",
            "completed_at": utc_now_iso(),
            "reason": reason,
            "message": message,
            "expected_world_size": self._expected_world_size,
            "finished_ranks": self._finished_ranks_snapshot(),
            "missing_ranks": missing,
        }
        try:
            self._logger.warning("[TraceML] %s", warning["message"])
        except Exception:
            # Best-effort warning emission: logging failures must not interrupt
            # finalization or alter the warning payload returned to callers.
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
        except Exception as exc:
            _LOGGER.warning(
                "[TraceML] Failed to write finalization artifact '%s': %s",
                filename,
                exc,
            )

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

    def _notify_run_finished(self) -> None:
        """Tell the display driver, once per run, that the run finished.

        Called from the loop thread. ``_current_run`` is read and changed
        under ``_drain_lock``, the lock marker ingestion holds, because
        ``stop()`` can still drain from the main thread when this loop
        outlives its join. The end-of-run settle's ``_finished_ranks`` is
        not touched.

        SQLite stamps an arrival when it flushes it, so everything queued
        is flushed first: every arrival of the finished run is then
        stamped at or before the moment the driver is told. Each attempt
        waits the render interval, clamped to
        ``_RUN_FINISHED_FLUSH_TIMEOUT_MIN_SEC`` ..
        ``_RUN_FINISHED_FLUSH_TIMEOUT_MAX_SEC``, and can block this loop for
        twice that (the barrier waits once to enter the queue, then once to
        be processed). When it does not complete, the driver is not told
        and the run is tried again on a later iteration: one render
        interval later (at least ``_RUN_FINISHED_RETRY_MIN_SEC``), doubling
        per failure, never more than ``_RUN_FINISHED_RETRY_MAX_SEC`` apart.
        Once backed off, a writer that never flushes costs at most one
        such stall per that many seconds. The driver's own failure is
        logged, not retried.
        """
        with self._drain_lock:
            generation = self._current_run.due(self._retry_clock())
        if generation is None:
            return
        interval_s = float(self._settings.render_interval_sec)
        flush_timeout_s = min(
            max(interval_s, _RUN_FINISHED_FLUSH_TIMEOUT_MIN_SEC),
            _RUN_FINISHED_FLUSH_TIMEOUT_MAX_SEC,
        )
        flushed = _safe(
            self._logger,
            "SQLite flush before display run_finished failed",
            lambda: self._sqlite_writer.force_flush(flush_timeout_s),
        )
        with self._drain_lock:
            if flushed:
                told = self._current_run.notified(generation)
            else:
                told = False
                self._current_run.flush_failed(
                    generation,
                    now_s=self._retry_clock(),
                    first_delay_s=max(interval_s, _RUN_FINISHED_RETRY_MIN_SEC),
                )
        if told:
            _safe(
                self._logger,
                "Display driver run_finished failed",
                self._display_driver.run_finished,
            )

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
            self._notify_run_finished()

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
        self._notify_run_finished()
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
