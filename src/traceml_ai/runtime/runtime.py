# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Per-rank TraceML runtime agent."""

import sys
import threading
import time
from typing import Any, Callable, List, Optional

from traceml_ai.loggers.error_log import get_error_logger, setup_error_logger
from traceml_ai.runtime.config import config
from traceml_ai.runtime.exporter import TelemetryExporter
from traceml_ai.runtime.identity import resolve_runtime_identity
from traceml_ai.runtime.sampler_registry import build_samplers
from traceml_ai.runtime.sender import SenderIdentity, TelemetryPublisher
from traceml_ai.runtime.state import (
    TraceMLRecordingStatus,
    configure_trace_recording,
    get_trace_recording_state,
    mark_trace_recording_drained,
)
from traceml_ai.samplers.base_sampler import BaseSampler
from traceml_ai.telemetry.control import build_rank_finished_payload
from traceml_ai.transport.tcp_transport import TCPClient, TCPConfig

from .settings import TraceMLSettings

# At a normal stop, queue-backed samplers are ticked again while they still
# hold unresolved data, for at most this long. `stop()` extends its sampler
# join timeout by the same amount.
_STOP_DRAIN_MAX_SEC = 2.0
_STOP_DRAIN_POLL_SEC = 0.02
# How often a flush waiter re-checks that the loop is still running.
_FLUSH_POLL_SEC = 0.05


def _safe(logger, label: str, fn: Callable[[], Any]) -> Any:
    """Execute `fn()` and log exceptions; never raise."""
    try:
        return fn()
    except Exception as e:
        logger.error(f"[TraceML] {label}: {e}")
        return None


class TraceMLRuntime:
    """Runs samplers and publishes telemetry from one training rank."""

    def __init__(
        self,
        settings: Optional[TraceMLSettings] = None,
    ) -> None:
        self._settings = settings or TraceMLSettings()
        configure_trace_recording(self._settings.trace_max_steps)

        # Global config
        config.enable_logging = bool(self._settings.enable_logging)
        config.logs_dir = str(self._settings.logs_dir)
        config.session_id = self._settings.session_id

        self.mode = self._settings.mode
        self.profile = getattr(self._settings, "profile", "run")

        setup_error_logger(role="rank")
        self._logger = get_error_logger("TraceMLRuntime")

        # Runtime identity separates local device rank from globally unique
        # telemetry rank. Sampler selection still needs local rank, while
        # outbound payloads must identify the worker by global rank.
        self.identity = resolve_runtime_identity()
        self.local_rank = self.identity.local_rank
        self.global_rank = self.identity.global_rank
        self.world_size = self.identity.world_size

        # Stop event shared by all internal threads in this process
        self._stop_event = threading.Event()
        # Ends the sampler loop's wait before its interval is over. stop()
        # sets it too, so the loop still notices a stop at once.
        self._wake_event = threading.Event()
        # Numbers loop ticks and records the last one that published and the
        # last one that raised, so a flush can wait for a tick that began
        # after its request and learn how it ended.
        self._tick_cond = threading.Condition()
        self._ticks_started = 0
        self._last_published_tick = 0
        self._last_failed_tick = 0

        # Samplers (all ranks)
        self._samplers = self._build_samplers()

        # Transport: every rank has a TCP client
        self._tcp_client = TCPClient(
            TCPConfig(
                host=self._settings.aggregator.connect_host,
                port=int(self._settings.aggregator.port),
            )
        )
        # Exporter owns all TCP send/connect work on its own thread. The
        # publisher enqueues batches into it instead of sending inline.
        self._exporter = TelemetryExporter(
            tcp_client=self._tcp_client,
            logger=self._logger,
        )
        self._publisher = TelemetryPublisher(
            tcp_client=self._exporter,
            identity=SenderIdentity(
                global_rank=self.identity.global_rank,
                local_rank=self.identity.local_rank,
                world_size=self.identity.world_size,
                local_world_size=self.identity.local_world_size,
                node_rank=self.identity.node_rank,
                hostname=self.identity.hostname,
                pid=self.identity.pid,
                session_id=str(self._settings.session_id or ""),
                run_nonce=str(self._settings.run_nonce or ""),
                session_source=str(self._settings.session_source or ""),
            ),
            logger=self._logger,
        )
        self._publisher.attach_senders(self._samplers)

        # Sampler thread (per-rank)
        self._sampler_thread = threading.Thread(
            target=self._sampler_loop,
            name=f"TraceMLSampler(local_rank={self.local_rank})",
            daemon=True,
        )

    def _build_samplers(self) -> List[BaseSampler]:
        """
        Build samplers for this rank based on profile and UI mode using the
        runtime sampler registry.
        """
        return build_samplers(
            profile=self.profile,
            mode=self.mode,
            local_rank=self.local_rank,
            world_size=self.world_size,
            logger=self._logger,
        )

    def _tick(self, samplers: Optional[List[BaseSampler]] = None) -> None:
        """
        Run all samplers once and delegate publishing.

        Phase 1 — Sample + local DB write
        ----------------------------------
        Each sampler collects its metrics and writes to the local DB.
        Sampling failures are logged and skipped so user training continues.

        Phase 2 — Publish
        -----------------
        TelemetryPublisher flushes sampler writers, collects incremental
        payloads, and sends a single TCP batch.
        """
        target_samplers = self._samplers if samplers is None else samplers

        for sampler in target_samplers:
            _safe(
                self._logger,
                f"{sampler.sampler_name}.sample failed",
                sampler.sample,
            )

        self._publisher.publish(target_samplers)

    def _recording_drain_samplers(self) -> List[BaseSampler]:
        """Return the samplers marked for a final recording drain."""
        return [
            sampler
            for sampler in self._samplers
            if bool(getattr(sampler, "drain_on_recording_stop", False))
        ]

    def _drain_recording_samplers(self) -> None:
        """Drain queue-backed samplers after recording has stopped."""
        drain_samplers = self._recording_drain_samplers()
        if drain_samplers:
            self._tick(drain_samplers)
        pending = any(
            bool(sampler.has_pending_recording_data())
            for sampler in drain_samplers
        )
        if not pending:
            mark_trace_recording_drained()

    def _drain_pending_on_stop(self) -> bool:
        """
        Tick drain samplers until their pending data is sent, within a bound.

        Runs after the stop event is set, so it sleeps instead of waiting on
        the event. Resolution stays polling; nothing here blocks on CUDA.
        The bound keeps the drain inside the join timeout in `stop()`.

        Returns True when nothing is pending at exit, False at the deadline.
        """
        drain_samplers = self._recording_drain_samplers()
        failed_checks: set[int] = set()

        def _pending() -> List[BaseSampler]:
            pending = []
            for idx, sampler in enumerate(drain_samplers):
                try:
                    if sampler.has_pending_recording_data():
                        pending.append(sampler)
                except Exception as e:
                    # A raising check counts as not pending; log it once.
                    if idx not in failed_checks:
                        failed_checks.add(idx)
                        self._logger.error(
                            f"[TraceML] {sampler.sampler_name}"
                            f".has_pending_recording_data failed: {e}"
                        )
            return pending

        deadline = time.monotonic() + _STOP_DRAIN_MAX_SEC
        while True:
            pending = _pending()
            if not pending:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                names = ", ".join(s.sampler_name for s in pending)
                self._logger.error(
                    "[TraceML] WARNING: stop drain gave up after "
                    f"{_STOP_DRAIN_MAX_SEC}s with data still pending in: "
                    f"{names}"
                )
                return False
            time.sleep(min(_STOP_DRAIN_POLL_SEC, remaining))
            self._tick(drain_samplers)

    def _counted_tick(self, tick: Callable[[], None]) -> None:
        """Run one loop tick and record whether it published or raised."""
        with self._tick_cond:
            self._ticks_started += 1
            tick_id = self._ticks_started
        published = False
        try:
            tick()
            published = True
        finally:
            with self._tick_cond:
                if published:
                    self._last_published_tick = tick_id
                else:
                    self._last_failed_tick = tick_id
                self._tick_cond.notify_all()

    def _wait_for_next_tick(self) -> None:
        """Wait one sampler interval, or less when woken by flush or stop."""
        self._wake_event.wait(float(self._settings.sampler_interval_sec))
        self._wake_event.clear()

    def _sampler_loop(self) -> None:
        """Sampler loop (all ranks)."""
        while not self._stop_event.is_set():
            status = get_trace_recording_state().status
            if status == TraceMLRecordingStatus.COMPLETE:
                return
            if status == TraceMLRecordingStatus.DRAINING:
                self._counted_tick(self._drain_recording_samplers)
                if (
                    get_trace_recording_state().status
                    == TraceMLRecordingStatus.COMPLETE
                ):
                    return
                self._wait_for_next_tick()
                continue

            self._counted_tick(self._tick)
            self._wait_for_next_tick()

        status = get_trace_recording_state().status
        if status == TraceMLRecordingStatus.RECORDING:
            self._tick()
            self._drain_pending_on_stop()
        elif status == TraceMLRecordingStatus.DRAINING:
            self._drain_recording_samplers()
            if (
                self._drain_pending_on_stop()
                and get_trace_recording_state().status
                == TraceMLRecordingStatus.DRAINING
            ):
                mark_trace_recording_drained()

    def start(self) -> None:
        """Start the exporter, then the sampler thread."""
        # Exporter must be running before samplers start enqueuing batches.
        self._exporter.start()

        try:
            self._sampler_thread.start()
        except Exception as e:
            self._logger.exception("[TraceML] Sampler thread start failed")
            raise RuntimeError("Failed to start TraceML sampler thread") from e

    @property
    def stopped(self) -> bool:
        """True once ``stop()`` has been called."""
        return self._stop_event.is_set()

    def _synchronize_caller_cuda(self) -> None:
        """
        Wait for this rank's queued GPU work, on the caller's thread.

        The step-time sampler resolves a step only once its CUDA end event
        has completed. Waiting here lets those events resolve in the tick
        that follows. It costs one device synchronize at report time.
        Torch is reached only through ``sys.modules``: a process that never
        imported torch has no CUDA work queued, and this module stays
        importable without torch. Never raises; a failure is logged.
        """
        torch = sys.modules.get("torch")
        if torch is None:
            return
        try:
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
        except Exception as e:
            self._logger.error(
                f"[TraceML] CUDA synchronize before flush failed: {e}"
            )

    def flush_pending_telemetry(self, timeout_sec: float) -> bool:
        """
        Wake the sampler loop so it publishes this rank's pending telemetry.

        Returns True once a tick that began after this call has handed its
        batch to the exporter, and also at once when recording is complete,
        since the final drain has then already published everything.
        Returns False when that tick raised, when ``timeout_sec`` passes
        first, or when the loop is stopping, stopped, or ended otherwise.
        The caller's queued CUDA work is synchronized first so the tick can
        resolve this rank's step timings; other ranks are not affected.
        ``timeout_sec`` starts after that synchronize, so the wait for the
        tick keeps its full budget however long the GPU tail takes.
        The calling thread only signals the loop and waits; sampling stays
        on the sampler thread. Never raises, and never blocks longer than
        the device synchronize plus ``timeout_sec``.
        """
        try:
            self._synchronize_caller_cuda()
            deadline = time.monotonic() + max(0.0, float(timeout_sec))
            with self._tick_cond:
                target = self._ticks_started + 1
            self._wake_event.set()
            with self._tick_cond:
                while self._last_published_tick < target:
                    if self._last_failed_tick >= target:
                        return False
                    if self._recording_complete():
                        return True
                    if (
                        self._stop_event.is_set()
                        or not self._sampler_thread.is_alive()
                    ):
                        return False
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        return False
                    self._tick_cond.wait(min(remaining, _FLUSH_POLL_SEC))
            return True
        except Exception as e:
            self._logger.error(
                f"[TraceML] flush_pending_telemetry failed: {e}"
            )
            return False

    @staticmethod
    def _recording_complete() -> bool:
        """True once the final recording drain left nothing pending."""
        return (
            get_trace_recording_state().status
            == TraceMLRecordingStatus.COMPLETE
        )

    def stop(self) -> None:
        """
        Stop TraceML runtime and release resources (best effort).

        - Signals the sampler thread to stop
        - Joins the sampler thread
        - Enqueues a final telemetry batch and a rank-finished control message
        - Drains the export queue and closes the TCP client (bounded)
        """
        self._stop_event.set()
        self._wake_event.set()

        # stop sampler; its stop drain may use up to _STOP_DRAIN_MAX_SEC
        self._sampler_thread.join(
            timeout=float(self._settings.sampler_interval_sec) * 5.0
            + _STOP_DRAIN_MAX_SEC
        )

        if self._sampler_thread.is_alive():
            self._logger.error(
                "[TraceML] WARNING: sampler thread did not terminate"
            )

        self._publisher.publish(self._samplers)
        self._publisher.send_control(
            build_rank_finished_payload(
                global_rank=self.identity.global_rank,
                world_size=self.identity.world_size,
                node_rank=self.identity.node_rank,
                hostname=self.identity.hostname,
                session_id=str(self._settings.session_id or ""),
                run_nonce=str(self._settings.run_nonce or ""),
                session_source=str(self._settings.session_source or ""),
            )
        )

        # drain the export queue and close the client last
        self._exporter.stop()

    def log_summaries(self, path: Optional[str] = None) -> None:
        """
        Log summaries (rank0 only).

        With the 'store-only' design, summaries should be implemented in renderers
        or in the aggregator; the runtime itself doesn't compute summaries.
        """
        # Intentionally no-op here. Keep the method to avoid breaking callers.
        pass
