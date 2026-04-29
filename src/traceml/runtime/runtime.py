"""
TraceML runtime  (per-rank agent).

This module implements the *per-rank* TraceML runtime that runs alongside the
user's training code (inside the torchrun worker process). It is responsible
for:

- Periodically running samplers in a dedicated background thread
- Flushing sampler writers (temporary legacy path; can be removed later)
- Shipping incremental telemetry rows to an out-of-process aggregator via TCP

Design notes
------------
- The runtime is intentionally "agent-only": it does NOT run a TCP server,
  it does NOT own the unified store, and it does NOT render UI.
- Cross-process communication is TCP. The aggregator (separate process) owns:
  TCPServer + RemoteDBStore + renderers + display.
- Even for WORLD_SIZE=1, telemetry is sent over loopback TCP to keep the same
  code-path and allow future remote aggregators without refactoring.

Failure behavior
----------------
If the aggregator becomes unavailable during training, sender flush failures are
logged and ignored. Training should proceed normally.
"""

import threading
from typing import Any, Callable, List, Optional

from traceml.loggers.error_log import get_error_logger, setup_error_logger
from traceml.runtime.config import config
from traceml.runtime.sampler_registry import build_samplers
from traceml.runtime.sender import TelemetryPublisher
from traceml.runtime.stdout_stderr_capture import StreamCapture
from traceml.samplers.base_sampler import BaseSampler
from traceml.transport.distributed import get_ddp_info
from traceml.transport.tcp_transport import TCPClient, TCPConfig

from .settings import TraceMLSettings


def _safe(logger, label: str, fn: Callable[[], Any]) -> Any:
    """Execute `fn()` and log exceptions; never raise."""
    try:
        return fn()
    except Exception as e:
        logger.error(f"[TraceML] {label}: {e}")
        return None


class TraceMLRuntime:
    """
    Per-rank TraceML runtime (agent).

    Responsibilities:
    - Creates samplers and runs them periodically.
    - Attaches DBIncrementalSender to sampler DBs that support sending.
    - Flushes senders every tick (rank0 sends to rank0 as well).

    This runtime intentionally does not share data structures with the
    aggregator. The aggregator is an out-of-process component that receives
    rows over TCP and renders UI from a unified RemoteDBStore.
    """

    def __init__(
        self,
        settings: Optional[TraceMLSettings] = None,
    ) -> None:
        self._settings = settings or TraceMLSettings()

        # Global config
        config.enable_logging = bool(self._settings.enable_logging)
        config.logs_dir = str(self._settings.logs_dir)
        config.session_id = self._settings.session_id

        self.mode = self._settings.mode
        self.profile = getattr(self._settings, "profile", "run")

        setup_error_logger()
        self._logger = get_error_logger("TraceMLRuntime")

        # DDP identity
        self.is_ddp, self.local_rank, self.world_size = get_ddp_info()

        # Stop event shared by all internal threads in this process
        self._stop_event = threading.Event()

        # Samplers (all ranks)
        self._samplers = self._build_samplers()

        # Transport: every rank has a TCP client
        self._tcp_client = TCPClient(
            TCPConfig(
                host=self._settings.tcp.host, port=int(self._settings.tcp.port)
            )
        )
        self._publisher = TelemetryPublisher(
            tcp_client=self._tcp_client,
            rank=self.local_rank,
            logger=self._logger,
        )
        self._publisher.attach_senders(self._samplers)

        # Sampler thread (per-rank)
        self._sampler_thread = threading.Thread(
            target=self._sampler_loop,
            name=f"TraceMLSampler(rank={self.local_rank})",
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
            is_ddp=self.is_ddp,
            local_rank=self.local_rank,
            logger=self._logger,
        )

    def _tick(self) -> None:
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
        for sampler in self._samplers:
            _safe(
                self._logger,
                f"{sampler.sampler_name}.sample failed",
                sampler.sample,
            )

        self._publisher.publish(self._samplers)

    def _sampler_loop(self) -> None:
        """Sampler loop (all ranks)."""
        while not self._stop_event.is_set():
            self._tick()
            self._stop_event.wait(float(self._settings.sampler_interval_sec))

        # final tick
        self._tick()

    def start(self) -> None:
        """
        Start TraceML runtime.

        Start order:
        1) enable stdout/stderr capture (CLI mode only, dashboard no need)
        2) start sampler thread
        """
        if self.mode == "cli":
            _safe(
                self._logger,
                "Stdout/stderr capture enable failed",
                StreamCapture.redirect_to_capture,
            )

        try:
            self._sampler_thread.start()
        except Exception as e:
            self._logger.exception("[TraceML] Sampler thread start failed")
            raise RuntimeError("Failed to start TraceML sampler thread") from e

    def stop(self) -> None:
        """
        Stop TraceML runtime and release resources (best effort).

        - Signals the sampler thread to stop
        - Joins the sampler thread
        - Closes TCP client
        - Restores stdout/stderr (CLI mode only)
        """
        self._stop_event.set()

        # stop sampler
        self._sampler_thread.join(
            timeout=float(self._settings.sampler_interval_sec) * 5.0
        )

        if self._sampler_thread.is_alive():
            self._logger.error(
                "[TraceML] WARNING: sampler thread did not terminate"
            )

        # close client last
        self._publisher.close()

        # restore stdout/stderr
        if self.mode == "cli":
            _safe(
                self._logger,
                "Stdout/stderr restore failed",
                StreamCapture.redirect_to_original,
            )

    def log_summaries(self, path: Optional[str] = None) -> None:
        """
        Log summaries (rank0 only).

        With the 'store-only' design, summaries should be implemented in renderers
        or in the aggregator; the runtime itself doesn't compute summaries.
        """
        # Intentionally no-op here. Keep the method to avoid breaking callers.
        pass
