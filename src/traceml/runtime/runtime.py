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
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from traceml.loggers.error_log import get_error_logger, setup_error_logger
from traceml.runtime.config import config
from traceml.runtime.stdout_stderr_capture import StreamCapture
from traceml.samplers.base_sampler import BaseSampler
from traceml.samplers.layer_backward_memory_sampler import LayerBackwardMemorySampler
from traceml.samplers.layer_backward_time_sampler import LayerBackwardTimeSampler
from traceml.samplers.layer_forward_memory_sampler import LayerForwardMemorySampler
from traceml.samplers.layer_forward_time_sampler import LayerForwardTimeSampler
from traceml.samplers.layer_memory_sampler import LayerMemorySampler
from traceml.samplers.process_sampler import ProcessSampler
from traceml.samplers.step_memory_sampler import StepMemorySampler
from traceml.samplers.stdout_stderr_sampler import StdoutStderrSampler
from traceml.samplers.system_sampler import SystemSampler
from traceml.samplers.step_time_sampler import StepTimeSampler
from traceml.transport.distributed import get_ddp_info
from traceml.transport.tcp_transport import TCPClient, TCPConfig

from .session import get_session_id
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
    - On rank0, starts a TraceMLAggregator thread that owns store + renderers.

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
        self._attach_senders()

        # Sampler thread (per-rank)
        self._sampler_thread = threading.Thread(
            target=self._sampler_loop,
            name=f"TraceMLSampler(rank={self.local_rank})",
            daemon=True,
        )

    def _build_samplers(self) -> List[BaseSampler]:
        """
        Build default samplers for this rank.

        SystemSampler only runs on rank0 to avoid duplicating host-level metrics.
        """
        is_ddp, local_rank, _ = get_ddp_info()
        samplers: List[BaseSampler] = []

        if not (is_ddp and local_rank != 0):
            samplers.append(SystemSampler())

        samplers += [
            ProcessSampler(),
            LayerMemorySampler(),
            LayerForwardMemorySampler(),
            LayerBackwardMemorySampler(),
            LayerForwardTimeSampler(),
            LayerBackwardTimeSampler(),
            StepTimeSampler(),
            StepMemorySampler(),
            StdoutStderrSampler(),
        ]
        return samplers


    def _attach_senders(self) -> None:
        """
        Attach DBIncrementalSender to sampler DBs that support sending.

        All ranks attach senders, including rank0, so that rank0 sends its own
        rows through the same TCP pipeline as worker ranks.
        """
        for sampler in self._samplers:
            if not getattr(sampler, "sender", None):
                continue
            # sender has attributes: sender.sender (transport) and sender.rank
            sampler.sender.sender = self._tcp_client
            sampler.sender.rank = self.local_rank


    def _tick(self) -> None:
        """
        Run all samplers once and flush local writers + telemetry senders.

        Note:
        - Local DB writes are temporary and can be removed as we migrate to a
          store-only architecture.
        - The telemetry sender flush is the primary pipeline.
        - Sender failures should not break training (best-effort telemetry).
        """
        for sampler in self._samplers:
            _safe(
                self._logger,
                f"{sampler.sampler_name}.sample failed",
                sampler.sample,
            )

            db = getattr(sampler, "db", None)
            if db is not None:
                _safe(
                    self._logger,
                    f"{sampler.sampler_name}.writer.flush failed",
                    db.writer.flush,
                )

            sender = getattr(sampler, "sender", None) # stdout aggregation is unnecessary
            if sender is not None:
                _safe(
                    self._logger,
                    f"{sampler.sampler_name}.sender.flush failed",
                    sender.flush,
                )

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

        _safe(
            self._logger,
            "Sampler thread start failed",
            self._sampler_thread.start,
        )

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
        _safe(self._logger, "final tick failed", self._tick)

        if self._sampler_thread.is_alive():
            self._logger.error("[TraceML] WARNING: sampler thread did not terminate")

        # close client last
        _safe(self._logger, "TCPClient.close failed", self._tcp_client.close)

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
