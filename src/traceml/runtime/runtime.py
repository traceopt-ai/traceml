"""
TraceML runtime + rank-0 aggregator (single-node DDP / single-process).

1) TraceMLRuntime (per-rank)
   - Runs samplers periodically in a background thread.
   - Flushes local sampler DB writers (temporary; can be removed later).
   - Ships incremental DB rows to the rank-0 aggregator over TCP (ALWAYS).

2) TraceMLAggregator (rank-0 only)
   - Owns the unified telemetry store: RemoteDBStore.
   - Runs a TCP server to receive rows from all ranks (including rank 0).
   - Builds renderers and drives UI updates. Renderers read ONLY from RemoteDBStore.

Why TCP is always used
----------------------
Even in single-process mode (WORLD_SIZE=1), rank 0 sends its own telemetry to the
aggregator via TCP loopback. This avoids shared-memory coupling between sampler
and rendering paths and makes it trivial to move the aggregator into a separate
process later.

Threading model
---------------
- Per rank: exactly one sampler thread (TraceMLRuntime).
- Rank 0 only: one aggregator thread (TraceMLAggregator).
These two threads share almost no data structures; communication is via TCP.
"""

import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type


# Telemetry transport + store
from traceml.database.remote_database_store import RemoteDBStore
from traceml.loggers.error_log import get_error_logger, setup_error_logger
from traceml.renderers.base_renderer import BaseRenderer

# Display backends
from traceml.renderers.display.managers.cli_display_manager import (
    CLIDisplayManager,
)
from traceml.renderers.display.managers.nicegui_display_manager import (
    NiceGUIDisplayManager,
)
from traceml.renderers.display.managers.notebook_display_manager import (
    NotebookDisplayManager,
)
from traceml.renderers.layer_combined_memory.renderer import (
    LayerCombinedMemoryRenderer,
)
from traceml.renderers.layer_combined_time.renderer import (
    LayerCombinedTimeRenderer,
)
from traceml.renderers.process.renderer import ProcessRenderer
from traceml.renderers.step_combined.renderer import StepCombinedRenderer

# Renderers (IMPORTANT: must read ONLY from RemoteDBStore)
from traceml.renderers.system.renderer import SystemRenderer
from traceml.renderers.step_memory.renderer import StepMemoryRenderer
from traceml.renderers.stdout_stderr_renderer import StdoutStderrRenderer
from traceml.runtime.config import config
from traceml.runtime.stdout_stderr_capture import StreamCapture
from traceml.samplers.base_sampler import BaseSampler
from traceml.samplers.layer_backward_memory_sampler import (
    LayerBackwardMemorySampler,
)
from traceml.samplers.layer_backward_time_sampler import (
    LayerBackwardTimeSampler,
)
from traceml.samplers.layer_forward_memory_sampler import (
    LayerForwardMemorySampler,
)
from traceml.samplers.layer_forward_time_sampler import LayerForwardTimeSampler
from traceml.samplers.layer_memory_sampler import LayerMemorySampler
from traceml.samplers.process_sampler import ProcessSampler
from traceml.samplers.step_memory_sampler import StepMemorySampler
from traceml.samplers.stdout_stderr_sampler import StdoutStderrSampler

# Samplers
from traceml.samplers.system_sampler import SystemSampler
from traceml.samplers.time_sampler import TimeSampler
from traceml.transport.distributed import get_ddp_info
from traceml.transport.tcp_transport import TCPClient, TCPConfig, TCPServer

from .session import get_session_id


# Configuration (runtime-level)
@dataclass(frozen=True)
class TraceMLTCPSettings:
    """TCP configuration for TraceML telemetry transport."""

    host: str = "127.0.0.1"
    port: int = 29765


@dataclass(frozen=True)
class TraceMLSettings:
    """
    High-level TraceML runtime settings.

    Notes:
    - `sampler_interval_sec` controls sampling cadence (all ranks).
    - `render_interval_sec` controls UI cadence (rank0 only).
    - TCP is always used for telemetry, including rank0 -> rank0.
    """

    mode: str = "cli"  # "cli" | "notebook" | "dashboard"
    sampler_interval_sec: float = 1.0
    render_interval_sec: float = 1.0
    num_display_layers: int = 20
    logs_dir: str = "./logs"
    enable_logging: bool = False
    remote_max_rows: int = 200
    tcp: TraceMLTCPSettings = TraceMLTCPSettings()
    session_id: str = ""


# Display backend registry
_DISPLAY_BACKENDS: Dict[str, Tuple[Type[Any], str]] = {
    "cli": (CLIDisplayManager, "get_panel_renderable"),
    "notebook": (NotebookDisplayManager, "get_notebook_renderable"),
    "dashboard": (NiceGUIDisplayManager, "get_dashboard_renderable"),
}


# Internal helpers
def _safe(logger, label: str, fn: Callable[[], Any]) -> Any:
    """Execute `fn()` and log exceptions; never raise."""
    try:
        return fn()
    except Exception as e:
        logger.error(f"[TraceML] {label}: {e}")
        return None


# Rank-0 Aggregator
class TraceMLAggregator:
    """
    Rank-0 telemetry aggregator + renderer driver.

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
            mode=settings.mode,
            num_display_layers=settings.num_display_layers,
            remote_store=self._store,
        )

        self._registered = False
        self._thread = threading.Thread(
            target=self._loop,
            name="TraceMLAggregator(rank0)",
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

    @staticmethod
    def _build_renderers(
        mode: str,
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
        if mode == "cli" :
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


# Per-rank Runtime
class TraceMLRuntime:
    """
    Per-rank TraceML runtime.

    - Creates samplers and runs them periodically in a sampler thread.
    - Attaches DBIncrementalSender to sampler DBs that support sending.
    - Flushes senders every tick (rank0 sends to rank0 as well).
    - On rank0, starts a TraceMLAggregator thread that owns store + renderers.

    This class intentionally avoids shared data structures between sampler and
    rendering logic: all cross-thread communication is via TCP.
    """

    def __init__(
        self,
        settings: Optional[TraceMLSettings] = None,
    ) -> None:
        self._settings = settings or TraceMLSettings()

        # Global config
        config.enable_logging = bool(self._settings.enable_logging)
        config.logs_dir = str(self._settings.logs_dir)
        config.session_id = self._settings.session_id or get_session_id()

        setup_error_logger()
        self._logger = get_error_logger("TraceMLRuntime")

        # DDP identity
        self.is_ddp, self.local_rank, self.world_size = get_ddp_info()
        self.is_rank0 = (not self.is_ddp) or self.local_rank == 0

        # Stop event shared by all internal threads in this process
        self._stop_event = threading.Event()

        # Samplers (all ranks)
        self._samplers = self._build_samplers()

        # Transport: every rank has a TCP client (rank0 -> rank0 included)
        self._tcp_client = TCPClient(
            TCPConfig(
                host=self._settings.tcp.host, port=int(self._settings.tcp.port)
            )
        )
        self._attach_senders()

        # Rank0 aggregator
        self._aggregator: Optional[TraceMLAggregator] = None
        if self.is_rank0:
            self._aggregator = TraceMLAggregator(
                logger=self._logger,
                stop_event=self._stop_event,
                settings=self._settings,
            )

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
            TimeSampler(),
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
            sampler.sender.sender = self._tcp_client
            sampler.sender.rank = self.local_rank

    def _tick(self) -> None:
        """
        Run all samplers once and flush local writers + telemetry senders.

        Note:
        - Local DB writes are temporary and can be removed as you migrate to a
          store-only architecture.
        - The telemetry sender flush is the primary pipeline.
        """
        for sampler in self._samplers:
            _safe(
                self._logger,
                f"{sampler.sampler_name}.sample failed",
                sampler.sample,
            )

            db = getattr(sampler, "db", None)
            if db is None:
                continue

            _safe(
                self._logger,
                f"{sampler.sampler_name}.writer.flush failed",
                db.writer.flush,
            )

            sender = getattr(sampler, "sender", None)
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
        1) enable stdout/stderr capture
        2) start aggregator (rank0) so TCP server is listening
        3) start sampler thread
        """
        _safe(
            self._logger,
            "Stdout/stderr capture enable failed",
            StreamCapture.redirect_to_capture,
        )

        if self._aggregator is not None:
            self._aggregator.start()
        _safe(
            self._logger,
            "Sampler thread start failed",
            self._sampler_thread.start,
        )

    def stop(self) -> None:
        """
        Stop TraceML runtime and release resources (best effort).
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

        # stop aggregator
        if self._aggregator is not None:
            self._aggregator.stop(
                timeout_sec=float(self._settings.render_interval_sec) * 5.0
            )

        # close client last
        _safe(self._logger, "TCPClient.close failed", self._tcp_client.close)

        # restore stdout/stderr
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
