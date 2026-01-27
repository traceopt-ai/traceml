"""
TraceML runtime + rank-0 aggregator (single-node DDP / single-process).

This module provides a **clean, production-ready separation** between:

1) TraceMLRuntime (per-rank, hot path)
   - Runs samplers periodically in a background thread.
   - Flushes local sampler DB writers (temporary; can be removed later).
   - Ships incremental DB rows to the rank-0 aggregator over TCP (ALWAYS).
   - Does not render. Does not aggregate. Does not read any DB for display.

2) TraceMLAggregator (rank-0 only, cold path)
   - Owns the unified telemetry store: RemoteDBStore.
   - Runs a TCP server to receive rows from all ranks (including rank 0).
   - Builds renderers and drives UI updates. Renderers read ONLY from RemoteDBStore.
   - Does not run samplers and does not touch local sampler DB objects.

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

Assumptions / required adaptations
----------------------------------
This module assumes you will adapt renderer constructors to accept `remote_store`
ONLY, e.g. `StepTimerRenderer(remote_store=...)`, and they must not depend on any
sampler-local DB objects.

Similarly, `RemoteDBStore.ingest(msg)` must be able to ingest the payloads emitted
by DBIncrementalSender -> TCPClient -> TCPServer.
"""

`sdasdas CHANGE FROM HERE 

import threading
from typing import Callable, List, Optional, Tuple

from traceml.config import config
from traceml.distributed import get_ddp_info
from traceml.loggers.error_log import get_error_logger, setup_error_logger
from traceml.stdout_stderr_capture import StreamCapture

from .session import get_session_id

from traceml.samplers.base_sampler import BaseSampler
from traceml.renderers.base_renderer import BaseRenderer

# Samplers
from traceml.samplers.system_sampler import SystemSampler
from traceml.samplers.process_sampler import ProcessSampler
from traceml.samplers.step_memory_sampler import StepMemorySampler
from traceml.samplers.layer_memory_sampler import LayerMemorySampler
from traceml.samplers.layer_forward_memory_sampler import LayerForwardMemorySampler
from traceml.samplers.layer_backward_memory_sampler import LayerBackwardMemorySampler
from traceml.samplers.steptimer_sampler import StepTimerSampler
from traceml.samplers.layer_forward_time_sampler import LayerForwardTimeSampler
from traceml.samplers.layer_backward_time_sampler import LayerBackwardTimeSampler
from traceml.samplers.stdout_stderr_sampler import StdoutStderrSampler

# Renderers
from traceml.renderers.system.system_renderer import SystemRenderer
from traceml.renderers.process_renderer import ProcessRenderer
from traceml.renderers.layer_combined_memory_renderer import LayerCombinedMemoryRenderer
from traceml.renderers.steptimer_renderer import StepTimerRenderer
from traceml.renderers.model_combined_renderer import ModelCombinedRenderer
from traceml.renderers.layer_combined_timing_renderer import LayerCombinedTimerRenderer
from traceml.renderers.stdout_stderr_renderer import StdoutStderrRenderer

# Display backends
from traceml.renderers.display.cli_display_manager import CLIDisplayManager
from traceml.renderers.display.notebook_display_manager import NotebookDisplayManager
from traceml.renderers.display.nicegui_display_manager import NiceGUIDisplayManager

# DDP telemetry
from traceml.database.remote_database_store import RemoteDBStore
from traceml.database.database_sender import DBIncrementalSender
from traceml.tcp_transport import TCPServer, TCPClient, TCPConfig



_DISPLAY_BACKENDS = {
    "cli": (CLIDisplayManager, "get_panel_renderable"),
    "notebook": (NotebookDisplayManager, "get_notebook_renderable"),
    "dashboard": (NiceGUIDisplayManager, "get_dashboard_renderable"),
}

class TraceMLAggregator:
    """
    Rank-0 aggregator + rendering loop.

    This component is rank0-only and is responsible for:
    - draining remote TCP telemetry (workers -> rank0) into RemoteDBStore
    - registering renderer outputs into the display manager (once)
    - periodically calling display update (CLI / notebook / dashboard)

    It runs on a dedicated thread so sampling cadence is not coupled to UI cadence.
    """

    def __init__(
        self,
        logger,
        stop_event: threading.Event,
        interval_sec: float,
        display_manager,
        renderers: List[BaseRenderer],
        render_attr: str,
        enable_ddp_telemetry: bool,
        tcp_server: Optional[TCPServer],
        remote_store: Optional[RemoteDBStore],
    ) -> None:
        self.logger = logger
        self._stop_event = stop_event
        self.interval_sec = float(interval_sec)

        self.display_manager = display_manager
        self.renderers = renderers
        self._render_attr = render_attr

        self.enable_ddp_telemetry = bool(enable_ddp_telemetry)
        self._tcp_server = tcp_server
        self.remote_store = remote_store

        self._registered = False
        self._thread = threading.Thread(
            target=self._loop,
            name="TraceMLAggregator(rank0)",
            daemon=True,
        )

    def _safe(self, label: str, fn: Callable[[], object]) -> object:
        try:
            return fn()
        except Exception as e:
            self.logger.error(f"[TraceML] {label}: {e}")
            return None

    def start(self) -> None:
        """Start the rank-0 aggregator loop thread (best effort)."""
        self._safe("Aggregator thread start failed", self._thread.start)

    def stop(self, timeout_sec: float) -> None:
        """Stop (join) the aggregator loop thread (best effort)."""
        self._thread.join(timeout=timeout_sec)
        if self._thread.is_alive():
            self.logger.error("[TraceML] WARNING: aggregator thread did not terminate")

    def _register_renderers_once(self) -> None:
        """
        Register renderer content providers into the display manager once.
        """
        if self._registered or self.display_manager is None:
            return

        for r in self.renderers:
            def register(rr=r):
                render_fn = getattr(rr, self._render_attr)
                self.display_manager.register_layout_content(
                    rr.layout_section_name,
                    render_fn,
                )

            self._safe(f"{r.__class__.__name__}.register failed", register)
        self._registered = True

    def _drain_remote(self) -> None:
        """Drain TCP messages from workers and ingest into RemoteDBStore."""
        if not (self.enable_ddp_telemetry and self._tcp_server and self.remote_store):
            return

        for msg in self._tcp_server.poll():
            try:
                self.remote_store.ingest(msg)
            except Exception as e:
                self.logger.error(f"[TraceML] Remote ingest failed: {e}")


    def _render_and_update(self) -> None:
        """Update UI by calling display manager update."""
        if self.display_manager is None:
            return
        self._register_renderers_once()
        self._safe("Display update failed", self.display_manager.update_display)


    def _loop(self) -> None:
        """Aggregator loop: start display, then periodically drain + render."""
        if self.display_manager is not None:
            self._safe("Display start failed", self.display_manager.start_display)

        while not self._stop_event.is_set():
            self._drain_remote()
            self._render_and_update()
            self._stop_event.wait(self.interval_sec)

        # Final drain + render on shutdown.
        self._drain_remote()
        self._render_and_update()




class TraceMLRuntime:
    """
    TraceMLRuntime (per-rank).

    What it does
    ------------
    - Starts a sampler loop thread that periodically:
      - calls `.sample()` on each sampler
      - flushes sampler DB writers
      - on DDP worker ranks: flushes DBIncrementalSender over TCP (rank -> rank0)

    - On rank0, additionally starts a TraceMLAggregator thread that:
      - drains TCP telemetry (rank>0 -> rank0) into RemoteDBStore
      - updates the configured display backend

    Key properties
    --------------
    - Safe shutdown: best-effort stop; errors are logged, not raised.

    Tuning
    ------
    - `interval_sec`: sampler cadence (also default render cadence)
    - `render_interval_sec`: optional separate cadence for rank0 rendering
      (recommended to be slower than sampling if UI is heavy)
    """

    def __init__(
        self,
        samplers: Optional[List[BaseSampler]] = None,
        renderers: Optional[List[BaseRenderer]] = None,
        interval_sec: float = 1.0,
        mode: str = "cli",
        num_display_layers: int = 20,
        enable_logging: bool = False,
        logs_dir: str = "./logs",
        enable_ddp_telemetry: bool = False,
        remote_max_rows: int = 200,
        tcp_host: str = "127.0.0.1",
        tcp_port: int = 29765,
        session_id: str = "",
        render_interval_sec: Optional[float] = None,
    ):
        # ---- Global config ----
        config.enable_logging = enable_logging
        config.logs_dir = logs_dir
        config.session_id = session_id or get_session_id()

        setup_error_logger()
        self.logger = get_error_logger("TraceMLRuntime")

        # ---- DDP identity ----
        self.is_ddp, self.local_rank, self.world_size = get_ddp_info()
        self.is_rank0 = (not self.is_ddp) or self.local_rank == 0
        self.is_worker = self.is_ddp and self.local_rank != 0

        # ---- Cadence ----
        self.sampler_interval_sec = float(interval_sec)
        self.render_interval_sec = (
            float(render_interval_sec)
            if render_interval_sec is not None else float(interval_sec)
        )
        self.mode = mode

        # ---- Telemetry (rank0 aggregation store) ----
        self.enable_ddp_telemetry = bool(enable_ddp_telemetry and self.is_ddp)
        if self.enable_ddp_telemetry and self.is_rank0:
            self.remote_store = RemoteDBStore(max_rows=remote_max_rows)
        else:
            self.remote_store = None

        # ---- Display (rank0 only) ----
        self.display_manager_cls, self._render_attr = _DISPLAY_BACKENDS[mode]
        self.display_manager = self.display_manager_cls() if self.is_rank0 else None

        # ---- Build components ----
        if samplers is None or renderers is None:
            built_samplers, built_renderers = self._build_components(
                mode=mode,
                num_display_layers=num_display_layers,
                remote_store=self.remote_store,
            )
            self.samplers = built_samplers
            self.renderers = built_renderers if self.is_rank0 else []
        else:
            self.samplers = samplers
            self.renderers = renderers if self.is_rank0 else []

        # ---- TCP transport ----
        self._tcp_server: Optional[TCPServer] = None
        self._tcp_client: Optional[TCPClient] = None
        if self.enable_ddp_telemetry:
            self._init_ddp_transport_runtime(
                host=tcp_host,
                port=tcp_port,
                remote_max_rows=remote_max_rows,
            )

        # ---- Concurrency ----
        self._stop_event = threading.Event()

        # Sampler loop thread (all ranks)
        self._sampler_thread = threading.Thread(
            target=self._sampler_loop,
            name=f"TraceMLRuntime(rank={self.local_rank})",
            daemon=True,
        )

        # Aggregator loop thread (rank0 only)
        self._aggregator: Optional[TraceMLAggregator] = None
        if self.is_rank0:
            self._aggregator = TraceMLAggregator(
                logger=self.logger,
                stop_event=self._stop_event,
                interval_sec=self.render_interval_sec,
                display_manager=self.display_manager,
                renderers=self.renderers,
                render_attr=self._render_attr,
                enable_ddp_telemetry=self.enable_ddp_telemetry,
                tcp_server=self._tcp_server,
                remote_store=self.remote_store,
            )


    def _safe(self, label: str, fn: Callable[[], object]) -> object:
        try:
            return fn()
        except Exception as e:
            self.logger.error(f"[TraceML] {label}: {e}")
            return None

    def _init_ddp_transport_runtime(self, host: str, port: int, remote_max_rows: int):
        """
        Initialize TCP transport used for cross-rank telemetry.

        - Rank0 starts a TCPServer to receive incremental DB rows from workers.
        - Worker ranks create a TCPClient; samplers that support DDP sending
          get a DBIncrementalSender attached to their DB.
        """
        cfg = TCPConfig(host=host, port=port)

        if self.is_rank0:
            self._tcp_server = TCPServer(cfg)
            self._safe("TCPServer.start failed", self._tcp_server.start)

        # Worker ranks
        self._tcp_client = TCPClient(cfg)

        for sampler in self.samplers:
            if not getattr(sampler, "enable_ddp_send", False):
                continue

            db = getattr(sampler, "db", None)
            if db is None:
                continue

            sender = DBIncrementalSender(
                db=db,
                sampler_name=sampler.sampler_name,
                sender=self._tcp_client,
                rank=self.local_rank,
            )
            db.sender = sender

    @staticmethod
    def _build_components(
        mode: str,
        num_display_layers: int,
        remote_store: Optional[RemoteDBStore] = None,
    ) -> Tuple[List[BaseSampler], List[BaseRenderer]]:
        """
        Build default samplers + renderers.

        Notes:
        - SystemSampler runs only on rank0.
        - Renderers are created for rank0; workers will have empty renderer list.
        """
        is_ddp, local_rank, _ = get_ddp_info()

        samplers: List[BaseSampler] = []
        renderers: List[BaseRenderer] = []

        # Rank 0 only: system / host sampler
        if not (is_ddp and local_rank != 0):
            sys_sampler = SystemSampler()
            samplers.append(sys_sampler)
            renderers.append(SystemRenderer(database=sys_sampler.db))

        # Process sampler
        proc_sampler = ProcessSampler()
        samplers.append(proc_sampler)
        renderers.append(
            ProcessRenderer(database=proc_sampler.db, remote_store=remote_store)
        )

        # Layer memory
        layer_mem = LayerMemorySampler()
        fwd_mem = LayerForwardMemorySampler()
        bwd_mem = LayerBackwardMemorySampler()
        samplers += [layer_mem, fwd_mem, bwd_mem]
        renderers.append(
            LayerCombinedMemoryRenderer(
                layer_db=layer_mem.db,
                layer_forward_db=fwd_mem.db,
                layer_backward_db=bwd_mem.db,
                top_n_layers=num_display_layers,
                remote_store=remote_store,
            )
        )

        # Step memory + timers
        step_mem = StepMemorySampler()
        step_timer = StepTimerSampler()
        samplers += [step_mem, step_timer]
        renderers += [
            StepTimerRenderer(database=step_timer.db, remote_store=remote_store),
            ModelCombinedRenderer(
                time_db=step_timer.db,
                memory_db=step_mem.db,
                remote_store=remote_store,
            ),
        ]

        # Layer timing
        fwd_time = LayerForwardTimeSampler()
        bwd_time = LayerBackwardTimeSampler()
        samplers += [fwd_time, bwd_time]
        renderers.append(
            LayerCombinedTimerRenderer(
                forward_db=fwd_time.db,
                backward_db=bwd_time.db,
                top_n_layers=num_display_layers,
                remote_store=remote_store,
            )
        )

        std_sampler = StdoutStderrSampler()
        samplers += [std_sampler]
        if mode == "cli" and local_rank < 1:
            renderers += [StdoutStderrRenderer(database=std_sampler.db)]

        return samplers, renderers

    def _run_samplers_and_flush(self):
        """
        Run all samplers once and flush their writers.

        On DDP worker ranks with telemetry enabled, also flush DB sender to rank0.
        """
        for sampler in self.samplers:
            self._safe(f"{sampler.sampler_name}.sample failed", sampler.sample)

            db = getattr(sampler, "db", None)
            if db is None:
                continue

            self._safe(f"{sampler.sampler_name}.writer.flush failed", db.writer.flush)

            if self.enable_ddp_telemetry and db.sender is not None:
                self._safe(
                    f"{sampler.sampler_name}.sender.flush failed", db.sender.flush
                )

    def _sampler_loop(self) -> None:
        """Background sampler loop thread (all ranks)."""
        while not self._stop_event.is_set():
            self._run_samplers_and_flush()
            self._stop_event.wait(self.sampler_interval_sec)

        # Final pass on shutdown
        self._run_samplers_and_flush()

    def start(self) -> None:
        """
        Start TraceML background threads.

        - Enables stdout/stderr capture (process-wide)
        - Starts sampler loop (all ranks)
        - Starts aggregator loop (rank0 only)
        """
        self._safe("Stdout/stderr capture enable failed", StreamCapture.redirect_to_capture)

        self._safe("Failed to start sampler thread", self._sampler_thread.start)

        if self._aggregator is not None:
            self._aggregator.start()

    def stop(self) -> None:
        """
        Stop TraceML background threads and release resources.

        Best-effort:
        - never raises
        - logs warnings/errors
        """
        self._stop_event.set()

        # Join sampler thread
        self._sampler_thread.join(timeout=self.sampler_interval_sec * 5)
        if self._sampler_thread.is_alive():
            self.logger.error("[TraceML] WARNING: sampler thread did not terminate")

        # Join aggregator thread (rank0)
        if self._aggregator is not None:
            self._aggregator.stop(timeout_sec=self.render_interval_sec * 5)

        # Release display (rank0)
        if self.is_rank0 and self.display_manager is not None:
            self._safe("Display release failed", self.display_manager.release_display)

        # Stop TCP transport
        if self._tcp_server:
            self._safe("TCPServer.stop failed", self._tcp_server.stop)

        if self._tcp_client:
            self._safe("TCPClient.close failed", self._tcp_client.close)

        # Restore stdout/stderr
        self._safe("Stdout/stderr restore failed", StreamCapture.redirect_to_original)


    def log_summaries(self, path=None):
        if not self.is_rank0:
            return
        for r in self.renderers:
            self._safe(
                f"{r.__class__.__name__}.log_summary failed",
                lambda rr=r: rr.log_summary(path),
            )
