import threading
from typing import List, Optional, Tuple

from traceml.config import config
from traceml.loggers.error_log import get_error_logger, setup_error_logger
from traceml.distributed import get_ddp_info
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
from traceml.renderers.system_renderer import SystemRenderer
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

from traceml.stdout_stderr_capture import StreamCapture



import sys

_DISPLAY_BACKENDS = {
    "cli": (CLIDisplayManager, "get_panel_renderable"),
    "notebook": (NotebookDisplayManager, "get_notebook_renderable"),
    "dashboard": (NiceGUIDisplayManager, "get_dashboard_renderable"),
}


class TraceMLRuntime:
    """
    Single TraceML runtime for:
      - single-process
      - DDP single-node (CPU / GPU, NCCL or Gloo)

    Principles:
      - samplers run on all ranks
      - renderers + display run only on rank 0
      - workers send incremental DB rows via TCP
      - rank 0 stores remotes it in RemoteDBStore
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
    ):
        # Update global config
        config.enable_logging = enable_logging
        config.logs_dir = logs_dir
        if session_id:
            config.session_id = session_id
        else:
            config.session_id = get_session_id()

        setup_error_logger()
        self.logger = get_error_logger("TraceMLRuntime")

        self.is_ddp, self.local_rank, self.world_size = get_ddp_info()
        self.is_rank0 = (not self.is_ddp) or self.local_rank == 0
        self.is_worker = self.is_ddp and self.local_rank != 0

        self.interval_sec = float(interval_sec)
        self.mode = mode

        self.enable_ddp_telemetry = bool(enable_ddp_telemetry and self.is_ddp)
        if self.enable_ddp_telemetry and self.is_rank0:
            self.remote_store = RemoteDBStore(max_rows=remote_max_rows)
        else:
            self.remote_store = None

        self.display_manager_cls, self._render_attr = _DISPLAY_BACKENDS[mode]
        self.display_manager = self.display_manager_cls() if self.is_rank0 else None

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

        self._tcp_server: Optional[TCPServer] = None
        self._tcp_client: Optional[TCPClient] = None

        if self.enable_ddp_telemetry:
            self._init_ddp_transport_runtime(
                host=tcp_host,
                port=tcp_port,
                remote_max_rows=remote_max_rows,
            )


        # Runtime thread
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"TraceMLRuntime(rank={self.local_rank})",
            daemon=True,
        )


    def _safe(self, label: str, fn):
        try:
            return fn()
        except Exception as e:
            self.logger.error(f"[TraceML] {label}: {e}")
            return None


    def _init_ddp_transport_runtime(self, host: str, port: int, remote_max_rows: int):
        cfg = TCPConfig(host=host, port=port)

        if self.is_rank0:
            self._tcp_server = TCPServer(cfg)
            self._safe("TCPServer.start failed", self._tcp_server.start)
            return

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
        is_ddp, local_rank, _ = get_ddp_info()

        samplers: List[BaseSampler] = []
        renderers: List[BaseRenderer] = []

        # Rank 0 only: system / process
        if not (is_ddp and local_rank != 0):
            sys_sampler = SystemSampler()
            proc_sampler = ProcessSampler()
            samplers += [sys_sampler, proc_sampler]
            renderers += [
                SystemRenderer(database=sys_sampler.db),
                ProcessRenderer(database=proc_sampler.db),
            ]

        # Layer memory
        layer_mem = LayerMemorySampler()
        fwd_mem = LayerForwardMemorySampler()
        bwd_mem = LayerBackwardMemorySampler()
        samplers += [layer_mem, fwd_mem, bwd_mem]
        renderers += [
            LayerCombinedMemoryRenderer(
                layer_db=layer_mem.db,
                layer_forward_db=fwd_mem.db,
                layer_backward_db=bwd_mem.db,
                top_n_layers=num_display_layers,
                remote_store=remote_store,
            )
        ]

        # Step memory + timers
        step_mem = StepMemorySampler()
        step_timer = StepTimerSampler()
        samplers += [step_mem, step_timer]
        renderers += [
            StepTimerRenderer(database=step_timer.db),
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
        renderers += [
            LayerCombinedTimerRenderer(
                forward_db=fwd_time.db,
                backward_db=bwd_time.db,
                top_n_layers=num_display_layers,
                remote_store=remote_store,
            )
        ]

        std_sampler = StdoutStderrSampler()
        samplers += [std_sampler]
        if mode == "cli" and local_rank < 1:
            renderers += [StdoutStderrRenderer(database=std_sampler.db)]

        return samplers, renderers


    def _run_samplers_and_flush(self):
        for sampler in self.samplers:
            self._safe(f"{sampler.sampler_name}.sample failed", sampler.sample)

            db = getattr(sampler, "db", None)
            if db is None:
                continue

            self._safe(f"{sampler.sampler_name}.writer.flush failed", db.writer.flush)

            if self.is_worker and self.enable_ddp_telemetry and db.sender is not None:
                self._safe(f"{sampler.sampler_name}.sender.flush failed", db.sender.flush)

        if self.is_rank0 and self.display_manager:
            self._safe("Display update failed", self.display_manager.update_display)


    def _drain_remote(self):
        if not (self.is_rank0 and self.enable_ddp_telemetry and self._tcp_server):
            return

        for msg in self._tcp_server.poll():
            try:
                self.remote_store.ingest(msg)
            except Exception as e:
                self.logger.error(f"[TraceML] Remote ingest failed: {e}")


    def _render_and_update(self):
        if not self.is_rank0 or self.display_manager is None:
            return

        for renderer in self.renderers:
            def register(r=renderer):
                render_fn = getattr(r, self._render_attr)
                self.display_manager.register_layout_content(
                    r.layout_section_name,
                    render_fn,
                )

            self._safe(f"{renderer.__class__.__name__}.register failed", register)

        # self._safe("Display update failed", self.display_manager.update_display)


    def _run_once(self):
        self._run_samplers_and_flush()
        self._drain_remote()
        self._render_and_update()


    def _run(self):
        if self.is_rank0 and self.display_manager is not None:
            self._safe("Display start failed", self.display_manager.start_display)

        while not self._stop_event.is_set():
            self._run_once()
            self._stop_event.wait(self.interval_sec)

        self._run_once()


    def start(self):
        # Enable stdout/stderr capture for the whole process
        self._safe(
            "Stdout/stderr capture enable failed",
            StreamCapture.redirect_to_capture,
        )
        self._safe("Failed to start TraceMLRuntime", self._thread.start)

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=self.interval_sec * 5)

        if self._thread.is_alive():
            self.logger.error("[TraceML] WARNING: runtime thread did not terminate")

        if self.is_rank0 and self.display_manager is not None:
            self._safe("Display release failed", self.display_manager.release_display)

        if self._tcp_server:
            self._safe("TCPServer.stop failed", self._tcp_server.stop)

        if self._tcp_client:
            self._safe("TCPClient.close failed", self._tcp_client.close)

        self._safe(
            "Stdout/stderr restore failed",
            StreamCapture.redirect_to_original,
        )



    def log_summaries(self, path=None):
        if not self.is_rank0:
            return
        for r in self.renderers:
            self._safe(
                f"{r.__class__.__name__}.log_summary failed",
                lambda rr=r: rr.log_summary(path),
            )