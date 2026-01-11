import threading
from typing import List, Optional, Tuple

from traceml.config import config
from traceml.loggers.error_log import get_error_logger, setup_error_logger

from traceml.distributed import get_ddp_info
from traceml.samplers.base_sampler import BaseSampler
from traceml.samplers.system_sampler import SystemSampler
from traceml.samplers.process_sampler import ProcessSampler
from traceml.samplers.step_memory_sampler import StepMemorySampler
from traceml.samplers.layer_memory_sampler import LayerMemorySampler
from traceml.samplers.layer_forward_memory_sampler import LayerForwardMemorySampler
from traceml.samplers.layer_backward_memory_sampler import LayerBackwardMemorySampler
from traceml.samplers.steptimer_sampler import StepTimerSampler
from traceml.samplers.layer_forward_time_sampler import LayerForwardTimeSampler
from traceml.samplers.layer_backward_time_sampler import LayerBackwardTimeSampler

from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.system_renderer import SystemRenderer
from traceml.renderers.process_renderer import ProcessRenderer
from traceml.renderers.layer_combined_memory_renderer import (
    LayerCombinedMemoryRenderer,
)
from traceml.renderers.steptimer_renderer import StepTimerRenderer
from traceml.renderers.model_combined_renderer import ModelCombinedRenderer
from traceml.renderers.layer_combined_timing_renderer import (
    LayerCombinedTimerRenderer,
)
from traceml.renderers.stdout_stderr_renderer import StdoutStderrRenderer

from traceml.renderers.display.cli_display_manager import CLIDisplayManager
from traceml.renderers.display.notebook_display_manager import NotebookDisplayManager
from traceml.renderers.display.nicegui_display_manager import NiceGUIDisplayManager


_DISPLAY_BACKENDS = {
    "cli": (CLIDisplayManager, "get_panel_renderable"),
    "notebook": (NotebookDisplayManager, "get_notebook_renderable"),
    "dashboard": (NiceGUIDisplayManager, "get_dashboard_renderable"),
}


class TraceMLRuntime:
    """
    Central coordinator for TraceML runtime components.

    Responsibilities:
      - own and run samplers (data collection)
      - own and run renderers (visualization / summaries)
      - manage display lifecycle
      - ensure tracing never crashes user training
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
    ):
        # Global TraceML configuration
        config.enable_logging = enable_logging
        config.logs_dir = logs_dir

        # Centralized error logging
        setup_error_logger()
        self.logger = get_error_logger("TraceMLRuntime")

        # Resolve display backend
        try:
            self.display_manager_cls, self._render_attr = _DISPLAY_BACKENDS[mode]
        except KeyError:
            raise ValueError(
                f"Unsupported mode '{mode}'. Choose from {list(_DISPLAY_BACKENDS)}"
            )

        # Build default samplers / renderers unless explicitly provided
        if samplers is None or renderers is None:
            self.samplers, self.renderers = self._build_components(
                mode=mode,
                num_display_layers=num_display_layers,
            )
        else:
            self.samplers = samplers
            self.renderers = renderers

        self.display_manager = self.display_manager_cls()
        self.interval_sec = interval_sec

        # Background execution
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="TraceMLRuntimeThread",
            daemon=True,
        )

    def _safe(self, label: str, fn):
        """
        Execute a callable and swallow any exception.

        TraceML must never crash the training process.
        """
        try:
            return fn()
        except Exception as e:
            self.logger.error(f"[TraceML] {label}: {e}")
            return None

    @staticmethod
    def _build_components(
        mode: str,
        num_display_layers: int,
    ) -> Tuple[List[BaseSampler], List[BaseRenderer]]:
        """
        Construct default samplers and renderers.

        This function encodes TraceML's default observability stack.
        """
        is_ddp, local_rank, _ = get_ddp_info()

        samplers: List[BaseSampler] = []
        renderers: List[BaseRenderer] = []

        # System / process metrics (rank 0 only in DDP)
        if not (is_ddp and local_rank != 0):
            sys_sampler = SystemSampler()
            proc_sampler = ProcessSampler()

            samplers += [sys_sampler, proc_sampler]
            renderers += [
                SystemRenderer(database=sys_sampler.db),
                ProcessRenderer(database=proc_sampler.db),
            ]

        # Layer memory (forward + backward combined)
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
            )
        ]

        # Step-level memory
        step_mem = StepMemorySampler()
        samplers += [step_mem]

        # Step timing (model + per-step)
        step_timer = StepTimerSampler()
        samplers += [step_timer]
        renderers += [
            StepTimerRenderer(database=step_timer.db),
            ModelCombinedRenderer(
                time_db=step_timer.db,
                memory_db=step_mem.db,
            ),
        ]

        # Layer timing (forward + backward combined)
        fwd_time = LayerForwardTimeSampler()
        bwd_time = LayerBackwardTimeSampler()

        samplers += [fwd_time, bwd_time]
        renderers += [
            LayerCombinedTimerRenderer(
                forward_db=fwd_time.db,
                backward_db=bwd_time.db,
                top_n_layers=num_display_layers,
            )
        ]

        # CLI-only stdout / stderr capture
        if mode == "cli":
            renderers.append(StdoutStderrRenderer())

        return samplers, renderers

    def _run_samplers(self):
        """
        Run all samplers once and flush their databases.
        """
        for sampler in self.samplers:
            self._safe(
                f"Sampler {sampler.__class__.__name__}.sample() failed",
                sampler.sample,
            )

            if getattr(sampler, "db", None) is not None:
                self._safe(
                    f"Sampler {sampler.__class__.__name__}.db.flush() failed",
                    sampler.db.writer.flush,
                )
            else:
                print(f"DB IS MISSING {sampler.__class__.__name__}")

    def _run_renderers(self):
        """
        Register renderers with the active display backend.
        """
        for renderer in self.renderers:

            def register(r=renderer):
                render_fn = getattr(r, self._render_attr)
                self.display_manager.register_layout_content(
                    r.layout_section_name,
                    render_fn,
                )

            self._safe(
                f"Renderer {renderer.__class__.__name__} register failed",
                register,
            )

    def _run_once(self):
        """
        Single tracking iteration:
          - sample
          - render
          - update display
        """
        self._run_samplers()
        self._run_renderers()
        self._safe(
            "Display update failed",
            self.display_manager.update_display,
        )

    def _run(self):
        """
        Background tracking loop.
        """
        self._safe(
            "Display start failed",
            self.display_manager.start_display,
        )

        while not self._stop_event.is_set():
            self._run_once()
            self._stop_event.wait(self.interval_sec)

        # Final flush
        self._run_once()

    # ------------------------------------------------------------------
    # Public lifecycle API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Start background tracking.
        """
        self._safe(
            "Failed to start TraceMLRuntime thread",
            self._thread.start,
        )

    def stop(self) -> None:
        """
        Stop tracking and release display resources.
        """
        self._stop_event.set()
        self._thread.join(timeout=self.interval_sec * 5)

        if self._thread.is_alive():
            self.logger.error(
                "[TraceML] WARNING: Tracker thread did not terminate in time."
            )

        self._safe(
            "Display release failed",
            self.display_manager.release_display,
        )

    def log_summaries(self, path=None) -> None:
        """
        Persist renderer summaries to disk (optional).
        """
        for renderer in self.renderers:
            self._safe(
                f"Renderer {renderer.__class__.__name__}.log_summary failed",
                lambda r=renderer: r.log_summary(path),
            )
