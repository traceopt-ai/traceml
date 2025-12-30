import threading
from typing import List, Optional
from traceml.loggers.error_log import get_error_logger, setup_error_logger
from traceml.config import config
from traceml.renderers.display.cli_display_manager import CLIDisplayManager
from traceml.renderers.display.notebook_display_manager import NotebookDisplayManager
from traceml.renderers.display.nicegui_display_manager import NiceGUIDisplayManager

from traceml.utils.distributed import get_ddp_info
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
from traceml.renderers.stdout_stderr_renderer import StdoutStderrRenderer
from traceml.renderers.layer_combined_timing_renderer import LayerCombinedTimerRenderer


class TrackerManager:
    _DISPLAY = {
        "cli": (CLIDisplayManager, "get_panel_renderable"),
        "notebook": (NotebookDisplayManager, "get_notebook_renderable"),
        "dashboard": (NiceGUIDisplayManager, "get_dashboard_renderable"),
    }

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
        config.enable_logging = enable_logging
        config.logs_dir = logs_dir

        setup_error_logger()
        self.logger = get_error_logger("TrackerManager")

        try:
            self.display_manager, self._render_attr = self._DISPLAY[mode]
        except KeyError:
            raise ValueError(
                f"Unsupported mode: {mode}. Choose from {list(self._DISPLAY)}"
            )

        if samplers is None or renderers is None:
            self.samplers, self.renderers = self._build_components(
                mode=mode,
                num_display_layers=num_display_layers,
            )
        else:
            self.samplers = samplers
            self.renderers = renderers

        self.interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _safe(self, label: str, fn):
        try:
            return fn()
        except Exception as e:
            self.logger.error(f"[TraceML] {label}: {e}")
            return None

    @staticmethod
    def _build_components(mode: str, num_display_layers: int):
        is_ddp, local_rank, _world = get_ddp_info()

        samplers: List[BaseSampler] = []
        renderers: List[BaseRenderer] = []

        # System / process (rank 0 only)
        if not (is_ddp and local_rank != 0):
            sys_s = SystemSampler()
            proc_s = ProcessSampler()
            samplers += [sys_s, proc_s]
            renderers += [
                SystemRenderer(database=sys_s.db),
                ProcessRenderer(database=proc_s.db),
            ]

        # Layer memory (3 samplers -> 1 combined renderer)
        layer_s = LayerMemorySampler()
        fwd_mem_s = LayerForwardMemorySampler()
        bwd_mem_s = LayerBackwardMemorySampler()
        samplers += [layer_s, fwd_mem_s, bwd_mem_s]
        renderers += [
            LayerCombinedMemoryRenderer(
                layer_db=layer_s.db,
                layer_forward_db=fwd_mem_s.db,
                layer_backward_db=bwd_mem_s.db,
                top_n_layers=num_display_layers,
            )
        ]

        step_mem_sampler = StepMemorySampler()
        samplers += [step_mem_sampler]

        # Step timer
        step_s = StepTimerSampler()
        samplers += [step_s]
        renderers += [StepTimerRenderer(database=step_s.db)]
        renderers += [ModelCombinedRenderer(time_db=step_s.db, memory_db=step_mem_sampler.db)]

        # Timing (2 samplers -> 1 combined renderer)
        fwd_time_s = LayerForwardTimeSampler()
        bwd_time_s = LayerBackwardTimeSampler()
        samplers += [fwd_time_s, bwd_time_s]
        renderers += [
            LayerCombinedTimerRenderer(
                forward_db=fwd_time_s.db,
                backward_db=bwd_time_s.db,
                top_n_layers=num_display_layers,
            )
        ]


        # CLI-only stdout/stderr renderer
        if mode == "cli":
            renderers += [StdoutStderrRenderer()]

        return samplers, renderers

    def _run_samplers(self):
        for s in self.samplers:
            self._safe(f"Sampler {s.__class__.__name__}.sample() failed", s.sample)
            if getattr(s, "db", None) is not None:
                self._safe(
                    f"Sampler {s.__class__.__name__}.db.writer.flush() failed",
                    s.db.writer.flush,
                )

    def _run_renderers(self):
        for r in self.renderers:

            def register():
                render_fn = getattr(r, self._render_attr)
                self.display_manager.register_layout_content(
                    r.layout_section_name, render_fn
                )

            self._safe(f"Renderer {r.__class__.__name__} register failed", register)

    def _run_once(self):
        self._run_samplers()
        self._run_renderers()
        self._safe("Display update failed", self.display_manager.update_display)

    def _run(self):
        self._safe("Display start failed", self.display_manager.start_display)
        while not self._stop_event.is_set():
            self._run_once()
            self._stop_event.wait(self.interval_sec)
        self._run_once()

    def start(self) -> None:
        self._safe("Failed to start TrackerManager thread", self._thread.start)

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=self.interval_sec * 5)
        if self._thread.is_alive():
            self.logger.error(
                "[TraceML] WARNING: Tracker thread did not terminate within timeout."
            )
        self._safe("Display release failed", self.display_manager.release_display)

    def log_summaries(self, path="traceml_system_summary.txt") -> None:
        for r in self.renderers:
            self._safe(
                f"Renderer {r.__class__.__name__}.log_summary failed",
                lambda rr=r: rr.log_summary(path),
            )
