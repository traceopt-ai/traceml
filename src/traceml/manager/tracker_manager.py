import threading
from typing import List, Tuple
from traceml.loggers.error_log import get_error_logger, setup_error_logger
from traceml.config import config
from traceml.renderers.display.cli_display_manager import CLIDisplayManager
from traceml.renderers.display.notebook_display_manager import NotebookDisplayManager
from traceml.renderers.display.nicegui_display_manager import NiceGUIDisplayManager

from traceml.utils.distributed import get_ddp_info
from traceml.samplers.base_sampler import BaseSampler
from traceml.samplers.system_sampler import SystemSampler
from traceml.samplers.process_sampler import ProcessSampler
from traceml.samplers.layer_memory_sampler import LayerMemorySampler
from traceml.samplers.activation_memory_sampler import ActivationMemorySampler
from traceml.samplers.gradient_memory_sampler import GradientMemorySampler
from traceml.samplers.steptimer_sampler import StepTimerSampler
from traceml.samplers.activation_time_sampler import ActivationTimeSampler
from traceml.samplers.gradient_time_sampler import GradientTimeSampler

from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.system_renderer import SystemRenderer
from traceml.renderers.process_renderer import ProcessRenderer
from traceml.renderers.layer_combined_memory_renderer import (
    LayerCombinedMemoryRenderer,
)
from traceml.renderers.activation_gradient_memory_renderer import (
    ActivationGradientRenderer,
)
from traceml.renderers.steptimer_renderer import StepTimerRenderer
from traceml.renderers.stdout_stderr_renderer import StdoutStderrRenderer

from traceml.renderers.layer_combined_timing_renderer import LayerCombinedTimerRenderer


class TrackerManager:
    """
    Manages periodic sampling and logging of system metrics (CPU, memory, tensors, etc.)
    using a background thread. Each component defines a sampler and a list of associated loggers.

    This class ensures consistent sampling even if some components fail intermittently.
    """

    def __init__(
        self,
        components: List[Tuple[List[BaseSampler], List[BaseRenderer]]] = None,
        interval_sec: float = 1.0,
        mode: str = "cli",  # "cli" or "notebook"
        num_display_layers: int = 20,
        enable_logging: bool = False,
        logs_dir: str = "./logs",
    ):
        """
        Args:
            components (list of tuples): List of (sampler, list of loggers) pairs.
                                         Each sampler's output is sent to all loggers in its list.
            interval_sec (int): Time interval in seconds between samples.
        """
        config.enable_logging = enable_logging
        config.logs_dir = logs_dir
        setup_error_logger()
        self.logger = get_error_logger("TrackerManager")
        if components is None:
            self.components = self._components(mode, num_display_layers)
        else:
            self.components = components
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        if mode == "cli":
            self.display_manager = CLIDisplayManager
            self._render_attr = "get_panel_renderable"
        elif mode == "notebook":
            self.display_manager = NotebookDisplayManager
            self._render_attr = "get_notebook_renderable"
        elif mode == "dashboard":
            self.display_manager = NiceGUIDisplayManager
            self._render_attr = "get_dashboard_renderable"
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    @staticmethod
    def _components(
        mode: str, num_display_layers: int
    ) -> List[Tuple[List[BaseSampler], List[BaseRenderer]]]:

        is_ddp, local_rank, world_size = get_ddp_info()

        components = []
        components += TrackerManager.get_system_components(is_ddp, local_rank)
        components += TrackerManager.get_process_components(is_ddp, local_rank)
        components += TrackerManager.get_memory_components(num_display_layers)
        components += TrackerManager.get_step_timer_components()
        components += TrackerManager.get_timing_components(num_display_layers)

        if mode == "cli":
            components += TrackerManager.get_stdout_components()

        return components

    @staticmethod
    def get_system_components(is_ddp: bool, local_rank: int):
        if is_ddp and local_rank != 0:
            return []  # disable on non-zero ranks

        system_sampler = SystemSampler()
        system_renderer = SystemRenderer(database=system_sampler.db)
        return [([system_sampler], [system_renderer])]


    @staticmethod
    def get_process_components(is_ddp: bool, local_rank: int):
        if is_ddp and local_rank != 0:
            return []
        process_sampler = ProcessSampler()
        process_renderer = ProcessRenderer(database=process_sampler.db)

        return [([process_sampler], [process_renderer])]

    @staticmethod
    def get_memory_components(num_display_layers: int):
        layer_memory_sampler = LayerMemorySampler()
        activation_memory_sampler = ActivationMemorySampler()
        gradient_memory_sampler = GradientMemorySampler()

        layer_combined_renderer = LayerCombinedMemoryRenderer(
            layer_db=layer_memory_sampler.db,
            activation_db=activation_memory_sampler.db,
            gradient_db=gradient_memory_sampler.db,
            top_n_layers=num_display_layers,
        )

        activation_gradient_renderer = ActivationGradientRenderer(
            layer_db=layer_memory_sampler.db,
            activation_db=activation_memory_sampler.db,
            gradient_db=gradient_memory_sampler.db,
        )
        return [
            (
                [
                    layer_memory_sampler,
                    activation_memory_sampler,
                    gradient_memory_sampler,
                ],
                [layer_combined_renderer, activation_gradient_renderer],
            )
        ]

    @staticmethod
    def get_step_timer_components():
        step_timer_sampler = StepTimerSampler()
        step_timer_renderer = StepTimerRenderer(database=step_timer_sampler.db)
        return [([step_timer_sampler], [step_timer_renderer])]

    @staticmethod
    def get_stdout_components():
        stdout_renderer = StdoutStderrRenderer()
        return [([], [stdout_renderer])]

    @staticmethod
    def get_timing_components(num_display_layers: int):
        activation_timing_sampler = ActivationTimeSampler()
        gradient_timing_sampler = GradientTimeSampler()
        combined_timing_rendered = LayerCombinedTimerRenderer(
            activation_db=activation_timing_sampler.db,
            gradient_db=gradient_timing_sampler.db,
            top_n_layers=num_display_layers
        )
        return [([activation_timing_sampler, gradient_timing_sampler], [combined_timing_rendered])]


    def _run_once(self):
        """Single sampling + writing to file + logging + display update pass."""
        for samplers, loggers in self.components:
            for sampler in samplers:
                try:
                    sampler.sample()
                    sampler.db.writer.flush()
                except Exception as e:
                    self.logger.error(
                        f"[TraceML] Error in sampler '{sampler.__class__.__name__}'.sample(): {e}"
                    )

            # Log to all renderers
            for logger in loggers:
                try:
                    render_fn = getattr(logger, self._render_attr)
                    self.display_manager.register_layout_content(
                        logger.layout_section_name, render_fn
                    )
                except Exception as e:
                    self.logger.error(
                        f"[TraceML] Error in logger '{logger.__class__.__name__}'.log(): {e}"
                    )

        self.display_manager.update_display()

    def _run(self):
        """
        Background thread loop that continuously samples and logs live snapshots.
        """
        self.display_manager.start_display()

        while not self._stop_event.is_set():
            self._run_once()
            self._stop_event.wait(self.interval_sec)
        self._run_once()  # Final pass on stop

    def start(self) -> None:
        """
        Starts the background tracking thread.
        """
        try:
            self._thread.start()
        except Exception as e:
            self.logger.error(f"[TraceML] Failed to start TrackerManager: {e}")

    def stop(self) -> None:
        """
        Signals the background thread to stop and waits for it to terminate.
        """
        try:
            self._stop_event.set()
            self._thread.join(timeout=self.interval_sec * 5)

            if self._thread.is_alive():
                self.logger.error(
                    "[TraceML] WARNING: Tracker thread did not terminate within timeout."
                )

            # Logger shutdown is now handled more broadly by the main execution context
            # calling CLIDisplayManager.stop_display() and individual log_summaries.
            for _, loggers in self.components:
                for logger in loggers:
                    try:
                        if hasattr(logger, "shutdown"):
                            logger.shutdown()
                        self.display_manager.release_display()
                    except Exception as e:
                        self.logger.error(
                            f"[TraceML] Logger '{logger.__class__.__name__}' shutdown error: {e}"
                        )

        except Exception as e:
            self.logger.error(f"[TraceML] Failed to stop TrackerManager: {e}")

    def log_summaries(self) -> None:
        """
        Logs final summaries from each sampler (or group of samplers) after tracking stops.
        """

        for _, loggers in self.components:
            # pass merged summaries to all loggers for this group
            for logger in loggers:
                try:
                    logger.log_summary()
                except Exception as e:
                    self.logger.error(
                        f"[TraceML] Error in logger '{logger.__class__.__name__}'.log_summary(): {e}"
                    )
