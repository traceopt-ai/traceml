import threading
from typing import List, Tuple, Any, Dict
from traceml.loggers.error_log import get_error_logger, setup_error_logger
from traceml.renderers.display.cli_display_manager import CLIDisplayManager
from traceml.renderers.display.notebook_display_manager import (
    NotebookDisplayManager,
)

from traceml.samplers.base_sampler import BaseSampler
from traceml.samplers.system_sampler import SystemSampler
from traceml.samplers.process_sampler import ProcessSampler
from traceml.samplers.layer_memory_sampler import LayerMemorySampler
from traceml.samplers.activation_memory_sampler import ActivationMemorySampler
from traceml.samplers.gradient_memory_sampler import GradientMemorySampler
from traceml.samplers.steptimer_sampler import StepTimerSampler

from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.system_process_renderer import SystemProcessRenderer
from traceml.renderers.layer_combined_stdout_renderer import (
    LayerCombinedRenderer,
)
from traceml.renderers.activation_gradient_memory_renderer import (
    ActivationGradientRenderer,
)
from traceml.renderers.steptimer_renderer import StepTimerRenderer


class TrackerManager:
    """
    Manages periodic sampling and logging of system metrics (CPU, memory, tensors, etc.)
    using a background thread. Each component defines a sampler and a list of associated loggers.

    This class ensures consistent sampling even if some components fail intermittently.
    """

    @staticmethod
    def _components() -> List[Tuple[List[BaseSampler], List[BaseRenderer]]]:
        system_sampler = SystemSampler()
        process_sampler = ProcessSampler()
        layer_memory_sampler = LayerMemorySampler()
        activation_memory_sampler = ActivationMemorySampler()
        gradient_memory_sampler = GradientMemorySampler()
        steptimer_sampler = StepTimerSampler()

        system_process_renderer = SystemProcessRenderer()
        layer_combined_renderer = LayerCombinedRenderer()
        activation_gradient_renderer = ActivationGradientRenderer()
        steptimer_renderer = StepTimerRenderer()

        # Collect all trackers
        sampler_logger_pairs = [
            ([system_sampler, process_sampler], [system_process_renderer]),
            (
                [
                    layer_memory_sampler,
                    activation_memory_sampler,
                    gradient_memory_sampler,
                ],
                [layer_combined_renderer, activation_gradient_renderer],
            ),
            ([steptimer_sampler], [steptimer_renderer]),
        ]
        return sampler_logger_pairs

    def __init__(
        self,
        components: List[Tuple[List[BaseSampler], List[BaseRenderer]]] = None,
        interval_sec: float = 1.0,
        mode: str = "cli",  # "cli" or "notebook"
    ):
        """
        Args:
            components (list of tuples): List of (sampler, list of loggers) pairs.
                                         Each sampler's output is sent to all loggers in its list.
            interval_sec (int): Time interval in seconds between samples.
        """
        setup_error_logger()
        self.logger = get_error_logger("TrackerManager")
        if components is None:
            self.components = self._components()
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
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def _run(self):
        """
        Background thread loop that continuously samples and logs live snapshots.
        """
        self.display_manager.start_display()

        while not self._stop_event.is_set():
            for samplers, loggers in self.components:
                if not isinstance(samplers, (list, tuple)):
                    samplers = [samplers]

                snapshots = {}
                for sampler in samplers:
                    try:
                        snapshots[sampler.__class__.__name__] = sampler.sample()
                    except Exception as e:
                        self.logger.error(
                            f"[TraceML] Error in sampler '{sampler.__class__.__name__}'.sample(): {e}"
                        )
                        snapshots[sampler.__class__.__name__] = {
                            "error": str(e),
                            "sampler_name": sampler.__class__.__name__,
                        }

                # 2. Log snapshot to all associated loggers
                for logger in loggers:
                    render_fn = getattr(logger, self._render_attr)
                    self.display_manager.register_layout_content(
                        logger.layout_section_name, render_fn
                    )

                    try:
                        logger.log(snapshots)
                    except Exception as e:
                        self.logger.error(
                            f"[TraceML] Error in logger '{logger.__class__.__name__}'.log() for sampler '{sampler.__class__.__name__}': {e}"
                        )
                self.display_manager.update_display()

            # 3. Wait for the next interval
            self._stop_event.wait(self.interval_sec)

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
            self._thread.join(timeout=self.interval_sec * 2)

            if self._thread.is_alive():
                self.logger.error(
                    "[TraceML] WARNING: Tracker thread did not terminate within timeout."
                )

            # Logger shutdown is now handled more broadly by the main execution context
            # calling CLIDisplayManager.stop_display() and individual log_summaries.
            for _, loggers in self.components:
                for logger in loggers:
                    try:
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

        for samplers, loggers in self.components:
            if not isinstance(samplers, (list, tuple)):
                samplers = [samplers]

            # collect summaries from all samplers in this group
            summaries: Dict[str, Any] = {}
            for sampler in samplers:
                try:
                    summaries[sampler.__class__.__name__] = sampler.get_summary()
                except Exception as e:
                    self.logger.error(
                        f"[TraceML] Error getting summary from sampler '{sampler.__class__.__name__}': {e}"
                    )
                    summaries[sampler.__class__.__name__] = {
                        "error": str(e),
                        "sampler_name": sampler.__class__.__name__,
                    }

            # pass merged summaries to all loggers for this group
            for logger in loggers:
                try:
                    logger.log_summary(summaries)
                except Exception as e:
                    self.logger.error(
                        f"[TraceML] Error in logger '{logger.__class__.__name__}'.log_summary(): {e}"
                    )
