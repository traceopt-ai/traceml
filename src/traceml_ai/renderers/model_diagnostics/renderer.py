"""
Dashboard-only renderer for unified model diagnostics.

This renderer composes a precomputed Step Time diagnosis with Step Memory.

It returns one structured payload for the NiceGUI "Model Diagnostics" section.
Live Step Time freshness remains owned by ``LiveStepTimeSession``.
"""

from typing import Any, Dict

from rich.panel import Panel

from traceml_ai.aggregator.display_drivers.layout import (
    MODEL_DIAGNOSTICS_LAYOUT,
)
from traceml_ai.diagnostics.model_diagnostics import (
    ModelDiagnosticsPayload,
    build_model_diagnostics_payload,
)
from traceml_ai.loggers.error_log import get_error_logger
from traceml_ai.renderers.base_renderer import BaseRenderer
from traceml_ai.renderers.step_memory.computer import StepMemoryMetricsComputer
from traceml_ai.step_time.pipeline import LiveStepTimeResult


class ModelDiagnosticsRenderer(BaseRenderer):
    """
    Unified diagnostics renderer for dashboard composition.
    """

    NAME = "ModelDiagnostics"

    def __init__(self, db_path: str) -> None:
        super().__init__(
            name=self.NAME,
            layout_section_name=MODEL_DIAGNOSTICS_LAYOUT,
        )
        self._logger = get_error_logger("ModelDiagnosticsRenderer")
        self._step_memory = StepMemoryMetricsComputer(db_path=db_path)

    def get_panel_renderable(self) -> Panel:
        """
        CLI is not the primary target for this renderer.
        """
        return Panel(
            "Model diagnostics are available in dashboard mode.",
            title="Model Diagnostics",
        )

    def get_dashboard_renderable(
        self,
        step_time: LiveStepTimeResult,
    ) -> Dict[str, Any]:
        """Combine one existing Step Time diagnosis with Step Memory."""
        try:
            step_memory = self._step_memory.compute_dashboard()
        except Exception:
            # Step Memory is independent telemetry. Its failure must not hide
            # the Step Time diagnosis already produced for this dashboard
            # tick.
            self._logger.exception("Step Memory diagnostics compute failed")
            step_memory_metrics = ()
            step_memory_status = "Step Memory data unavailable"
            step_memory_gpu_total_bytes = None
        else:
            step_memory_metrics = step_memory.metrics
            step_memory_status = step_memory.status_message
            step_memory_gpu_total_bytes = step_memory.gpu_total_bytes

        try:
            payload: ModelDiagnosticsPayload = build_model_diagnostics_payload(
                step_time_window=step_time.analysis.window,
                step_time_diagnosis=step_time.analysis.diagnosis.primary,
                step_memory_metrics=step_memory_metrics,
                step_memory_status_message=step_memory_status,
                gpu_total_bytes=step_memory_gpu_total_bytes,
            )
            return payload.to_dict()

        except Exception:
            self._logger.exception("ModelDiagnosticsRenderer compute failed")
            return ModelDiagnosticsPayload(
                generated_at_s=0.0,
                overall_severity="info",
                status_message="NO DATA",
                items=[],
            ).to_dict()
