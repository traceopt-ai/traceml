"""
Dashboard-only renderer for unified model diagnostics.

This renderer composes:
- step-time diagnosis
- step-memory diagnosis

It returns one structured payload for the NiceGUI "Model Diagnostics" section.
On transient failures it returns last-good payload to avoid UI blanking.
"""

from typing import Any, Dict, Optional

from rich.panel import Panel

from traceml.aggregator.display_drivers.layout import MODEL_DIAGNOSTICS_LAYOUT
from traceml.diagnostics.model_diagnostics import (
    ModelDiagnosticsPayload,
    build_model_diagnostics_payload,
)
from traceml.loggers.error_log import get_error_logger
from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.step_memory.computer import StepMemoryMetricsComputer
from traceml.renderers.step_time.compute import StepCombinedComputer


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
        self._step_time = StepCombinedComputer(db_path=db_path)
        self._step_memory = StepMemoryMetricsComputer(db_path=db_path)

        self._cached: Optional[Dict[str, Any]] = None

    def get_panel_renderable(self) -> Panel:
        """
        CLI is not the primary target for this renderer.
        """
        return Panel(
            "Model diagnostics are available in dashboard mode.",
            title="Model Diagnostics",
        )

    def get_dashboard_renderable(self) -> Dict[str, Any]:
        """
        Return unified diagnostics payload for dashboard use.

        Never raises; returns last-good payload on failures.
        """
        try:
            step_time = self._step_time.compute_cli()
            step_memory = self._step_memory.compute_cli()

            payload: ModelDiagnosticsPayload = build_model_diagnostics_payload(
                step_time_metrics=step_time.metrics,
                step_memory_metrics=step_memory.metrics,
                gpu_total_bytes=None,
            )
            out = payload.to_dict()

            if out.get("items"):
                self._cached = out
                return out

            if self._cached is not None:
                return self._cached
            return out

        except Exception:
            self._logger.exception("ModelDiagnosticsRenderer compute failed")
            if self._cached is not None:
                return self._cached

            return ModelDiagnosticsPayload(
                generated_at_s=0.0,
                overall_severity="info",
                status_message="NO DATA",
                items=[],
            ).to_dict()
