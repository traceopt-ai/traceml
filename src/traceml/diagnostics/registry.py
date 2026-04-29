"""
Diagnostic domain registries.

The model diagnostics card combines multiple diagnostic domains into one
dashboard payload. This module provides the extension point for that
composition so adding another domain does not require rewriting the model
diagnostics orchestration loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from traceml.core.registry import Registry
from traceml.renderers.step_memory.schema import StepMemoryCombinedMetric
from traceml.renderers.step_time.schema import StepCombinedTimeMetric


@dataclass(frozen=True)
class ModelDiagnosticContext:
    """
    Inputs available to model-diagnostic domain builders.

    The context intentionally carries renderer-facing payloads rather than raw
    SQL rows. Domain builders should be pure analysis/adaptation code and must
    not perform database I/O.
    """

    step_time_metrics: Sequence[StepCombinedTimeMetric]
    step_memory_metrics: Sequence[StepMemoryCombinedMetric]
    step_memory_status_message: Optional[str] = None
    gpu_total_bytes: Optional[float] = None


ModelDiagnosticBuilder = Callable[[ModelDiagnosticContext], Any]


@dataclass(frozen=True)
class DiagnosticDomainSpec:
    """
    Registered model-diagnostic domain.

    ``builder`` returns the domain-specific payload object consumed by the
    model diagnostics composer. The concrete return type is intentionally kept
    outside this module to avoid a circular import with ``model_diagnostics``.
    """

    name: str
    title: str
    builder: ModelDiagnosticBuilder

    def build(self, context: ModelDiagnosticContext) -> Any:
        """Build this domain's model-diagnostic item."""
        return self.builder(context)


class DiagnosticDomainRegistry(Registry[DiagnosticDomainSpec]):
    """Typed registry for model-diagnostic domain specs."""


__all__ = [
    "DiagnosticDomainRegistry",
    "DiagnosticDomainSpec",
    "ModelDiagnosticBuilder",
    "ModelDiagnosticContext",
]
