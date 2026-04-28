"""
Shared TraceML contracts.

The core package is dependency-light by design. It should define small typing
contracts that other packages can adopt without importing torch, Rich, NiceGUI,
SQLite helpers, or TraceML runtime modules.
"""

from .lifecycle import Startable, Stoppable, Tickable
from .registry import Registry
from .rendering import Formatter, RenderContext, Renderer
from .summaries import ReportGenerator, SummaryResult, SummarySection
from .telemetry import MetricComputer, ProjectionWriter, TelemetryPayload

__all__ = [
    "Startable",
    "Stoppable",
    "Tickable",
    "Formatter",
    "RenderContext",
    "Renderer",
    "Registry",
    "ReportGenerator",
    "SummaryResult",
    "SummarySection",
    "MetricComputer",
    "ProjectionWriter",
    "TelemetryPayload",
]
