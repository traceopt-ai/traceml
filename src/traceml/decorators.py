"""
Legacy public decorator compatibility surface.

Importing ``traceml.decorators`` historically enabled TraceML's legacy
 automatic instrumentation path as a side effect. The real implementation now
 lives in ``traceml.sdk.decorators_compat``; this module preserves the public
 import path for older code and keeps the behavior discoverable.
"""

from traceml.sdk.decorators_compat import (
    TraceState,
    trace_model_instance,
    trace_step,
    trace_time,
)

__all__ = [
    "TraceState",
    "trace_step",
    "trace_model_instance",
    "trace_time",
]
