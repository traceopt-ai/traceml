"""
Legacy instrumentation import path.

The canonical implementation lives in ``traceml.sdk.instrumentation``. This
module intentionally contains no mutable state so the SDK, integrations, and
legacy imports all share the same ``TraceSessionState``.
"""

from traceml.sdk.instrumentation import (
    TraceSessionState,
    TraceState,
    get_trace_session_state,
    trace_model_instance,
    trace_step,
    trace_time,
)

__all__ = [
    "TraceSessionState",
    "TraceState",
    "get_trace_session_state",
    "trace_step",
    "trace_model_instance",
    "trace_time",
]
