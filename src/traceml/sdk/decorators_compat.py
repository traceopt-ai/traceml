"""
Legacy decorator compatibility layer.

This module preserves the historical TraceML behavior where importing
`traceml.decorators` automatically enabled automatic patch-based
instrumentation.

New code should prefer the explicit path:

    import traceml
    traceml.init(...)
    with traceml.trace_step(...):
        ...

Backward compatibility
----------------------
Importing this module still triggers automatic patch installation once per
process, matching the legacy user experience.
"""

from traceml.sdk.initial import enable_legacy_decorator_auto_init
from traceml.sdk.instrumentation import (
    TraceSessionState,
    TraceState,
    get_trace_session_state,
    trace_model_instance,
    trace_step,
    trace_time,
)

enable_legacy_decorator_auto_init()

__all__ = [
    "TraceSessionState",
    "TraceState",
    "get_trace_session_state",
    "trace_step",
    "trace_model_instance",
    "trace_time",
]
