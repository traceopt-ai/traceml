"""Reject the unsupported ``traceml`` import path."""

raise ImportError(
    "The 'traceml' import path is not supported. "
    "Use 'import traceml_ai as traceml' instead."
)
