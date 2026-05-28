"""Canonical TraceML instrumentation internals.

This package owns the hook installers and monkey patches used by the SDK.
Public user APIs stay in :mod:`traceml` and :mod:`traceml.sdk`; code outside
TraceML should not need to import from this package directly.
"""

__all__: list[str] = []
