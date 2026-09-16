"""Canonical TraceML instrumentation internals.

This package owns the SDK's hook installers and monkey patches, along with
the pending step capture, event types, and queues used to transfer measurements
to samplers.
Public user APIs stay in :mod:`traceml` and :mod:`traceml.sdk`; code outside
TraceML should not need to import from this package directly.
"""

__all__: list[str] = []
