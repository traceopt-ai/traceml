"""Process-local tracing-armed flag.

Patch wrappers consult ``is_tracing_armed()`` to decide whether to time an
intercepted call. This lives in its own leaf module (no imports beyond the
standard library) so the patch modules under ``instrumentation/patches/`` can
import it without pulling in ``traceml_ai.sdk``, whose package init imports
``sdk.instrumentation``, which imports the patch modules themselves.
``traceml_ai.sdk.initial`` re-exports this for the public/internal call sites
that already reference it there.
"""

from __future__ import annotations

_TRACING_ARMED: bool = False


def is_tracing_armed() -> bool:
    """
    Return True only after the requested automatic patches installed cleanly.

    Patch wrappers consult this gate so a failed partial installation degrades
    to native pass-through behavior instead of half-instrumenting training.
    """
    return _TRACING_ARMED


def _set_tracing_armed(value: bool) -> None:
    global _TRACING_ARMED
    _TRACING_ARMED = bool(value)


__all__ = ["is_tracing_armed"]
