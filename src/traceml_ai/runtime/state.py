"""
Runtime tracing state for a TraceML process.

This module owns mutable, process-local state that used to live directly on
``traceml.sdk.instrumentation.TraceState``. Keeping the storage here gives the
runtime and integrations a concrete object to depend on while the SDK keeps a
small compatibility facade for existing imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock


@dataclass
class TraceSessionState:
    """
    Process-local state shared by TraceML instrumentation paths.

    The first state carried here is the semantic training-step counter. It is
    intentionally independent from framework-specific counters such as
    Lightning's global step because TraceML also treats gradient-accumulation
    micro-batches as distinct traceable steps.
    """

    initial_step: int = 0
    _step: int = field(init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self._step = self._validate_step(self.initial_step)

    @property
    def step(self) -> int:
        """Return the current semantic TraceML step."""
        with self._lock:
            return self._step

    def set_step(self, value: int) -> int:
        """
        Set the current semantic step and return the stored value.

        Parameters
        ----------
        value:
            Non-negative integer step value.
        """
        next_step = self._validate_step(value)
        with self._lock:
            self._step = next_step
            return self._step

    def advance_step(self, delta: int = 1) -> int:
        """
        Advance the semantic step counter and return the new value.

        ``delta`` exists for tests and future framework adapters, but normal
        instrumentation should advance by one step at a time.
        """
        if not isinstance(delta, int):
            raise TypeError("TraceML step delta must be an integer.")
        if delta < 0:
            raise ValueError("TraceML step delta must be non-negative.")

        with self._lock:
            self._step = self._validate_step(self._step + delta)
            return self._step

    def reset(self, step: int = 0) -> int:
        """Reset the semantic step counter and return the stored value."""
        return self.set_step(step)

    @staticmethod
    def _validate_step(value: int) -> int:
        if not isinstance(value, int):
            raise TypeError("TraceML step must be an integer.")
        if value < 0:
            raise ValueError("TraceML step must be non-negative.")
        return value


_DEFAULT_TRACE_SESSION_STATE = TraceSessionState()


def get_trace_session_state() -> TraceSessionState:
    """
    Return the active process-local TraceML session state.

    New instrumentation code should use this function instead of mutating
    ``TraceState.step`` directly.
    """
    return _DEFAULT_TRACE_SESSION_STATE


def reset_trace_session_state(step: int = 0) -> int:
    """Reset the active process-local TraceML session state."""
    return _DEFAULT_TRACE_SESSION_STATE.reset(step)


__all__ = [
    "TraceSessionState",
    "get_trace_session_state",
    "reset_trace_session_state",
]
