"""
Runtime tracing state for a TraceML process.

This module owns mutable, process-local state that used to live directly on
``traceml.sdk.instrumentation.TraceState``. Keeping the storage here gives the
runtime and integrations a concrete object to depend on while the SDK keeps a
small compatibility facade for existing imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Optional


class TraceMLRecordingStatus(Enum):
    """Lifecycle state for TraceML telemetry recording."""

    RECORDING = "recording"
    DRAINING = "draining"
    COMPLETE = "complete"


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


@dataclass
class TraceMLRecordingState:
    """
    Process-local state controlling whether TraceML should record telemetry.

    Patch installation is separate from recording state: patches may remain
    installed after the step budget is reached, but they should no-op.
    """

    max_steps: Optional[int] = None
    _status: TraceMLRecordingStatus = field(
        default=TraceMLRecordingStatus.RECORDING, init=False, repr=False
    )
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_steps is not None:
            self.max_steps = self._validate_max_steps(self.max_steps)

    @property
    def status(self) -> TraceMLRecordingStatus:
        """Return the current recording lifecycle status."""
        with self._lock:
            return self._status

    def should_record(self) -> bool:
        """Return True while instrumentation should emit telemetry."""
        return self.status == TraceMLRecordingStatus.RECORDING

    def mark_step_flushed(self, step: int) -> TraceMLRecordingStatus:
        """
        Update recording state after a TraceML step has been flushed.

        The configured max step is inclusive: max_steps=100 records and flushes
        step 100, then moves the runtime into final-drain mode.
        """
        TraceSessionState._validate_step(step)
        with self._lock:
            if (
                self.max_steps is not None
                and step >= self.max_steps
                and self._status == TraceMLRecordingStatus.RECORDING
            ):
                self._status = TraceMLRecordingStatus.DRAINING
            return self._status

    def mark_drained(self) -> TraceMLRecordingStatus:
        """Mark recording complete after the runtime performs its final drain."""
        with self._lock:
            if self._status == TraceMLRecordingStatus.DRAINING:
                self._status = TraceMLRecordingStatus.COMPLETE
            return self._status

    @staticmethod
    def _validate_max_steps(value: int) -> int:
        if not isinstance(value, int):
            raise TypeError("TraceML recording max_steps must be an integer.")
        if value <= 0:
            raise ValueError("TraceML recording max_steps must be positive.")
        return value


_DEFAULT_TRACE_SESSION_STATE = TraceSessionState()
_DEFAULT_TRACE_RECORDING_STATE = TraceMLRecordingState()


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


def configure_trace_recording(
    max_steps: Optional[int] = None,
) -> TraceMLRecordingState:
    """Configure process-local TraceML recording state."""
    global _DEFAULT_TRACE_RECORDING_STATE
    _DEFAULT_TRACE_RECORDING_STATE = TraceMLRecordingState(max_steps=max_steps)
    return _DEFAULT_TRACE_RECORDING_STATE


def get_trace_recording_state() -> TraceMLRecordingState:
    """Return the active process-local TraceML recording state."""
    return _DEFAULT_TRACE_RECORDING_STATE


def should_record_trace_events() -> bool:
    """Return True while TraceML instrumentation should emit telemetry."""
    return _DEFAULT_TRACE_RECORDING_STATE.should_record()


def mark_trace_step_flushed(step: int) -> TraceMLRecordingStatus:
    """Update recording state after a TraceML step has been flushed."""
    return _DEFAULT_TRACE_RECORDING_STATE.mark_step_flushed(step)


def mark_trace_recording_drained() -> TraceMLRecordingStatus:
    """Mark recording complete after runtime queue draining finishes."""
    return _DEFAULT_TRACE_RECORDING_STATE.mark_drained()


__all__ = [
    "TraceMLRecordingState",
    "TraceMLRecordingStatus",
    "TraceSessionState",
    "configure_trace_recording",
    "get_trace_recording_state",
    "get_trace_session_state",
    "mark_trace_recording_drained",
    "mark_trace_step_flushed",
    "reset_trace_session_state",
    "should_record_trace_events",
]
