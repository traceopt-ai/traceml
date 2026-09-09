import os

from traceml_ai.runtime.state import should_record_trace_events

from .step_memory import flush_step_memory_buffer
from .timing import flush_step_time_buffer


def _traceml_disabled() -> bool:
    return os.environ.get("TRACEML_DISABLED") == "1"


def flush_step_events(step: int) -> None:
    """Flush pending memory and timing at the caller's existing step boundary."""
    if _traceml_disabled() or not should_record_trace_events():
        return

    flush_step_memory_buffer(step)
    flush_step_time_buffer(step)
