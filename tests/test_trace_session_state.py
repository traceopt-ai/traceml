import pytest

from traceml.runtime.state import (
    TraceSessionState,
    get_trace_session_state,
    reset_trace_session_state,
)
from traceml.sdk.instrumentation import TraceState


def test_trace_session_state_tracks_step_with_explicit_methods():
    state = TraceSessionState(initial_step=2)

    assert state.step == 2
    assert state.advance_step() == 3
    assert state.advance_step(2) == 5
    assert state.reset() == 0
    assert state.set_step(9) == 9


@pytest.mark.parametrize("value", [-1, 1.2, "3"])
def test_trace_session_state_rejects_invalid_step_values(value):
    state = TraceSessionState()

    with pytest.raises((TypeError, ValueError)):
        state.set_step(value)


@pytest.mark.parametrize("delta", [-1, 1.2, "1"])
def test_trace_session_state_rejects_invalid_step_delta(delta):
    state = TraceSessionState()

    with pytest.raises((TypeError, ValueError)):
        state.advance_step(delta)


def test_trace_state_facade_uses_runtime_session_state():
    reset_trace_session_state()
    state = get_trace_session_state()

    TraceState.step += 1
    assert state.step == 1
    assert TraceState.step == 1

    TraceState.step = 7
    assert state.step == 7
    assert TraceState.session() is state
    assert TraceState.advance() == 8
    assert TraceState.reset() == 0


def test_decorators_import_path_shares_trace_state():
    import traceml.decorators as decorators

    TraceState.reset()
    decorators.TraceState.step += 2

    assert TraceState.step == 2
    assert decorators.TraceState.step == 2
    assert decorators.trace_step is not None
