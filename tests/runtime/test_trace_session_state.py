import pytest

from traceml_ai.runtime.state import (
    TraceMLRecordingState,
    TraceMLRecordingStatus,
    TraceSessionState,
    configure_trace_recording,
    get_trace_recording_state,
    get_trace_session_state,
    mark_trace_recording_drained,
    mark_trace_step_flushed,
    reset_trace_session_state,
    should_record_trace_events,
)
from traceml_ai.sdk.instrumentation import TraceState


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


def test_trace_recording_state_is_unlimited_by_default():
    state = TraceMLRecordingState()

    assert state.should_record() is True
    assert state.mark_step_flushed(1000) == TraceMLRecordingStatus.RECORDING
    assert state.should_record() is True


def test_trace_recording_state_transitions_after_max_step():
    state = TraceMLRecordingState(max_steps=2)

    assert state.mark_step_flushed(1) == TraceMLRecordingStatus.RECORDING
    assert state.should_record() is True
    assert state.mark_step_flushed(2) == TraceMLRecordingStatus.DRAINING
    assert state.should_record() is False
    assert state.mark_drained() == TraceMLRecordingStatus.COMPLETE
    assert state.should_record() is False


def test_global_trace_recording_helpers_configure_active_state():
    configure_trace_recording(max_steps=1)

    assert should_record_trace_events() is True
    assert mark_trace_step_flushed(1) == TraceMLRecordingStatus.DRAINING
    assert should_record_trace_events() is False
    assert mark_trace_recording_drained() == TraceMLRecordingStatus.COMPLETE
    assert (
        get_trace_recording_state().status == TraceMLRecordingStatus.COMPLETE
    )

    configure_trace_recording()


def test_timed_region_noops_when_recording_is_complete():
    import traceml_ai.utils.timing as timing

    configure_trace_recording(max_steps=1)
    mark_trace_step_flushed(1)
    mark_trace_recording_drained()

    original_size = len(timing._STEP_BUFFER)
    with timing.timed_region("_test_region", scope=timing.TimeScope.STEP):
        pass

    assert len(timing._STEP_BUFFER) == original_size

    configure_trace_recording()
