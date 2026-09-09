"""The DataLoader fetch patch can be suppressed for non-training loaders."""

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from traceml_ai.instrumentation.patches.dataloader_patch import (  # noqa: E402
    patch_dataloader,
    suppress_dataloader_timing,
)
from traceml_ai.instrumentation.step_events import (  # noqa: E402
    abort_step_capture,
    begin_step_capture,
)
from traceml_ai.runtime.arming import (  # noqa: E402
    _set_tracing_armed,
    is_tracing_armed,
)
from traceml_ai.runtime.state import configure_trace_recording  # noqa: E402

_FETCH = "_traceml_internal:dataloader_next"


@pytest.fixture(autouse=True)
def _armed_and_clean(monkeypatch):
    monkeypatch.delenv("TRACEML_DISABLED", raising=False)
    previous = is_tracing_armed()
    patch_dataloader()
    _set_tracing_armed(True)
    configure_trace_recording(max_steps=None)
    abort_step_capture(begin_step_capture())
    yield
    abort_step_capture(begin_step_capture())
    _set_tracing_armed(previous)


def _loader(n: int = 6) -> DataLoader:
    return DataLoader(TensorDataset(torch.arange(n)), batch_size=1)


def _fetch_events() -> int:
    events = begin_step_capture().timing_events
    return sum(1 for evt in events if evt.name == _FETCH)


def test_fetches_are_timed_when_not_suppressed():
    it = iter(_loader(3))
    for _ in range(3):
        next(it)

    assert _fetch_events() == 3


def test_suppressed_fetches_record_nothing():
    it = iter(_loader(6))
    next(it)
    assert _fetch_events() == 1

    with suppress_dataloader_timing():
        next(it)
        next(it)
    assert _fetch_events() == 1, "fetches inside suppression must not record"

    next(it)
    assert _fetch_events() == 2, "timing resumes after the context exits"


def test_suppression_applies_per_next_not_per_iterator():
    # The iterator is created while suppressed; the gate must still be
    # re-evaluated on every fetch, so fetches after exit are timed.
    with suppress_dataloader_timing():
        it = iter(_loader(4))
        next(it)
    next(it)

    assert _fetch_events() == 1


def test_nested_suppression_restores_outer_state():
    it = iter(_loader(6))
    with suppress_dataloader_timing():
        with suppress_dataloader_timing():
            next(it)
        next(it)  # still inside the outer context
    next(it)

    assert _fetch_events() == 1


def test_suppression_survives_an_exception_inside():
    it = iter(_loader(4))
    with pytest.raises(RuntimeError):
        with suppress_dataloader_timing():
            next(it)
            raise RuntimeError("user error")
    next(it)

    assert _fetch_events() == 1


def test_suppression_is_a_no_op_when_disarmed():
    _set_tracing_armed(False)
    with suppress_dataloader_timing():
        for _ in _loader(2):
            pass
    for _ in _loader(2):
        pass

    assert _fetch_events() == 0
