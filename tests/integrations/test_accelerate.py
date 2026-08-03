import pytest
import torch
import torch.nn as nn

import traceml_ai as traceml
from traceml_ai.runtime.arming import _set_tracing_armed, is_tracing_armed

try:
    import accelerate  # noqa: F401

    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

INPUT_DIM = 16
HIDDEN_DIM = 32
NUM_CLASSES = 4
BATCH_SIZE = 8


class _TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


class _FakeAccelerator:
    """A minimal stand-in for ``accelerate.Accelerator``.

    It reproduces the exact surface the TraceML recipe touches:

    - ``.prepare(model, optimizer, dataloader)`` returns them unchanged,
      matching real Accelerate's non-distributed behavior,
    - ``.unwrap_model(model)`` returns the model unchanged, matching real
      Accelerate when there is no distributed wrapper to strip,
    - ``.backward(loss)`` calls ``loss.backward()`` (real Accelerate reaches
      the same ``torch.Tensor.backward`` when not using mixed-precision
      scaling), so TraceML's backward auto-timer fires.
    """

    def prepare(self, *args):
        return args

    def unwrap_model(self, model):
        return model

    def backward(self, loss):
        loss.backward()


def _drain_step_time_queue() -> list:
    """Drain all StepTimeBatch entries from the shared queue."""
    from traceml_ai.utils.timing import get_step_time_queue

    queue = get_step_time_queue()
    batches = []
    while not queue.empty():
        batches.append(queue.get_nowait())
    return batches


def _reset_traceml_state() -> None:
    """Reset TraceML's step counter, recording state, and step-time queue."""
    from traceml_ai.runtime.state import (
        configure_trace_recording,
        reset_trace_session_state,
    )
    from traceml_ai.utils.timing import _STEP_BUFFER

    reset_trace_session_state()
    configure_trace_recording(max_steps=None)
    _drain_step_time_queue()
    _STEP_BUFFER.clear()


def _install_auto_instrumentation() -> None:
    from traceml_ai.instrumentation.hooks.optimizer_hooks import (
        ensure_optimizer_timing_installed,
    )
    from traceml_ai.instrumentation.patches.backward_auto_timer_patch import (
        patch_backward,
    )
    from traceml_ai.instrumentation.patches.dataloader_patch import (
        patch_dataloader,
    )
    from traceml_ai.instrumentation.patches.forward_auto_timer_patch import (
        patch_forward,
    )
    from traceml_ai.instrumentation.patches.h2d_auto_timer_patch import (
        patch_h2d,
    )

    patch_forward()
    patch_backward()
    patch_h2d()
    patch_dataloader()
    ensure_optimizer_timing_installed()


@pytest.fixture(autouse=True)
def _armed_tracing():
    previous = is_tracing_armed()
    _set_tracing_armed(True)
    yield
    _set_tracing_armed(previous)


def test_accelerate_recipe_brackets_step():
    from traceml_ai.runtime.state import get_trace_session_state

    _reset_traceml_state()
    _install_auto_instrumentation()

    model = _TinyMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    accelerator = _FakeAccelerator()
    num_steps = 3

    model, optimizer = accelerator.prepare(model, optimizer)
    traced_model = accelerator.unwrap_model(model)

    step_before = get_trace_session_state().step

    for _ in range(num_steps):
        # This is exactly the recipe shown in the docs and example.
        with traceml.trace_step(traced_model):
            x = torch.randn(BATCH_SIZE, INPUT_DIM)
            y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)

            accelerator.backward(loss)
            optimizer.step()

    step_after = get_trace_session_state().step
    assert step_after - step_before == num_steps, (
        "trace_step must advance the TraceML step counter once per "
        f"Accelerate step; advanced by {step_after - step_before}."
    )

    batches = _drain_step_time_queue()
    assert len(batches) == num_steps, (
        f"Expected one StepTimeBatch per step ({num_steps}), "
        f"got {len(batches)}."
    )

    def _every_batch_has(event_name: str) -> bool:
        return all(
            any(evt.name == event_name for evt in batch.events)
            for batch in batches
        )

    assert _every_batch_has(
        "_traceml_internal:forward_time"
    ), "forward timing should be captured on the unwrapped model."
    assert _every_batch_has("_traceml_internal:backward_time"), (
        "backward timing should be captured because accelerator.backward(loss) "
        "reaches torch.Tensor.backward()."
    )
    assert _every_batch_has("_traceml_internal:optimizer_step"), (
        "optimizer timing should be captured because optimizer.step() is a "
        "real torch optimizer step."
    )
    assert _every_batch_has(
        "_traceml_internal:step_time"
    ), "step timing should be captured once per trace_step block."


def test_accelerate_recipe_emits_dataloader_next_over_real_loader():
    """The recipe must emit dataloader_next when iterating a real DataLoader.

    dataloader_next is the fragile stream the docs promise (it records
    DataLoader fetch time). It rides a class-level patch of
    ``DataLoader.__iter__``, so it only lands when a real ``torch``
    DataLoader is iterated. The recipe iterates the loader OUTSIDE
    ``trace_step``; the fetch is buffered by the process-wide recording
    gate and flushed into that step's StepTimeBatch, so every batch must
    carry the event. Asserted as rows landed, not as "a patch ran".
    """
    from torch.utils.data import DataLoader, TensorDataset

    _reset_traceml_state()
    _install_auto_instrumentation()

    num_steps = 3
    dataset = TensorDataset(
        torch.randn(num_steps * BATCH_SIZE, INPUT_DIM),
        torch.randint(0, NUM_CLASSES, (num_steps * BATCH_SIZE,)),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    model = _TinyMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    accelerator = _FakeAccelerator()

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    traced_model = accelerator.unwrap_model(model)

    for batch_x, batch_y in loader:
        # Fetch happens OUTSIDE trace_step, exactly like the documented recipe.
        with traceml.trace_step(traced_model):
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            accelerator.backward(loss)
            optimizer.step()

    batches = _drain_step_time_queue()
    assert len(batches) == num_steps, (
        f"Expected one StepTimeBatch per step ({num_steps}), "
        f"got {len(batches)}."
    )

    dark = [
        i
        for i, batch in enumerate(batches)
        if not any(
            evt.name == "_traceml_internal:dataloader_next"
            for evt in batch.events
        )
    ]
    assert not dark, (
        "dataloader_next stream is dark for the Accelerate recipe over a "
        f"real DataLoader: StepTimeBatch(es) {dark} of {num_steps} carry no "
        "event."
    )


@pytest.mark.skipif(not HAS_ACCELERATE, reason="accelerate not installed")
def test_accelerate_unwrap_model_is_noop_single_process():
    """Verifies the actual design decision the docs are built around.

    On a plain, non-distributed process there is nothing for Accelerate to
    wrap, so accelerator.unwrap_model(model) should return the exact same
    object accelerator.prepare() produced. The fake stand-in above assumes
    this; this test checks it against the real library instead of just
    assuming it.
    """
    from accelerate import Accelerator

    accelerator = Accelerator()
    model = _TinyMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    prepared_model, _ = accelerator.prepare(model, optimizer)
    unwrapped = accelerator.unwrap_model(prepared_model)

    assert unwrapped is prepared_model, (
        "accelerator.unwrap_model() should return the same object "
        "accelerator.prepare() produced when running outside a distributed "
        "launch; if this ever changes, the docs' claim that unwrap_model() "
        "is a no-op on CPU/single-GPU needs to be revisited too."
    )


@pytest.mark.skipif(not HAS_ACCELERATE, reason="accelerate not installed")
def test_accelerate_recipe_over_real_prepare_and_backward():
    """Runs the documented recipe through the real Accelerate library.

    The fake accelerator above pins the TraceML-facing contract cheaply;
    this test instead exercises the real `Accelerator.prepare()` (dataloader
    included) and the real `accelerator.backward()`, so a future Accelerate
    release that changes either path would surface here.
    """
    from accelerate import Accelerator
    from torch.utils.data import DataLoader, TensorDataset

    _reset_traceml_state()
    _install_auto_instrumentation()

    num_steps = 3
    dataset = TensorDataset(
        torch.randn(num_steps * BATCH_SIZE, INPUT_DIM),
        torch.randint(0, NUM_CLASSES, (num_steps * BATCH_SIZE,)),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    model = _TinyMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    accelerator = Accelerator()

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    traced_model = accelerator.unwrap_model(model)

    for batch_x, batch_y in loader:
        with traceml.trace_step(traced_model):
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            accelerator.backward(loss)
            optimizer.step()

    batches = _drain_step_time_queue()
    assert len(batches) == num_steps, (
        f"Expected one StepTimeBatch per step ({num_steps}), "
        f"got {len(batches)}."
    )

    def _every_batch_has(event_name: str) -> bool:
        return all(
            any(evt.name == event_name for evt in batch.events)
            for batch in batches
        )

    for event_name in (
        "_traceml_internal:forward_time",
        "_traceml_internal:backward_time",
        "_traceml_internal:optimizer_step",
        "_traceml_internal:step_time",
        "_traceml_internal:dataloader_next",
    ):
        assert _every_batch_has(event_name), (
            f"{event_name} should be captured for every step when running "
            "the recipe through the real Accelerate prepare/backward path."
        )
