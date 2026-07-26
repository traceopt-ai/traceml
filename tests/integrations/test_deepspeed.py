import pytest
import torch
import torch.nn as nn

import traceml_ai as traceml

try:
    import deepspeed  # noqa: F401

    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

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


class _FakeDeepSpeedEngine(nn.Module):
    """A minimal stand-in for DeepSpeed's ``model_engine``.

    It reproduces the exact surface the TraceML recipe touches:

    - ``.module`` is the unwrapped model (what we pass to ``trace_step``),
    - ``engine(x)`` runs forward through the module,
    - ``engine.backward(loss)`` calls ``loss.backward()`` (real DeepSpeed
      reaches the same ``torch.Tensor.backward`` via its loss scaler), so
      TraceML's backward auto-timer fires,
    - ``engine.step()`` calls a real torch optimizer's ``step()``, so
      TraceML's optimizer hook fires.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

    def forward(self, x):
        return self.module(x)

    def backward(self, loss):
        loss.backward()

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)


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
    # Restore a fresh RECORDING state in case a prior test left the runtime in
    # a draining/complete state (which would silently drop step events).
    configure_trace_recording(max_steps=None)
    _drain_step_time_queue()
    # Iterating a DataLoader to exhaustion leaves one unflushed dataloader_next
    # event from the terminal StopIteration fetch; clear it so it cannot leak
    # into the next test's first StepTimeBatch.
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

    # Install the full auto set that init(mode="auto") installs, including the
    # DataLoader patch, so tests can verify the dataloader_next stream too.
    patch_forward()
    patch_backward()
    patch_h2d()
    patch_dataloader()
    ensure_optimizer_timing_installed()


def test_deepspeed_recipe_brackets_step():
    from traceml_ai.runtime.state import get_trace_session_state

    _reset_traceml_state()
    _install_auto_instrumentation()

    engine = _FakeDeepSpeedEngine(_TinyMLP())
    criterion = nn.CrossEntropyLoss()
    num_steps = 3

    step_before = get_trace_session_state().step

    for _ in range(num_steps):
        # This is exactly the recipe shown in the docs and example.
        with traceml.trace_step(engine.module):
            x = torch.randn(BATCH_SIZE, INPUT_DIM)
            y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

            logits = engine(x)
            loss = criterion(logits, y)

            engine.backward(loss)
            engine.step()

    step_after = get_trace_session_state().step
    assert step_after - step_before == num_steps, (
        "trace_step must advance the TraceML step counter once per "
        f"DeepSpeed step; advanced by {step_after - step_before}."
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
    ), "forward timing should be captured on model_engine.module."
    assert _every_batch_has("_traceml_internal:backward_time"), (
        "backward timing should be captured because engine.backward(loss) "
        "reaches torch.Tensor.backward()."
    )
    assert _every_batch_has("_traceml_internal:optimizer_step"), (
        "optimizer timing should be captured because engine.step() reaches "
        "the underlying torch optimizer's step()."
    )
    assert _every_batch_has(
        "_traceml_internal:step_time"
    ), "step timing should be captured once per trace_step block."


def test_deepspeed_recipe_emits_dataloader_next_over_real_loader():
    """The recipe must emit dataloader_next when iterating a real DataLoader.

    dataloader_next is the fragile stream the docs promise (it records
    DataLoader fetch time). It rides a class-level patch of
    ``DataLoader.__iter__``, so it only lands when a real ``torch`` DataLoader
    is iterated. The recipe iterates the loader OUTSIDE ``trace_step``; the
    fetch is buffered by the process-wide recording gate and flushed into that
    step's StepTimeBatch, so every batch must carry the event. Asserted as rows
    landed, not as "a patch ran".
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

    engine = _FakeDeepSpeedEngine(_TinyMLP())
    criterion = nn.CrossEntropyLoss()

    for batch_x, batch_y in loader:
        # Fetch happens OUTSIDE trace_step, exactly like the documented recipe.
        with traceml.trace_step(engine.module):
            logits = engine(batch_x)
            loss = criterion(logits, batch_y)
            engine.backward(loss)
            engine.step()

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
        "dataloader_next stream is dark for the DeepSpeed recipe over a real "
        f"DataLoader: StepTimeBatch(es) {dark} of {num_steps} carry no event."
    )


@pytest.mark.skipif(not HAS_DEEPSPEED, reason="deepspeed not installed")
def test_deepspeed_public_api_available():
    """When DeepSpeed is installed, the symbols the example relies on exist."""
    import deepspeed

    assert hasattr(deepspeed, "initialize")
    assert hasattr(deepspeed, "init_distributed")
