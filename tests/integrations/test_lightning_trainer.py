"""Real ``Trainer.fit`` tests for the PyTorch Lightning callback.

Every test runs a tiny LightningModule on CPU through a real Lightning
Trainer, in both the ``lightning`` and the legacy ``pytorch_lightning``
namespace, and asserts on the step batches the callback publishes.
"""

import importlib
import logging

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from traceml_ai.instrumentation.hooks.optimizer_hooks import (  # noqa: E402
    reset_optimizer_timing,
)
from traceml_ai.instrumentation.step_events import (  # noqa: E402
    abort_step_capture,
    begin_step_capture,
    drain_step_memory_events,
    drain_step_time_batches,
)
from traceml_ai.integrations import (  # noqa: E402
    lightning as traceml_lightning,
)
from traceml_ai.runtime.state import (  # noqa: E402
    configure_trace_recording,
    get_trace_session_state,
    reset_trace_session_state,
)

STEP = "_traceml_internal:step_time"
FETCH = "_traceml_internal:dataloader_next"
FORWARD = "_traceml_internal:forward_time"
BACKWARD = "_traceml_internal:backward_time"
OPTIMIZER = "_traceml_internal:optimizer_step"

BATCH = 4
ROWS = 16  # four training batches per epoch


@pytest.fixture(params=["lightning", "pytorch_lightning"])
def L(request):
    """The Lightning namespace under test (modern or legacy)."""
    return pytest.importorskip(request.param)


@pytest.fixture(autouse=True)
def _reset_traceml():
    # The callback times the optimizer itself. Global optimizer hooks left
    # installed by an earlier mode="auto" test would double-count it.
    reset_optimizer_timing()
    reset_trace_session_state()
    configure_trace_recording(max_steps=None)
    drain_step_time_batches()
    drain_step_memory_events()
    abort_step_capture(begin_step_capture())
    yield
    drain_step_time_batches()
    drain_step_memory_events()
    abort_step_capture(begin_step_capture())
    reset_optimizer_timing()


def _module_class(L):
    class Tiny(L.LightningModule):
        def __init__(self, *, fail_at_batch=None, manual_steps=0):
            super().__init__()
            self.net = nn.Linear(8, 4)
            self.fail_at_batch = fail_at_batch
            self.manual_steps = int(manual_steps)
            if self.manual_steps:
                self.automatic_optimization = False
            self.losses = []

        def forward(self, x):
            return self.net(x)

        def _loss(self, batch):
            x, y = batch
            return nn.functional.cross_entropy(self(x), y)

        def training_step(self, batch, batch_idx):
            if (
                self.fail_at_batch is not None
                and batch_idx == self.fail_at_batch
            ):
                raise RuntimeError("injected failure")
            if self.manual_steps:
                opt = self.optimizers()
                for _ in range(self.manual_steps):
                    opt.zero_grad()
                    loss = self._loss(batch)
                    self.manual_backward(loss)
                    opt.step()
                self.losses.append(float(loss.detach()))
                return loss
            loss = self._loss(batch)
            self.losses.append(float(loss.detach()))
            return loss

        def validation_step(self, batch, batch_idx):
            return self._loss(batch)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    return Tiny


def _loaders(rows=ROWS, seed=0):
    g = torch.Generator().manual_seed(seed)
    ds = TensorDataset(
        torch.randn(rows, 8, generator=g),
        torch.randint(0, 4, (rows,), generator=g),
    )
    val = TensorDataset(
        torch.randn(rows // 2, 8, generator=g),
        torch.randint(0, 4, (rows // 2,), generator=g),
    )
    return DataLoader(ds, batch_size=BATCH), DataLoader(val, batch_size=BATCH)


def _trainer(L, callbacks, **kwargs):
    defaults = dict(
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        max_epochs=1,
    )
    defaults.update(kwargs)
    return L.Trainer(callbacks=callbacks, **defaults)


def _counts(batches, name):
    return [sum(e.name == name for e in b.events) for b in batches]


def test_lightning_trainer_accumulation_emits_optimizer_only_on_updates(L):
    traceml_lightning.init()
    train, _ = _loaders()
    model = _module_class(L)()
    trainer = _trainer(
        L, [traceml_lightning.TraceMLCallback()], accumulate_grad_batches=2
    )

    trainer.fit(model, train_dataloaders=train)
    batches = drain_step_time_batches()

    assert [b.step for b in batches] == [1, 2, 3, 4]
    assert _counts(batches, FORWARD) == [1, 1, 1, 1]
    assert _counts(batches, BACKWARD) == [1, 1, 1, 1]
    assert _counts(batches, STEP) == [1, 1, 1, 1]
    assert _counts(batches, OPTIMIZER) == [0, 1, 0, 1]
    zero_length = [
        e
        for b in batches
        for e in b.events
        if e.name == OPTIMIZER and e.cpu_start == 0.0 and e.cpu_end == 0.0
    ]
    assert zero_length == [], "no fabricated optimizer measurements"
    assert trainer.global_step == 2


def test_lightning_trainer_validation_fetches_stay_out_of_input_wait(L):
    traceml_lightning.init()
    train, val = _loaders()
    model = _module_class(L)()
    trainer = _trainer(
        L,
        [traceml_lightning.TraceMLCallback()],
        num_sanity_val_steps=2,
        val_check_interval=2,
        limit_val_batches=2,
    )

    trainer.fit(model, train_dataloaders=train, val_dataloaders=val)
    batches = drain_step_time_batches()

    assert [b.step for b in batches] == [1, 2, 3, 4]
    assert _counts(batches, FETCH) == [1, 1, 1, 1]
    # The envelope and forward must survive the mid-epoch validation too.
    assert _counts(batches, STEP) == [1, 1, 1, 1]
    assert _counts(batches, FORWARD) == [1, 1, 1, 1]
    assert begin_step_capture().timing_events == []


def test_lightning_trainer_envelope_opens_before_the_batch_transfer(L):
    traceml_lightning.init()
    callback = traceml_lightning.TraceMLCallback()
    seen = []

    class Probe(_module_class(L)):
        def transfer_batch_to_device(self, batch, device, dataloader_idx):
            # Runs inside strategy.batch_to_device, i.e. during the H2D
            # window. The step envelope must already be open here.
            seen.append(callback._traceml_step_ctx is not None)
            return super().transfer_batch_to_device(
                batch, device, dataloader_idx
            )

    train, _ = _loaders()
    _trainer(L, [callback], max_steps=2).fit(Probe(), train_dataloaders=train)

    assert seen == [True, True]
    assert _counts(drain_step_time_batches(), STEP) == [1, 1]


def test_lightning_trainer_injected_failure_discards_the_partial_step(L):
    traceml_lightning.init()
    train, _ = _loaders()
    callback = traceml_lightning.TraceMLCallback()
    model = _module_class(L)(fail_at_batch=2)
    trainer = _trainer(L, [callback])

    with pytest.raises(RuntimeError, match="injected failure"):
        trainer.fit(model, train_dataloaders=train)

    batches = drain_step_time_batches()
    assert [b.step for b in batches] == [1, 2]
    assert get_trace_session_state().step == 2
    assert begin_step_capture().timing_events == []
    assert begin_step_capture().memory_event is None
    assert callback._traceml_step_ctx is None
    assert callback._backward_ctx is None
    assert callback._optimizer_ctx is None
    assert "forward" not in model.__dict__

    # A fresh fit after the failure starts clean.
    fresh = _module_class(L)()
    _trainer(L, [callback], max_steps=2).fit(fresh, train_dataloaders=train)
    later = drain_step_time_batches()
    assert [b.step for b in later] == [3, 4]
    assert _counts(later, FETCH) == [1, 1]
    assert _counts(later, FORWARD) == [1, 1]
    assert _counts(later, STEP) == [1, 1]


def test_lightning_trainer_second_fit_does_not_inherit_pending_events(L):
    traceml_lightning.init()
    train, _ = _loaders()
    callback = traceml_lightning.TraceMLCallback()

    # A full epoch exhausts the loader; the fetch that raises StopIteration
    # is recorded after the last batch completed.
    _trainer(L, [callback], max_epochs=1).fit(
        _module_class(L)(), train_dataloaders=train
    )
    first = drain_step_time_batches()
    assert _counts(first, FETCH) == [1, 1, 1, 1]
    assert (
        begin_step_capture().timing_events == []
    ), "teardown must drop what is still pending"

    _trainer(L, [callback], max_steps=2).fit(
        _module_class(L)(), train_dataloaders=train
    )
    second = drain_step_time_batches()
    assert [b.step for b in second] == [5, 6]
    assert _counts(second, FETCH) == [1, 1]
    assert _counts(second, FORWARD) == [1, 1]


def test_lightning_trainer_standalone_validate_publishes_nothing(L):
    traceml_lightning.init()
    _, val = _loaders()
    callback = traceml_lightning.TraceMLCallback()

    _trainer(L, [callback], limit_val_batches=2).validate(
        _module_class(L)(), dataloaders=val
    )

    assert drain_step_time_batches() == []
    assert begin_step_capture().timing_events == []
    assert callback._suppress_cm is None
    assert callback._suppress_depth == 0
    assert callback._traceml_step_ctx is None


def test_lightning_trainer_iterator_mode_uses_the_fallback_open(L):
    # training_step(dataloader_iter): Lightning hands over the iterator and
    # never calls strategy.batch_to_device, so the envelope must open at
    # on_train_batch_start. The user's fetch then happens inside it, which
    # the docs call out; this test pins that the step is still published.
    traceml_lightning.init()
    train, _ = _loaders()

    class IterMode(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(8, 4)

        def forward(self, x):
            return self.net(x)

        def training_step(self, dataloader_iter):
            # Lightning's iterator mode yields (batch, batch_idx, dl_idx).
            batch, _, _ = next(dataloader_iter)
            x, y = batch
            return nn.functional.cross_entropy(self(x), y)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    _trainer(L, [traceml_lightning.TraceMLCallback()], max_steps=2).fit(
        IterMode(), train_dataloaders=train
    )

    batches = drain_step_time_batches()
    assert [b.step for b in batches] == [1, 2]
    assert _counts(batches, STEP) == [1, 1]
    assert _counts(batches, FORWARD) == [1, 1]
    assert _counts(batches, FETCH) == [1, 1]


def test_lightning_trainer_callback_preserves_training(L):
    def run(with_callback):
        reset_trace_session_state()
        torch.manual_seed(42)
        train, _ = _loaders()
        model = _module_class(L)()
        callbacks = []
        if with_callback:
            traceml_lightning.init()
            callbacks = [traceml_lightning.TraceMLCallback()]
        _trainer(L, callbacks).fit(model, train_dataloaders=train)
        drained = len(drain_step_time_batches())
        return model.losses, model.state_dict(), drained

    untraced_losses, untraced_state, untraced_n = run(False)
    traced_losses, traced_state, traced_n = run(True)

    assert untraced_n == 0
    assert traced_n == ROWS // BATCH
    assert traced_losses == pytest.approx(untraced_losses)
    for key in untraced_state:
        torch.testing.assert_close(traced_state[key], untraced_state[key])


def test_lightning_trainer_manual_optimization_times_each_optimizer_step(L):
    traceml_lightning.init()
    train, _ = _loaders()
    model = _module_class(L)(manual_steps=2)
    # max_steps counts optimizer steps (two per batch here), so bound the
    # run by batches instead.
    _trainer(
        L, [traceml_lightning.TraceMLCallback()], limit_train_batches=2
    ).fit(model, train_dataloaders=train)

    batches = drain_step_time_batches()
    assert [b.step for b in batches] == [1, 2]
    assert _counts(batches, OPTIMIZER) == [2, 2]
    assert _counts(batches, BACKWARD) == [2, 2]
    assert _counts(batches, STEP) == [1, 1]


def test_lightning_trainer_warns_when_owed_streams_would_be_dark(L, caplog):
    import traceml_ai as traceml

    # Manual mode installs no automatic patches, so the two patch-gated
    # streams the callback owes (dataloader fetch, H2D) will be dark.
    traceml.init(mode="manual")
    train, _ = _loaders()
    model = _module_class(L)()

    with caplog.at_level(logging.WARNING):
        _trainer(L, [traceml_lightning.TraceMLCallback()], max_steps=1).fit(
            model, train_dataloaders=train
        )

    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "dataloader_fetch" in m and "h2d" in m for m in messages
    ), messages
    drain_step_time_batches()


def test_lightning_namespace_under_test_is_the_requested_one(L):
    # Guard for the parametrization itself: the legacy namespace must really
    # be the one driving the Trainer, not an alias of the modern package.
    assert importlib.import_module(L.__name__) is L
    assert L.Trainer.__module__.startswith(L.__name__)
