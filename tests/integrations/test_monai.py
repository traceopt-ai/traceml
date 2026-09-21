"""Real ``SupervisedTrainer`` tests for the MONAI handler.

Every test runs a tiny MONAI engine on CPU and asserts on the step batches
the handler publishes.
"""

import time
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")
pytest.importorskip("ignite")

from ignite.engine import Events  # noqa: E402
from monai.data import DataLoader, ThreadDataLoader  # noqa: E402
from monai.engines import (  # noqa: E402
    AdversarialTrainer,
    GanTrainer,
    SupervisedEvaluator,
    SupervisedTrainer,
)
from monai.engines.utils import default_prepare_batch  # noqa: E402
from monai.handlers import ValidationHandler  # noqa: E402
from torch.utils.data import Dataset, IterableDataset  # noqa: E402

import traceml_ai as traceml  # noqa: E402
from traceml_ai.instrumentation.hooks.optimizer_hooks import (  # noqa: E402
    reset_optimizer_timing,
)
from traceml_ai.instrumentation.patches import (  # noqa: E402
    h2d_auto_timer_patch,
)
from traceml_ai.instrumentation.patches.dataloader_patch import (  # noqa: E402
    _DL_TLS,
)
from traceml_ai.instrumentation.step_events import (  # noqa: E402
    abort_step_capture,
    begin_step_capture,
    drain_step_memory_events,
    drain_step_time_batches,
)
from traceml_ai.integrations import monai as traceml_monai  # noqa: E402
from traceml_ai.runtime.state import (  # noqa: E402
    configure_trace_recording,
    get_trace_session_state,
    reset_trace_session_state,
)
from traceml_ai.step_time.model import STEP_TIME_EVENT_NAMES  # noqa: E402
from traceml_ai.utils.timing import timed_region  # noqa: E402

STEP = "_traceml_internal:step_time"
FETCH = "_traceml_internal:dataloader_next"
FORWARD = "_traceml_internal:forward_time"
BACKWARD = "_traceml_internal:backward_time"
OPTIMIZER = "_traceml_internal:optimizer_step"
WARNING = "[TraceML] MONAI:"
ITEMS = 8  # four batches of two


@pytest.fixture(autouse=True)
def _reset_traceml(monkeypatch):
    # The fetch patch is process-wide and nothing in src/ removes it, so a
    # file that ran init() before this one would leave the handler's Input
    # Wait counted twice. Restoring it here keeps the isolation local: other
    # integrations' tests keep whatever state they already relied on.
    from torch.utils.data import DataLoader

    from traceml_ai.instrumentation.patches import dataloader_patch

    monkeypatch.setattr(
        DataLoader, "__iter__", dataloader_patch._ORIG_DATALOADER_ITER
    )
    monkeypatch.setattr(DataLoader, "_traceml_patched", False, raising=False)
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


def _item():
    return {"image": torch.zeros(1, 8, 8), "label": torch.zeros(1, 8, 8)}


class Volumes(Dataset):
    def __init__(self, n=ITEMS, delay=0.0):
        self.n, self.delay = n, delay

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        time.sleep(self.delay)
        return _item()


class Stream(IterableDataset):
    """The same items with no length, so Ignite learns it at epoch end."""

    def __iter__(self):
        return (_item() for _ in range(ITEMS))


class FailsAt(torch.nn.Module):
    def __init__(self, call):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, padding=1)
        self.calls, self.fail_at = 0, call

    def forward(self, x):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("boom")
        return self.conv(x)


class InheritsIteration(SupervisedTrainer):
    pass


class OverridesIteration(SupervisedTrainer):
    def _iteration(self, engine, batchdata):
        return super()._iteration(engine, batchdata)


def _trainer(handlers, cls=SupervisedTrainer, loader=None, network=None, **kw):
    if network is None:
        network = torch.nn.Conv2d(1, 1, 3, padding=1)
    if loader is None:  # never ``or``: bool() would call len() on the loader
        loader = DataLoader(Volumes(), batch_size=2)
    return cls(
        device=torch.device("cpu"),
        max_epochs=kw.pop("max_epochs", 1),
        train_data_loader=loader,
        network=network,
        optimizer=torch.optim.SGD(network.parameters(), lr=0.01),
        loss_function=torch.nn.MSELoss(),
        train_handlers=handlers,
        **kw,
    )


def _evaluator(handlers=(), delay=0.0):
    return SupervisedEvaluator(
        device=torch.device("cpu"),
        val_data_loader=DataLoader(Volumes(4, delay), batch_size=2),
        network=torch.nn.Conv2d(1, 1, 3, padding=1),
        val_handlers=list(handlers),
    )


def _gan(handlers):
    g = torch.nn.Sequential(
        torch.nn.Linear(16, 64), torch.nn.Unflatten(1, (1, 8, 8))
    )
    d = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 1))
    return GanTrainer(
        device=torch.device("cpu"),
        max_epochs=1,
        train_data_loader=DataLoader(Volumes(), batch_size=2),
        g_network=g,
        g_optimizer=torch.optim.SGD(g.parameters(), lr=0.01),
        g_loss_function=lambda fake: d(fake).mean(),
        d_network=d,
        d_optimizer=torch.optim.SGD(d.parameters(), lr=0.01),
        d_loss_function=lambda f, r: d(f.detach()).mean() - d(r[0]).mean(),
        latent_shape=16,
        train_handlers=handlers,
    )


def _adversarial(handlers):
    g = torch.nn.Conv2d(1, 1, 3, padding=1)
    d = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 1))
    return AdversarialTrainer(
        device=torch.device("cpu"),
        max_epochs=1,
        train_data_loader=DataLoader(Volumes(), batch_size=2),
        g_network=g,
        g_optimizer=torch.optim.SGD(g.parameters(), lr=0.01),
        g_loss_function=lambda logits: logits.mean(),
        recon_loss_function=torch.nn.MSELoss(),
        d_network=d,
        d_optimizer=torch.optim.SGD(d.parameters(), lr=0.01),
        d_loss_function=lambda real, fake: fake.mean() - real.mean(),
        train_handlers=handlers,
    )


def _published():
    return {b.step: b.events for b in drain_step_time_batches()}


def _count(events, name):
    return sum(e.name == name for e in events)


def _pending():
    return len(begin_step_capture().timing_events)


def _outside(events, windows):
    return all(
        e.cpu_end <= start or e.cpu_start >= end
        for e in events
        for start, end in windows
    )


@pytest.mark.parametrize(
    "loader_kind, accumulation, epochs, sizes, steps_at",
    [
        ("plain", 1, 1, [1, 1, 1, 1], [1, 2, 3, 4]),
        ("threaded", 1, 1, [1, 1, 1, 1], [1, 2, 3, 4]),
        ("plain", 2, 1, [2, 2], [2, 4]),
        ("plain", 3, 1, [3, 1], [3, 4]),
        # From epoch 2 the rule is epoch-local, not a raw iteration count.
        ("plain", 3, 2, [3, 1, 3, 1], [3, 4, 7, 8]),
        # A loader with no length takes MONAI's other accumulation branch.
        ("stream", 2, 1, [2, 2], [2, 4]),
        # Two epochs on a length-less loader: the epoch boundary lands
        # mid-window (raw iteration 4 of 8), MONAI zeroes its gradients
        # without stepping them, and the dropped iteration must not inflate
        # the group that steps next.
        ("stream", 3, 2, [3, 3, 1], [3, 7, 8]),
    ],
)
def test_one_step_per_optimizer_update(
    loader_kind, accumulation, epochs, sizes, steps_at
):
    traceml_monai.init()
    loaders = {
        "plain": lambda: DataLoader(Volumes(), batch_size=2),
        "threaded": lambda: ThreadDataLoader(Volumes(), batch_size=2),
        "stream": lambda: DataLoader(Stream(), batch_size=2),
    }
    trainer = _trainer(
        [traceml_monai.TraceMLHandler()],
        loader=loaders[loader_kind](),
        accumulation_steps=accumulation,
        max_epochs=epochs,
    )
    updates = []
    trainer.optimizer.register_step_post_hook(
        lambda *a: updates.append(trainer.state.iteration)
    )
    trainer.run()

    # The count alone does not pin WHICH iterations MONAI actually stepped
    # on; a handler whose grouping rule drifted from MONAI's could still
    # publish the right number of same-sized groups over the wrong
    # iterations. Pin the real iteration numbers too.
    assert updates == steps_at

    steps = _published()
    assert sorted(steps) == list(range(1, len(updates) + 1))
    assert [_count(steps[s], STEP) for s in sorted(steps)] == sizes
    assert [_count(steps[s], FETCH) for s in sorted(steps)] == sizes
    # One forward and one backward per iteration of the group, and exactly
    # one optimizer step for the group itself.
    assert [_count(steps[s], FORWARD) for s in sorted(steps)] == sizes
    assert [_count(steps[s], BACKWARD) for s in sorted(steps)] == sizes
    assert [_count(steps[s], OPTIMIZER) for s in sorted(steps)] == [1] * len(
        sizes
    )
    assert len(drain_step_memory_events()) == len(updates)
    assert _pending() == 0


ENGINES = {
    "SupervisedTrainer": (_trainer, True),
    "inherits _iteration": (
        lambda h: _trainer(h, cls=InheritsIteration),
        True,
    ),
    "overrides _iteration": (
        lambda h: _trainer(h, cls=OverridesIteration),
        False,
    ),
    "iteration_update": (
        lambda h: _trainer(h, iteration_update=lambda e, b: {}),
        False,
    ),
    "iteration_update from another trainer": (
        lambda h: _trainer(h, iteration_update=_trainer([])._iteration),
        False,
    ),
    "GanTrainer": (_gan, False),
    "AdversarialTrainer": (_adversarial, False),
    "SupervisedEvaluator": (_evaluator, False),
}


def test_the_handler_writes_the_names_the_summary_reads():
    assert traceml_monai._STEP == STEP_TIME_EVENT_NAMES["traced_step_time"]
    assert traceml_monai._FETCH == STEP_TIME_EVENT_NAMES["input_wait"]
    assert STEP == traceml_monai._STEP and FETCH == traceml_monai._FETCH


def _phase_ms(events, name):
    return sum(
        (e.cpu_end - e.cpu_start) * 1000 for e in events if e.name == name
    )


def _slow_net(seconds):
    class Slow(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 1, 3, padding=1)

        def forward(self, x):
            time.sleep(seconds)
            return self.conv(x)

    return Slow()


def test_published_durations_are_real_measurements():
    """Counting events is not enough: the clocks have to mean something."""
    traceml_monai.init()
    waits = {}
    for kind in ("plain", "threaded"):
        drain_step_time_batches()
        loader = (
            ThreadDataLoader(Volumes(delay=0.02), batch_size=2, buffer_size=8)
            if kind == "threaded"
            else DataLoader(Volumes(delay=0.02), batch_size=2)
        )
        # Compute is twice the cost of a batch, so a loader that prefetches
        # has ample room to stay ahead even on a busy machine.
        _trainer(
            [traceml_monai.TraceMLHandler()],
            loader=loader,
            network=_slow_net(0.08),
        ).run()
        events = [e for ev in _published().values() for e in ev]
        waits[kind] = _phase_ms(events, FETCH)
        assert _phase_ms(events, STEP) > 0

    # Four batches of two items, 20 ms per item, so a loader the loop waits
    # on costs it about 160 ms in total.
    assert waits["plain"] > 80
    # The threaded loader overlaps loading with compute, so the loop waits
    # for far less. This is the whole reason Input Wait comes from the
    # engine bracket rather than the torch patch.
    assert waits["threaded"] < waits["plain"] / 2


def _slow_loss(seconds):
    mse = torch.nn.MSELoss()

    def loss(prediction, target):
        time.sleep(seconds)
        return mse(prediction, target)

    return loss


def test_forward_excludes_zero_grad():
    """The forward window is the inferer call, not the whole iteration."""
    traceml_monai.init()

    class SlowZeroGrad(torch.optim.SGD):
        def zero_grad(self, *args, **kwargs):
            time.sleep(0.02)
            return super().zero_grad(*args, **kwargs)

    network = torch.nn.Conv2d(1, 1, 3, padding=1)
    trainer = SupervisedTrainer(
        device=torch.device("cpu"),
        max_epochs=1,
        train_data_loader=DataLoader(Volumes(), batch_size=2),
        network=network,
        optimizer=SlowZeroGrad(network.parameters(), lr=0.01),
        loss_function=_slow_loss(0.02),
        train_handlers=[traceml_monai.TraceMLHandler()],
    )
    trainer.run()

    steps = _published()
    assert len(steps) == ITEMS // 2
    for events in steps.values():
        # zero_grad runs before the inferer call and the loss after it.
        # Both are inside the step and outside forward.
        assert _count(events, FORWARD) == 1
        assert _phase_ms(events, FORWARD) > 0
        # 20 ms in zero_grad and 20 ms in the loss are inside the step and
        # outside forward, so forward is a small share of the step.
        assert _phase_ms(events, STEP) > 40
        assert _phase_ms(events, FORWARD) < _phase_ms(events, STEP) / 4


def test_nothing_stays_hooked_or_wrapped_after_a_run():
    traceml_monai.init()
    trainer = _trainer([traceml_monai.TraceMLHandler()])
    original_inferer = trainer.inferer
    trainer.run()

    assert trainer.inferer is original_inferer
    assert len(trainer.optimizer._optimizer_step_pre_hooks) == 0
    assert len(trainer.optimizer._optimizer_step_post_hooks) == 0


def test_nothing_stays_hooked_or_wrapped_after_a_crash():
    traceml_monai.init()
    trainer = _trainer([traceml_monai.TraceMLHandler()], network=FailsAt(3))
    original_inferer = trainer.inferer

    with pytest.raises(RuntimeError, match="boom"):
        trainer.run()

    assert trainer.inferer is original_inferer
    assert len(trainer.optimizer._optimizer_step_pre_hooks) == 0
    assert len(trainer.optimizer._optimizer_step_post_hooks) == 0


def test_an_optimizer_wrapped_by_the_sdk_is_not_timed_twice():
    traceml_monai.init()
    trainer = _trainer([traceml_monai.TraceMLHandler()])
    traceml.wrap_optimizer(trainer.optimizer)
    trainer.run()

    steps = _published()
    assert sorted(steps) == [1, 2, 3, 4]
    assert all(_count(ev, OPTIMIZER) == 1 for ev in steps.values())


def test_step_time_covers_the_work_of_the_iteration():
    traceml_monai.init()
    _trainer([traceml_monai.TraceMLHandler()], network=_slow_net(0.02)).run()
    for events in _published().values():
        assert _phase_ms(events, STEP) > 15


@pytest.mark.parametrize("name", sorted(ENGINES))
def test_only_supervised_trainer_iteration_is_traced(name, capsys):
    make, traced = ENGINES[name]
    traceml_monai.init()
    engine = make([traceml_monai.TraceMLHandler()])
    assert capsys.readouterr().err.count(WARNING) == (0 if traced else 1)

    engine.run()
    engine.run()

    assert len(_published()) == (8 if traced else 0)
    assert capsys.readouterr().err.count(WARNING) == 0
    assert _pending() == 0


def test_exception_propagates_and_leaves_nothing_behind():
    traceml_monai.init()
    handler = traceml_monai.TraceMLHandler()
    trainer = _trainer([handler], network=FailsAt(3))
    original_prepare = trainer.prepare_batch

    with pytest.raises(RuntimeError, match="boom"):
        trainer.run()

    assert sorted(_published()) == [1, 2]
    assert _pending() == 0
    assert trainer.prepare_batch is original_prepare
    raised = trainer._event_handlers.get(Events.EXCEPTION_RAISED, [])
    assert all(getattr(h, "__self__", None) is not handler for h, *_ in raised)

    _trainer([traceml_monai.TraceMLHandler()]).run()
    steps = _published()
    assert sorted(steps) == [3, 4, 5, 6]
    assert all(_count(ev, FETCH) == 1 for ev in steps.values())


def test_one_handler_traces_one_trainer(capsys):
    traceml_monai.init()
    handler = traceml_monai.TraceMLHandler()
    trainer = _trainer([handler])
    handler.attach(trainer)
    second = _trainer([handler])
    assert capsys.readouterr().err.count(WARNING) == 2

    trainer.run()
    steps = _published()
    assert sorted(steps) == [1, 2, 3, 4]
    assert all(
        _count(ev, STEP) == _count(ev, FETCH) == 1 for ev in steps.values()
    )

    second.run()
    assert _published() == {}


def test_instance_shared_with_evaluator_traces_the_trainer(capsys):
    traceml_monai.init()
    handler = traceml_monai.TraceMLHandler()
    evaluator = _evaluator([handler])
    validation = ValidationHandler(1, validator=evaluator, epoch_level=True)
    trainer = _trainer([validation, handler], max_epochs=2)
    trainer.run()

    assert sorted(_published()) == list(range(1, 9))
    err = capsys.readouterr().err
    assert err.count(WARNING) == 1
    assert "SupervisedEvaluator" in err


def test_events_pending_before_the_run_do_not_reach_step_one():
    traceml_monai.init()
    for _ in range(100):
        with timed_region(FETCH):
            pass
    assert _pending() == 100

    _trainer([traceml_monai.TraceMLHandler()]).run()
    assert _count(_published()[1], FETCH) == 1


@pytest.mark.parametrize("validation_first", [True, False])
def test_iteration_level_validation_stays_outside_the_step(validation_first):
    traceml_monai.init()
    evaluator = _evaluator(delay=0.01)
    windows = []
    evaluator.add_event_handler(
        Events.STARTED, lambda e: windows.append([time.time(), None])
    )
    evaluator.add_event_handler(
        Events.COMPLETED, lambda e: windows[-1].__setitem__(1, time.time())
    )
    validation = ValidationHandler(2, validator=evaluator, epoch_level=False)
    handler = traceml_monai.TraceMLHandler()
    order = (
        [validation, handler] if validation_first else [handler, validation]
    )
    _trainer(order).run()

    events = [e for ev in _published().values() for e in ev]
    assert len(windows) == 2
    assert _count(events, STEP) == 4
    assert _outside([e for e in events if e.name == STEP], windows)


def test_unmatched_fetch_start_on_a_length_less_loader_is_dropped():
    traceml_monai.init()
    windows = []

    def epoch_end_work(engine):
        start = time.time()
        time.sleep(0.02)
        windows.append((start, time.time()))

    trainer = _trainer(
        [traceml_monai.TraceMLHandler()],
        loader=DataLoader(Stream(), batch_size=2),
        max_epochs=2,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, epoch_end_work)
    trainer.run()

    steps = _published()
    assert sorted(steps) == list(range(1, 9))
    assert all(_count(ev, FETCH) == 1 for ev in steps.values())
    fetches = [e for ev in steps.values() for e in ev if e.name == FETCH]
    assert _outside(fetches, windows)


@pytest.mark.parametrize(
    "where",
    ["_start_run", "_open_step", "_end_run", "_discard_if_zeroed"],
)
def test_internal_error_never_reaches_training(where, monkeypatch, capsys):
    traceml_monai.init()
    handler = traceml_monai.TraceMLHandler()

    def fail(*args, **kwargs):
        raise RuntimeError("internal")

    monkeypatch.setattr(handler, where, fail)
    trainer = _trainer([handler])
    trainer.run()

    assert trainer.state.iteration == ITEMS // 2
    assert capsys.readouterr().err.count(WARNING) == 1
    assert _pending() == 0
    # Nothing opened a step here, so nothing counted one either.
    assert get_trace_session_state().step == len(_published())


def test_a_cleanup_failure_still_restores_prepare_batch(monkeypatch, capsys):
    traceml_monai.init()
    handler = traceml_monai.TraceMLHandler()
    trainer = _trainer([handler])
    original_prepare = trainer.prepare_batch

    def fail():
        raise RuntimeError("internal")

    monkeypatch.setattr(handler, "_abandon", fail)
    trainer.run()

    assert trainer.prepare_batch is original_prepare
    assert capsys.readouterr().err.count(WARNING) == 1


def test_a_failing_h2d_timer_never_reaches_training(monkeypatch, capsys):
    traceml_monai.init()

    def fail():
        raise RuntimeError("internal")

    monkeypatch.setattr(traceml_monai, "_enter_h2d", fail)
    trainer = _trainer([traceml_monai.TraceMLHandler()])
    trainer.prepare_batch = lambda b, *a, **k: default_prepare_batch(b)
    trainer.state.device = torch.device("cuda:0")
    trainer.run()

    assert trainer.state.iteration == ITEMS // 2
    assert sorted(_published()) == [1, 2, 3, 4]
    assert capsys.readouterr().err.count(WARNING) == 1


@pytest.mark.parametrize("phase", ["_open_step", "_hook_optimizer"])
def test_a_run_scoped_failure_reports_again_on_the_next_run(
    phase, monkeypatch, capsys
):
    traceml_monai.init()
    handler = traceml_monai.TraceMLHandler()

    def fail(*args, **kwargs):
        raise RuntimeError("internal")

    monkeypatch.setattr(handler, phase, fail)
    trainer = _trainer([handler])
    trainer.run()
    trainer.run()

    # Once per run: a silent second run would look healthy.
    assert capsys.readouterr().err.count(WARNING) == 2


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_h2d_timing_wraps_the_transfer_only_on_cuda(device):
    traceml_monai.init()
    trainer = _trainer([traceml_monai.TraceMLHandler()])
    seen = []

    def prepare(batchdata, device=None, non_blocking=False, **kwargs):
        seen.append(h2d_auto_timer_patch._enabled())
        return default_prepare_batch(batchdata, torch.device("cpu"))

    trainer.prepare_batch = prepare
    trainer.state.device = torch.device(device)
    trainer.run()

    assert seen == [device.startswith("cuda")] * (ITEMS // 2)
    assert h2d_auto_timer_patch._enabled() is False


def test_cuda_without_h2d_timing_warns(capsys):
    traceml.init(mode="manual")
    trainer = _trainer([traceml_monai.TraceMLHandler()])
    trainer.prepare_batch = lambda b, *a, **k: default_prepare_batch(b)
    trainer.state.device = torch.device("cuda:0")
    trainer.run()

    err = capsys.readouterr().err
    assert err.count(WARNING) == 1
    assert "H2D" in err


def test_init_with_the_fetch_patch_on_traces_nothing(capsys):
    traceml.init()  # mode="auto" installs the DataLoader fetch patch
    trainer = _trainer([traceml_monai.TraceMLHandler()])
    original_prepare = trainer.prepare_batch
    trainer.run()
    trainer.run()

    assert _published() == {}
    assert trainer.prepare_batch is original_prepare
    err = capsys.readouterr().err
    assert err.count(WARNING) == 1
    assert "traceml_ai.integrations.monai.init()" in err


def test_a_patch_installed_outside_init_is_refused(capsys):
    from traceml_ai.instrumentation.patches.dataloader_patch import (
        patch_dataloader,
    )

    patch_dataloader()  # what another integration's init() leaves installed
    traceml_monai.init()
    trainer = _trainer([traceml_monai.TraceMLHandler()])
    trainer.run()

    err = capsys.readouterr().err
    assert _published() == {}
    assert err.count(WARNING) == 1
    assert "fetch patch is installed" in err
    # Declining must not change timing policy for anything else running
    # in this process.
    assert getattr(_DL_TLS, "_traceml_dl_require_scope", False) is False


def test_the_kill_switch_flipped_during_a_run_stops_publishing(monkeypatch):
    traceml_monai.init()
    handler = traceml_monai.TraceMLHandler()
    trainer = _trainer([handler])
    original_prepare = trainer.prepare_batch
    pending_per_iteration = []
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(once=2),
        lambda e: monkeypatch.setenv("TRACEML_DISABLED", "1"),
    )
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED,
        lambda e: pending_per_iteration.append(_pending()),
    )
    trainer.run()

    assert sorted(_published()) == [1, 2]
    assert _pending() == 0
    assert trainer.prepare_batch is original_prepare
    # Every phase reads the switch, so the disabled iterations record
    # nothing at all rather than filling a capture nobody publishes.
    assert pending_per_iteration == [0, 0, 0, 0]


@pytest.mark.parametrize("phase", ["_wrap_inferer", "_hook_optimizer"])
def test_a_phase_that_cannot_be_installed_degrades_alone(
    phase, monkeypatch, capsys
):
    """One optional phase failing must not take the whole run dark."""
    traceml_monai.init()
    handler = traceml_monai.TraceMLHandler()

    def fail(*args, **kwargs):
        raise RuntimeError("internal")

    monkeypatch.setattr(handler, phase, fail)
    trainer = _trainer([handler])
    trainer.run()

    steps = _published()
    assert sorted(steps) == [1, 2, 3, 4]
    assert all(
        _count(ev, STEP) == _count(ev, FETCH) == 1 for ev in steps.values()
    )
    assert all(_count(ev, BACKWARD) == 1 for ev in steps.values())
    missing = FORWARD if phase == "_wrap_inferer" else OPTIMIZER
    assert all(_count(ev, missing) == 0 for ev in steps.values())
    assert _pending() == 0
    assert capsys.readouterr().err.count(WARNING) == 1


def test_an_optimizer_without_torch_step_hooks_loses_only_that_phase(capsys):
    """MONAI accepts any object with step() and zero_grad()."""
    traceml_monai.init()
    network = torch.nn.Conv2d(1, 1, 3, padding=1)
    inner = torch.optim.SGD(network.parameters(), lr=0.01)

    class DuckOptimizer:
        param_groups = inner.param_groups

        def step(self, *args, **kwargs):
            return inner.step(*args, **kwargs)

        def zero_grad(self, *args, **kwargs):
            return inner.zero_grad(*args, **kwargs)

    trainer = SupervisedTrainer(
        device=torch.device("cpu"),
        max_epochs=1,
        train_data_loader=DataLoader(Volumes(), batch_size=2),
        network=network,
        optimizer=DuckOptimizer(),
        loss_function=torch.nn.MSELoss(),
        train_handlers=[traceml_monai.TraceMLHandler()],
    )
    trainer.run()

    steps = _published()
    assert sorted(steps) == [1, 2, 3, 4]
    assert all(
        _count(ev, FORWARD) == _count(ev, BACKWARD) == 1
        for ev in steps.values()
    )
    assert all(_count(ev, OPTIMIZER) == 0 for ev in steps.values())
    err = capsys.readouterr().err
    assert err.count(WARNING) == 1
    assert "optimizer time is not measured" in err


def test_the_optimizer_window_is_the_step_call():
    """The window must be optimizer.step(), not the events around it."""
    traceml_monai.init()

    class SlowStep(torch.optim.SGD):
        def step(self, *args, **kwargs):
            time.sleep(0.02)
            return super().step(*args, **kwargs)

    network = torch.nn.Conv2d(1, 1, 3, padding=1)
    trainer = SupervisedTrainer(
        device=torch.device("cpu"),
        max_epochs=1,
        train_data_loader=DataLoader(Volumes(), batch_size=2),
        network=network,
        optimizer=SlowStep(network.parameters(), lr=0.01),
        # Five times the optimizer's own sleep, so a window that swallowed the
        # loss lands past 100 ms and the ceiling below still leaves room for a
        # busy machine to overshoot a 20 ms sleep.
        loss_function=_slow_loss(0.10),
        train_handlers=[traceml_monai.TraceMLHandler()],
    )
    trainer.run()

    for events in _published().values():
        # torch wraps step() per class, so a subclass calling super().step()
        # fires the hook pair twice; only one event may be recorded.
        assert _count(events, OPTIMIZER) == 1
        assert _phase_ms(events, OPTIMIZER) > 15
        # The loss sleeps too, and it is not the optimizer's time.
        assert _phase_ms(events, OPTIMIZER) < 60


def test_an_optimizer_step_between_iterations_is_not_timed():
    """The hooks stay registered for the run, so they must be gated."""
    traceml_monai.init()
    trainer = _trainer([traceml_monai.TraceMLHandler()])
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(once=1),
        lambda e: e.optimizer.step(),
    )
    trainer.run()

    steps = _published()
    assert sorted(steps) == [1, 2, 3, 4]
    # The extra step belongs to no iteration, so it is charged to none.
    assert all(_count(ev, OPTIMIZER) == 1 for ev in steps.values())


def test_an_inferer_call_between_iterations_is_not_timed():
    """The proxy is installed for the whole run, so it must be gated too."""
    traceml_monai.init()
    trainer = _trainer([traceml_monai.TraceMLHandler()])
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(once=1),
        lambda e: e.inferer(torch.zeros(1, 1, 8, 8), e.network),
    )
    trainer.run()

    steps = _published()
    assert sorted(steps) == [1, 2, 3, 4]
    # The extra call belongs to no iteration, so it is charged to none.
    assert all(_count(ev, FORWARD) == 1 for ev in steps.values())


def test_the_amp_branch_still_records_one_optimizer_event_per_group():
    """MONAI takes a different code path when amp=True.

    torch disables a CPU GradScaler, so the scaler's skip-on-inf path is
    not reachable here; this pins the branch, not the skip.
    """
    traceml_monai.init()
    trainer = _trainer([traceml_monai.TraceMLHandler()], amp=True)
    trainer.run()

    steps = _published()
    assert sorted(steps) == [1, 2, 3, 4]
    assert all(_count(ev, OPTIMIZER) == 1 for ev in steps.values())
    # MONAI fires BACKWARD_COMPLETED from a different line on this branch,
    # so the backward window has to be asserted here too.
    assert all(_count(ev, BACKWARD) == 1 for ev in steps.values())


def test_the_backward_window_is_the_backward_pass():
    traceml_monai.init()

    class SlowBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, grad):
            time.sleep(0.02)
            return grad

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 1, 3, padding=1)

        def forward(self, x):
            return SlowBackward.apply(self.conv(x))

    trainer = _trainer([traceml_monai.TraceMLHandler()], network=Network())
    trainer.run()

    for events in _published().values():
        assert _count(events, BACKWARD) == 1
        assert _phase_ms(events, BACKWARD) > 15
        # The backward pass is not the forward pass.
        assert _phase_ms(events, FORWARD) < _phase_ms(events, BACKWARD)


def test_state_written_to_the_traced_inferer_survives_the_run():
    """Tracing must not swallow what a handler stores on the inferer.

    The proxy is swapped out at run end, so a write that landed on the
    proxy instead of the real inferer would disappear with it.
    """
    traceml_monai.init()
    trainer = _trainer([traceml_monai.TraceMLHandler()])
    original = trainer.inferer
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(once=1),
        lambda e: setattr(e.inferer, "roi_state", "written mid-run"),
    )
    trainer.run()

    assert trainer.inferer is original
    assert original.roi_state == "written mid-run"


def test_an_inferer_replaced_during_the_run_is_left_alone(capsys):
    traceml_monai.init()
    trainer = _trainer([traceml_monai.TraceMLHandler()])
    from monai.inferers import SimpleInferer

    later = SimpleInferer()
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(once=1),
        lambda e: setattr(e, "inferer", later),
    )
    trainer.run()

    steps = _published()
    assert trainer.inferer is later
    assert _count(steps[1], FORWARD) == 1
    assert _count(steps[4], FORWARD) == 0
    err = capsys.readouterr().err
    assert err.count(WARNING) == 1
    assert "inferer was replaced" in err


def test_kill_switch_wraps_and_records_nothing(monkeypatch):
    traceml_monai.init()
    monkeypatch.setenv("TRACEML_DISABLED", "1")
    trainer = _trainer([traceml_monai.TraceMLHandler()])
    original_prepare = trainer.prepare_batch
    trainer.run()

    assert _published() == {}
    assert _pending() == 0
    assert trainer.prepare_batch is original_prepare


def test_a_replacement_made_before_the_run_is_traced():
    traceml_monai.init()
    trainer = _trainer([traceml_monai.TraceMLHandler()])
    trainer.prepare_batch = lambda *a, **k: default_prepare_batch(*a, **k)
    user_prepare = trainer.prepare_batch
    trainer.run()

    assert sorted(_published()) == [1, 2, 3, 4]
    assert trainer.prepare_batch is user_prepare


def test_a_replacement_made_during_the_run_stops_tracing_loudly(capsys):
    traceml_monai.init()
    trainer = _trainer([traceml_monai.TraceMLHandler()])

    def later(*a, **k):
        return default_prepare_batch(*a, **k)

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(once=1),
        lambda e: setattr(e, "prepare_batch", later),
    )
    trainer.run()

    # Only the first iteration ran through the handler's wrapper.
    assert sorted(_published()) == [1]
    assert _pending() == 0
    assert trainer.state.iteration == ITEMS // 2
    assert trainer.prepare_batch is later
    err = capsys.readouterr().err
    assert err.count(WARNING) == 1
    assert "prepare_batch was replaced" in err


def test_two_handlers_on_one_trainer_are_refused(capsys):
    traceml_monai.init()
    trainer = _trainer(
        [traceml_monai.TraceMLHandler(), traceml_monai.TraceMLHandler()]
    )
    assert capsys.readouterr().err.count(WARNING) == 1
    updates = []
    trainer.optimizer.register_step_post_hook(lambda *a: updates.append(1))
    trainer.run()

    steps = _published()
    assert sorted(steps) == list(range(1, len(updates) + 1))
    assert all(
        _count(ev, STEP) == _count(ev, FETCH) == 1 for ev in steps.values()
    )


def test_a_second_trainer_run_inside_a_group_does_not_lose_it(capsys):
    """Another producer may detach the process-wide capture mid-group."""
    traceml_monai.init()
    inner = _trainer(
        [traceml_monai.TraceMLHandler()], loader=DataLoader(Volumes(4), 2)
    )
    outer = _trainer([traceml_monai.TraceMLHandler()], accumulation_steps=2)
    updates = []
    for engine in (inner, outer):
        engine.optimizer.register_step_post_hook(lambda *a: updates.append(1))
    outer.add_event_handler(
        Events.ITERATION_COMPLETED(once=1), lambda e: inner.run()
    )
    outer.run()

    steps = _published()
    # One published step per optimizer update, and no hole in the numbering.
    assert sorted(steps) == list(range(1, len(updates) + 1))
    assert _pending() == 0
    assert "took the active capture" in capsys.readouterr().err


def test_an_epoch_boundary_mid_accumulation_drops_the_unstepped_group(
    capsys,
):
    """MONAI zeroes an unstepped window's gradients at the epoch boundary.

    A length-less loader learns its length only at the first epoch's end,
    so raw iteration 4 of 8 (accumulation_steps=3) opens a window MONAI
    never steps and then wipes when epoch 2 begins. That window must be
    dropped, not merged into the group that steps at iteration 7.

    Two things are proven independently of the handler's own output, so
    this cannot pass for the wrong reason: that MONAI really does zero
    iteration 4's gradients at iteration 5 without ever stepping on it
    (the premise the fix rests on), and that the real optimizer updates
    land at exactly the iterations MONAI's own rule says they do, not
    merely in the same quantity as before.
    """
    traceml_monai.init()
    trainer = _trainer(
        [traceml_monai.TraceMLHandler()],
        loader=DataLoader(Stream(), batch_size=2),
        accumulation_steps=3,
        max_epochs=2,
    )
    stepped_at, zeroed_at = [], []
    trainer.optimizer.register_step_post_hook(
        lambda *a: stepped_at.append(trainer.state.iteration)
    )
    original_zero_grad = trainer.optimizer.zero_grad

    def spy_zero_grad(*args, **kwargs):
        zeroed_at.append(trainer.state.iteration)
        return original_zero_grad(*args, **kwargs)

    trainer.optimizer.zero_grad = spy_zero_grad
    trainer.run()

    assert 5 in zeroed_at
    assert 5 not in stepped_at
    assert stepped_at == [3, 7, 8]

    steps = _published()
    assert sorted(steps) == [1, 2, 3]
    assert [_count(steps[s], STEP) for s in sorted(steps)] == [3, 3, 1]
    assert _pending() == 0
    assert len(drain_step_memory_events()) == 3
    err = capsys.readouterr().err
    assert err.count(WARNING) == 1
    assert "zeroed its gradients" in err


def test_a_stream_loader_with_evenly_dividing_accumulation_never_discards(
    capsys,
):
    """The negative control for the epoch-boundary discard.

    Every window in this run steps cleanly before the next one opens, on
    both sides of the length-less-to-known-length transition, so the
    discard's `self._capture is not None` guard must never trigger.
    """
    traceml_monai.init()
    trainer = _trainer(
        [traceml_monai.TraceMLHandler()],
        loader=DataLoader(Stream(), batch_size=2),
        accumulation_steps=2,
        max_epochs=2,
    )
    updates = []
    trainer.optimizer.register_step_post_hook(
        lambda *a: updates.append(trainer.state.iteration)
    )
    trainer.run()

    assert updates == [2, 4, 6, 8]
    steps = _published()
    assert sorted(steps) == [1, 2, 3, 4]
    assert [_count(steps[s], STEP) for s in sorted(steps)] == [2, 2, 2, 2]
    assert "unfinished group" not in capsys.readouterr().err


def test_the_fetch_clock_stops_when_the_step_budget_is_spent():
    traceml_monai.init()
    configure_trace_recording(max_steps=1)
    _trainer([traceml_monai.TraceMLHandler()]).run()

    assert sorted(_published()) == [1]
    # No clock is stamped once nothing will be published, so a CUDA run
    # stops taking events from the pool for every fetch.
    assert traceml_monai._start_fetch_clock() is None


def test_an_engine_without_accumulation_steps_is_not_traced(capsys):
    # A future MONAI could move the attribute the step boundary is read
    # from. Guessing it would publish one step per iteration under
    # accumulation, so the run is refused instead.
    traceml_monai.init()
    handler = traceml_monai.TraceMLHandler()
    engine = SimpleNamespace(
        state=SimpleNamespace(device=torch.device("cpu")),
        prepare_batch=default_prepare_batch,
    )
    handler._start_run(engine)

    err = capsys.readouterr().err
    assert handler._active is False
    assert engine.prepare_batch is default_prepare_batch
    assert err.count(WARNING) == 1
    assert "accumulation_steps" in err
