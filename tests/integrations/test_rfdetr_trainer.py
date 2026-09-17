"""RF-DETR correctness and offline end-to-end integration checks."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rfdetr")
pl = pytest.importorskip("pytorch_lightning")

from rfdetr.training.callbacks.ema import RFDETREMACallback  # noqa: E402

from traceml_ai.instrumentation.hooks.optimizer_hooks import (  # noqa: E402
    reset_optimizer_timing,
)
from traceml_ai.instrumentation.step_events import (  # noqa: E402
    abort_step_capture,
    begin_step_capture,
    drain_step_memory_events,
    drain_step_time_batches,
)
from traceml_ai.integrations import rfdetr as tracing  # noqa: E402
from traceml_ai.runtime.state import (  # noqa: E402
    configure_trace_recording,
    reset_trace_session_state,
)

FIXTURE = Path(__file__).parent / "fixtures" / "rfdetr_training.py"
_spec = importlib.util.spec_from_file_location("rfdetr_training", FIXTURE)
fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixture)
PREFIX = "_traceml_internal:"


@pytest.fixture(autouse=True)
def _isolation(monkeypatch):
    import rfdetr.training

    original_builder = rfdetr.training.build_trainer
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    reset_optimizer_timing()
    reset_trace_session_state()
    configure_trace_recording(max_steps=None)
    drain_step_time_batches()
    drain_step_memory_events()
    abort_step_capture(begin_step_capture())
    yield
    rfdetr.training.build_trainer = original_builder
    abort_step_capture(begin_step_capture())
    drain_step_time_batches()
    drain_step_memory_events()
    reset_optimizer_timing()
    torch.set_num_threads(threads)


def counts(batches, phase):
    return [
        sum(event.name == PREFIX + phase for event in batch.events)
        for batch in batches
    ]


@pytest.mark.parametrize(
    "accumulation,expected", [(1, [1, 1, 1]), (2, [2, 1])]
)
def test_rfdetr_accumulation_and_eval_isolation(
    tmp_path, accumulation, expected
):
    tracing.init()
    callback = tracing._callback_class()()
    with fixture.components(tmp_path, accumulation=accumulation) as (
        module,
        data,
    ):
        # Public RF-DETR can consume grid-preview batches before Trainer setup.
        data.setup("fit")
        next(iter(data.train_dataloader()))
        fit = fixture.trainer(
            [RFDETREMACallback(), callback],
            accumulation=accumulation,
            limit_train_batches=3,
            val_check_interval=1,
        )
        original_transfer = fit.strategy.batch_to_device
        fit.fit(module, datamodule=data)

        batches = drain_step_time_batches()
        assert [batch.step for batch in batches] == list(
            range(1, len(expected) + 1)
        )
        for phase in ("dataloader_next", "forward_time", "backward_time"):
            assert counts(batches, phase) == expected
        assert counts(batches, "optimizer_step") == [1] * len(expected)
        assert fit.global_step == len(expected)
        assert len(drain_step_memory_events()) == len(expected)
        assert "forward" not in module.model.__dict__
        assert fit.strategy.batch_to_device == original_transfer
        assert begin_step_capture().timing_events == []


def test_rfdetr_tracing_preserves_weights_optimizer_and_ema(tmp_path):
    def run(enabled):
        torch.manual_seed(13)
        reset_trace_session_state()
        ema = RFDETREMACallback()
        callbacks = [ema]
        if enabled:
            tracing.init()
            callbacks.append(tracing._callback_class()())
        with fixture.components(tmp_path) as (module, data):
            fit = fixture.trainer(callbacks, limit_train_batches=4)
            fit.fit(module, datamodule=data)
            return (
                deepcopy(module.state_dict()),
                deepcopy(fit.optimizers[0].state_dict()),
                deepcopy(ema._average_model.state_dict()),
                fit.callback_metrics["train/loss"].clone(),
            )

    baseline = run(False)
    traced = run(True)
    torch.testing.assert_close(traced, baseline, rtol=0, atol=0)
    assert len(drain_step_time_batches()) == 2


def test_rfdetr_ema_forward_uses_its_own_weights(tmp_path):
    tracing.init()
    ema = RFDETREMACallback()

    class CheckEMA(pl.Callback):
        def on_train_start(self, trainer, module):
            averaged = ema._average_model.module.model
            assert "forward" not in averaged.__dict__
            data = type("Batch", (), {"tensors": torch.ones(1, 3, 32, 32)})()
            live_before = module.model(data)["scores"].detach().clone()
            with torch.no_grad():
                averaged.projection.bias.add_(10)
            live_after = module.model(data)["scores"].detach()
            torch.testing.assert_close(live_after, live_before)
            assert not torch.allclose(averaged(data)["scores"], live_after)

    with fixture.components(tmp_path) as (module, data):
        fixture.trainer(
            [ema, tracing._callback_class()(), CheckEMA()],
            limit_train_batches=2,
        ).fit(module, datamodule=data)


def test_rfdetr_failure_discards_group_and_restores(tmp_path):
    tracing.init()
    callback = tracing._callback_class()()

    class FailSecondBatch(pl.Callback):
        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            if batch_idx == 1:
                raise RuntimeError("rfdetr test failure")

    with fixture.components(tmp_path) as (module, data):
        fit = fixture.trainer([callback, FailSecondBatch()])
        original_transfer = fit.strategy.batch_to_device
        with pytest.raises(RuntimeError, match="rfdetr test failure"):
            fit.fit(module, datamodule=data)
        assert drain_step_time_batches() == []
        assert drain_step_memory_events() == []
        assert begin_step_capture().timing_events == []
        assert "forward" not in module.model.__dict__
        assert fit.strategy.batch_to_device == original_transfer


def test_rfdetr_callback_reuse_does_not_leak_captures(tmp_path):
    tracing.init()
    callback = tracing._callback_class()()
    with fixture.components(tmp_path) as (module, data):
        for _ in range(2):
            fit = fixture.trainer(
                [RFDETREMACallback(), callback], limit_train_batches=2
            )
            fit.fit(module, datamodule=data)
            assert "forward" not in module.model.__dict__
            assert begin_step_capture().timing_events == []
    batches = drain_step_time_batches()
    assert [batch.step for batch in batches] == [1, 2]
    assert counts(batches, "forward_time") == [2, 2]
    assert counts(batches, "dataloader_next") == [2, 2]
    assert counts(batches, "optimizer_step") == [1, 1]


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def test_rfdetr_two_cpu_ranks_finalize_and_compare(tmp_path):
    if not torch.distributed.is_gloo_available():
        pytest.fail("The RF-DETR integration job requires CPU/Gloo support")
    env = dict(os.environ)
    env.pop("TRACEML_DISABLED", None)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            [
                str(Path(__file__).resolve().parents[2] / "src"),
                env.get("PYTHONPATH"),
            ],
        )
    )
    env["OMP_NUM_THREADS"] = "1"
    # Keep cold bytecode imports outside the launcher's readiness budget.
    warmup = subprocess.run(
        [sys.executable, "-c", "import traceml_ai.aggregator.aggregator_main"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert warmup.returncode == 0, warmup.stdout + warmup.stderr
    command = [sys.executable, "-m", "traceml_ai.launcher.cli"]
    for name in ("baseline", "candidate"):
        result = subprocess.run(
            command
            + [
                "run",
                str(FIXTURE),
                "--mode",
                "summary",
                "--nproc-per-node",
                "2",
                "--master-port",
                _free_port(),
                "--aggregator-port",
                _free_port(),
                "--run-name",
                name,
                "--logs-dir",
                str(tmp_path / "logs"),
                "--finalize-timeout-sec",
                "30",
                "--args",
                "--output",
                str(tmp_path / name),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        for rank in range(2):
            report = json.loads(
                (tmp_path / name / f"rank-{rank}.json").read_text()
            )
            assert report == {"rank": rank, "steps": 4}
        summary = tmp_path / "logs" / name / "final_summary.json"
        payload = json.loads(summary.read_text())
        step_time = payload["step_time"]
        assert step_time["metadata"]["training_total_steps"] == 4
        assert step_time["metadata"]["training_latest_step"] == 4
        assert payload["step_memory"]["metadata"]["training_total_steps"] == 4
        assert step_time["metadata"]["global_ranks_seen"] == 2
        assert step_time["metadata"]["global_ranks_used"] == 2
        assert step_time["global"]["window"]["steps_analyzed"] == 4
        assert set(step_time["groups"]["rows"]) == {"0", "1"}
        for metric in ("forward_ms", "backward_ms", "optimizer_ms"):
            assert step_time["global"]["average"][metric] > 0
            for rank in ("0", "1"):
                assert step_time["groups"]["rows"][rank]["metrics"][metric] > 0
        manifest = json.loads((summary.parent / "manifest.json").read_text())
        assert manifest["status"] == "completed"
        assert manifest["telemetry_status"] == "complete"

    result = subprocess.run(
        command
        + [
            "compare",
            str(tmp_path / "logs/baseline/final_summary.json"),
            str(tmp_path / "logs/candidate/final_summary.json"),
            "--output",
            str(tmp_path / "comparison"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((tmp_path / "comparison.json").read_text())
