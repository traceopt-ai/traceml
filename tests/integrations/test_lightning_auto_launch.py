"""Automatic Lightning attachment through the real launcher."""

import json
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from traceml_ai.integrations import lightning as traceml_lightning
from traceml_ai.runtime import lightning_auto
from tests.integrations.launcher_utils import _run

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")


def test_auto_policy_excludes_unsupported_modes(monkeypatch, capsys):
    class SingleDeviceStrategy:
        pass

    class DDPStrategy:
        _start_method = "popen"

    class DeepSpeedStrategy(DDPStrategy):
        pass

    strategies = SimpleNamespace(
        SingleDeviceStrategy=SingleDeviceStrategy,
        DDPStrategy=DDPStrategy,
        DeepSpeedStrategy=DeepSpeedStrategy,
    )
    assert lightning_auto._supported_strategy(strategies, DDPStrategy())
    assert not lightning_auto._supported_strategy(
        strategies, DeepSpeedStrategy()
    )

    class RFDETRModule(torch.nn.Module):
        pass

    RFDETRModule.__module__ = "rfdetr.training.module_model"
    compiled = torch.nn.Linear(2, 2)
    compiled._compiler_ctx = object()
    assert lightning_auto._is_rfdetr(RFDETRModule())
    assert lightning_auto._is_compiled(compiled)

    from traceml_ai.sdk import initial

    monkeypatch.setattr(initial, "get_init_config", lambda: None)
    monkeypatch.setattr(
        traceml_lightning,
        "init",
        lambda: (_ for _ in ()).throw(AssertionError("must not initialize")),
    )
    monkeypatch.setattr(lightning_auto, "_WARNINGS", set())
    lightning_auto._prepare_callback(SimpleNamespace(_traceml_auto_skip=True))
    lightning_auto._prepare_callback(
        SimpleNamespace(callbacks=[], lightning_module=RFDETRModule())
    )
    warning = capsys.readouterr().err
    assert "traced through RFDETR.train()" in warning
    assert "constructed directly" in warning
    assert "Lightning integration's init()" not in warning
    lightning_auto._prepare_callback(
        SimpleNamespace(callbacks=[], lightning_module=compiled)
    )


_WORKLOAD = """
import json
import os
from pathlib import Path
import torch
import NAMESPACE as L

class Model(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1))
    def training_step(self, batch, batch_idx):
        return self.net(batch[0]).square().mean()
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

callbacks = []
if MANUAL:
    from traceml_ai.integrations import lightning as trace
    trace.init()
    trace.init()
    callbacks = [trace.TraceMLCallback()]

torch.manual_seed(0)
world_size = int(os.environ.get("WORLD_SIZE", "1"))
model = Model()
initial_weight = model.net[0].weight.detach().clone()
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.ones(12 * world_size, 4)), batch_size=4
)
trainer = L.Trainer(accelerator="cpu", devices=world_size, max_epochs=1, accumulate_grad_batches=2,
    callbacks=callbacks, logger=False, enable_checkpointing=False, enable_progress_bar=False,
    enable_model_summary=False, num_sanity_val_steps=0)
trainer.fit(model, train_dataloaders=loader)
from traceml_ai.integrations.lightning import TraceMLCallback
rank = int(os.environ.get("RANK", "0"))
state_path = Path(__file__).with_suffix(
    ".json" if world_size == 1 else f".rank-{rank}.json"
)
state_path.write_text(json.dumps({
    "callbacks": sum(isinstance(cb, TraceMLCallback) for cb in trainer.callbacks),
    "global_step": trainer.global_step,
    "rank": rank,
    "world_size": world_size,
    "weight_updated": not torch.equal(
        initial_weight, model.net[0].weight.detach().cpu()
    ),
}))
"""


def _script(tmp_path, namespace="lightning.pytorch", manual=False):
    script = tmp_path / "train.py"
    script.write_text(
        _WORKLOAD.replace("NAMESPACE", namespace).replace(
            "MANUAL", str(manual)
        )
    )
    return script


def _records(tmp_path, run_name="lightning-auto"):
    db = tmp_path / "logs" / run_name / "aggregator" / "telemetry"
    with sqlite3.connect(db) as conn:
        return conn.execute(
            "SELECT step, events_json FROM step_time_samples ORDER BY step, id"
        ).fetchall()


@pytest.mark.parametrize(
    ("namespace", "manual"),
    [("lightning.pytorch", False), ("pytorch_lightning", True)],
)
def test_launcher_accumulates_and_attaches_once(tmp_path, namespace, manual):
    pytest.importorskip(namespace)
    script = _script(tmp_path, namespace, manual)
    result = _run(tmp_path, script, run_name="lightning-auto")
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(script.with_suffix(".json").read_text()) == {
        "callbacks": 1,
        "global_step": 2,
        "rank": 0,
        "world_size": 1,
        "weight_updated": True,
    }
    message = (
        "using existing TraceML callback"
        if manual
        else "TraceML callback added automatically"
    )
    assert (result.stdout + result.stderr).count(message) == 1
    assert "batch_to_device is not wrapped" not in result.stderr
    records = _records(tmp_path)
    assert [step for step, _ in records] == [1, 2]
    for (_, payload), expected in zip(records, [2, 1]):
        events = json.loads(payload)
        for name in (
            "dataloader_next",
            "forward_time",
            "backward_time",
            "step_time",
        ):
            assert (
                sum(
                    value["n_calls"]
                    for value in events[f"_traceml_internal:{name}"].values()
                )
                == expected
            )
        assert (
            sum(
                value["n_calls"]
                for value in events[
                    "_traceml_internal:optimizer_step"
                ].values()
            )
            == 1
        )


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="The two-rank torchrun smoke path is not verified on Windows.",
)
def test_lightning_launcher_two_rank_auto_attachment(tmp_path):
    script = _script(tmp_path)

    result = _run(
        tmp_path,
        script,
        nproc_per_node=2,
        run_name="lightning-auto-ddp",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(2):
        state = json.loads(
            script.with_suffix(f".rank-{rank}.json").read_text()
        )
        assert state == {
            "callbacks": 1,
            "global_step": 2,
            "rank": rank,
            "world_size": 2,
            "weight_updated": True,
        }

    session_root = tmp_path / "logs" / "lightning-auto-ddp"
    summary = json.loads((session_root / "final_summary.json").read_text())
    step_time = summary["step_time"]
    assert step_time["metadata"]["global_ranks_seen"] == 2
    assert step_time["metadata"]["global_ranks_used"] == 2
    assert step_time["global"]["window"]["steps_analyzed"] == 2
    assert set(step_time["groups"]["rows"]) == {"0", "1"}
    for metric in ("forward_ms", "backward_ms", "optimizer_ms"):
        assert step_time["global"]["average"][metric] > 0
        for rank in ("0", "1"):
            assert step_time["groups"]["rows"][rank]["metrics"][metric] > 0

    manifest = json.loads((session_root / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["telemetry_status"] == "complete"


def test_launcher_resumes_with_local_step_numbers(tmp_path):
    script = _script(tmp_path)
    original = script.read_text()
    script.write_text(
        original
        + '\ntrainer.save_checkpoint(str(Path(__file__).with_suffix(".ckpt")))\n'
    )
    first = _run(tmp_path, script, run_name="checkpoint-prepare")
    assert first.returncode == 0, first.stdout + first.stderr
    script.write_text(
        original.replace("max_epochs=1", "max_epochs=2").replace(
            "train_dataloaders=loader)",
            'train_dataloaders=loader, ckpt_path=str(Path(__file__).with_suffix(".ckpt")))',
        )
    )
    resumed = _run(tmp_path, script, run_name="lightning-auto")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert (
        json.loads(script.with_suffix(".json").read_text())["global_step"] == 4
    )
    assert [step for step, _ in _records(tmp_path)] == [1, 2]


@pytest.mark.parametrize(
    "namespace", ["lightning.pytorch", "pytorch_lightning"]
)
def test_disabled_launcher_installs_nothing(tmp_path, namespace):
    script = tmp_path / "disabled.py"
    script.write_text(f"""
import json, sys, torch
from pathlib import Path
from {namespace}.trainer.connectors.callback_connector import _CallbackConnector
from traceml_ai.integrations import lightning as trace
config = trace.init()
assert config.disabled
assert not getattr(_CallbackConnector._attach_model_callbacks, "_traceml_auto_attach", False)
assert not getattr(torch.Tensor, "_traceml_h2d_patched", False)
assert not getattr(torch.utils.data.DataLoader, "_traceml_patched", False)
assert not any(type(f).__name__ == "_ImportFinder" for f in sys.meta_path)
""")
    result = _run(tmp_path, script, disabled=True, run_name="lightning-auto")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "namespace", ["lightning.pytorch", "pytorch_lightning"]
)
def test_watch_does_not_install_training_instrumentation(tmp_path, namespace):
    pytest.importorskip(namespace)
    script = tmp_path / "watch.py"
    script.write_text(f"""
import sys, torch
from {namespace}.trainer.connectors.callback_connector import _CallbackConnector
assert not getattr(_CallbackConnector._attach_model_callbacks, "_traceml_auto_attach", False)
assert not getattr(torch.Tensor, "_traceml_h2d_patched", False)
assert not getattr(torch.utils.data.DataLoader, "_traceml_patched", False)
assert not any(type(f).__name__ == "_ImportFinder" for f in sys.meta_path)
""")
    result = _run(
        tmp_path,
        script,
        command="watch",
        run_name="lightning-watch",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_unrelated_launcher_does_not_import_training_frameworks(tmp_path):
    script = tmp_path / "plain.py"
    script.write_text("""
import sys
assert not any(name == root or name.startswith(root + ".")
               for name in sys.modules for root in ("lightning", "pytorch_lightning", "transformers", "rfdetr"))
""")
    result = _run(tmp_path, script, run_name="lightning-auto")
    assert result.returncode == 0, result.stdout + result.stderr
