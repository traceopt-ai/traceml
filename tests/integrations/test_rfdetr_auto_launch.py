"""RF-DETR automatic/manual launcher coverage using the offline fixture."""

import json
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("rfdetr")

from tests.integrations.launcher_utils import _run

_WORKLOAD = """
import json, sys, torch
from pathlib import Path
from rfdetr.training import build_trainer
from rfdetr_fixture import components

if MANUAL:
    from traceml_ai.integrations import rfdetr as tracing
    tracing.init()
    tracing.init()

torch.set_num_threads(1)
with components(Path(__file__).parent / "checkpoints") as (module, data):
    trainer = build_trainer(
        module.train_config, module.model_config, devices=1,
        limit_train_batches=3, limit_val_batches=0, num_sanity_val_steps=0,
        logger=False, enable_model_summary=False,
    )
    trainer.fit(module, datamodule=data)
from traceml_ai.integrations.rfdetr import _callback_class
Path(__file__).with_suffix(".json").write_text(json.dumps({
    "callbacks": sum(isinstance(cb, _callback_class()) for cb in trainer.callbacks),
    "global_step": trainer.global_step,
    "hook": any("rfdetr.training" in getattr(f, "targets", ()) for f in sys.meta_path),
}))
"""


@pytest.mark.parametrize("manual", [False, True])
def test_launcher_attaches_once_and_accumulates(tmp_path, manual):
    fixture = Path(__file__).parent / "fixtures/rfdetr_training.py"
    (tmp_path / "rfdetr_fixture.py").write_text(fixture.read_text())
    script = tmp_path / "train.py"
    script.write_text(_WORKLOAD.replace("MANUAL", str(manual)))
    result = _run(tmp_path, script, run_name="rfdetr-auto")
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(script.with_suffix(".json").read_text()) == {
        "callbacks": 1,
        "global_step": 2,
        "hook": False,
    }
    output = result.stdout + result.stderr
    assert (
        output.count(
            "RF-DETR Trainer detected; " "TraceML callback added automatically"
        )
        == 1
    )
    assert "PyTorch Lightning Trainer detected" not in output
    with sqlite3.connect(
        tmp_path / "logs/rfdetr-auto/aggregator/telemetry"
    ) as conn:
        rows = conn.execute(
            "SELECT step, events_json FROM step_time_samples ORDER BY step, id"
        ).fetchall()
    assert [step for step, _ in rows] == [1, 2]
    for (_, payload), expected in zip(rows, [2, 1]):
        events = json.loads(payload)
        for phase in (
            "step_time",
            "dataloader_next",
            "forward_time",
            "backward_time",
            "optimizer_step",
        ):
            assert sum(
                event["n_calls"]
                for event in events[f"_traceml_internal:{phase}"].values()
            ) == (1 if phase == "optimizer_step" else expected)


@pytest.mark.parametrize("launch", ["disabled", "watch"])
def test_resource_only_launcher_leaves_factory_unpatched(tmp_path, launch):
    script = tmp_path / "unpatched.py"
    script.write_text("""
import sys, torch
from rfdetr.training import build_trainer
assert not getattr(build_trainer, "_traceml_rfdetr_factory", False)
assert not getattr(torch.Tensor, "_traceml_h2d_patched", False)
assert not getattr(torch.utils.data.DataLoader, "_traceml_patched", False)
assert not any(type(f).__name__ == "_ImportFinder" for f in sys.meta_path)
""")
    result = _run(
        tmp_path,
        script,
        disabled=launch == "disabled",
        command="watch" if launch == "watch" else "run",
        run_name="rfdetr-unpatched",
    )
    assert result.returncode == 0, result.stdout + result.stderr
