"""Launcher boundaries for automatic Hugging Face Trainer attachment."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("accelerate")

from tests.integrations.launcher_utils import _run

_WORKLOAD = """
import json
import os
import sys
from pathlib import Path
import torch
from transformers import Trainer, TrainingArguments

class Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return 8
    def __getitem__(self, index):
        return {"input_ids": torch.ones(4), "labels": torch.tensor(index % 2)}

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
    def forward(self, input_ids, labels):
        logits = self.linear(input_ids)
        return {"loss": torch.nn.functional.cross_entropy(logits, labels),
                "logits": logits}

torch.manual_seed(0)
model = Model()
initial_weight = model.linear.weight.detach().clone()

if MANUAL:
    from traceml_ai.integrations import huggingface as traceml_hf
    traceml_hf.init()
    callbacks = [traceml_hf.TraceMLTrainerCallback()]
else:
    callbacks = []

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=str(Path(__file__).parent / "output"), max_steps=2,
        per_device_train_batch_size=2, report_to=[], logging_strategy="no",
        save_strategy="no", disable_tqdm=True, use_cpu=True,
    ),
    train_dataset=Dataset(), callbacks=callbacks,
)
trainer.train()
from traceml_ai.integrations.huggingface import TraceMLTrainerCallback
rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
state_path = Path(__file__).with_suffix(
    ".json" if world_size == 1 else f".rank-{rank}.json"
)
state_path.write_text(json.dumps({
    "callbacks": sum(isinstance(cb, TraceMLTrainerCallback)
                     for cb in trainer.callback_handler.callbacks),
    "global_step": trainer.state.global_step,
    "rank": rank,
    "world_size": world_size,
    "weight_updated": not torch.equal(
        initial_weight, model.linear.weight.detach().cpu()
    ),
    "transformers_loaded": "transformers.trainer" in sys.modules,
    "patched": bool(getattr(Trainer._inner_training_loop,
                            "_traceml_lifecycle_guard", False)),
    "hook": any("transformers.trainer" in getattr(f, "targets", ())
                for f in sys.meta_path),
}))
"""


@pytest.mark.parametrize("manual", [False, True])
def test_hf_launcher_attaches_once_and_publishes_steps(
    tmp_path: Path,
    manual: bool,
) -> None:
    script = tmp_path / "train.py"
    script.write_text(
        _WORKLOAD.replace("MANUAL", str(manual)), encoding="utf-8"
    )
    result = _run(tmp_path, script)
    assert result.returncode == 0, result.stdout + result.stderr
    state = json.loads(script.with_suffix(".json").read_text())
    assert state["callbacks"] == 1
    assert state["transformers_loaded"]
    assert state["patched"]
    assert not state["hook"]
    expected = (
        "using existing TraceML callback"
        if manual
        else "TraceML callback added automatically"
    )
    assert (result.stdout + result.stderr).count(expected) == 1

    db = tmp_path / "logs" / "hf-auto" / "aggregator" / "telemetry"
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT step, events_json FROM step_time_samples ORDER BY step, id"
        ).fetchall()
    steps = [row[0] for row in rows]
    assert steps == [1, 2]
    names = set().union(*(json.loads(row[1]) for row in rows))
    assert {
        "_traceml_internal:step_time",
        "_traceml_internal:forward_time",
        "_traceml_internal:backward_time",
        "_traceml_internal:optimizer_step",
        "_traceml_internal:dataloader_next",
    } <= names


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="The two-rank torchrun smoke path is not verified on Windows.",
)
def test_hf_launcher_two_rank_auto_attachment(tmp_path: Path) -> None:
    script = tmp_path / "train.py"
    script.write_text(_WORKLOAD.replace("MANUAL", "False"), encoding="utf-8")

    result = _run(
        tmp_path,
        script,
        nproc_per_node=2,
        run_name="hf-auto-ddp",
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
            "transformers_loaded": True,
            "patched": True,
            "hook": False,
        }

    session_root = tmp_path / "logs" / "hf-auto-ddp"
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


def test_non_hf_script_does_not_import_transformers(tmp_path: Path) -> None:
    script = tmp_path / "plain.py"
    script.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        'Path(__file__).with_suffix(".json").write_text('
        'json.dumps("transformers" in sys.modules))\n',
        encoding="utf-8",
    )
    result = _run(tmp_path, script)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(script.with_suffix(".json").read_text()) is False


def test_find_spec_probe_does_not_consume_hf_hook(tmp_path: Path) -> None:
    script = tmp_path / "probed_train.py"
    workload = _WORKLOAD.replace("MANUAL", "False").replace(
        "from transformers import Trainer, TrainingArguments",
        "import importlib.util\n"
        'assert importlib.util.find_spec("transformers.trainer") is not None\n'
        "from transformers import Trainer, TrainingArguments",
    )
    script.write_text(workload, encoding="utf-8")

    result = _run(tmp_path, script)

    assert result.returncode == 0, result.stdout + result.stderr
    state = json.loads(script.with_suffix(".json").read_text())
    assert state["callbacks"] == 1
    assert state["patched"]
    assert not state["hook"]


def test_watch_does_not_install_hf_hook(tmp_path: Path) -> None:
    script = tmp_path / "watch.py"
    script.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "from transformers import Trainer\n"
        'Path(__file__).with_suffix(".json").write_text(json.dumps({'
        '"patched": bool(getattr(Trainer._inner_training_loop, '
        '"_traceml_lifecycle_guard", False)), '
        '"hook": any(type(f).__name__ == "_ImportFinder" '
        "for f in sys.meta_path)}))\n",
        encoding="utf-8",
    )

    result = _run(tmp_path, script, command="watch")

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(script.with_suffix(".json").read_text()) == {
        "patched": False,
        "hook": False,
    }


def test_incompatible_explicit_config_is_kept_and_warned_once(
    tmp_path: Path,
) -> None:
    script = tmp_path / "manual_config.py"
    workload = _WORKLOAD.replace("MANUAL", "False").replace(
        "from transformers import Trainer, TrainingArguments",
        "import traceml_ai as traceml\n"
        'traceml.init(mode="manual")\n'
        "from transformers import Trainer, TrainingArguments",
    )
    script.write_text(workload, encoding="utf-8")
    result = _run(tmp_path, script)
    assert result.returncode == 0, result.stdout + result.stderr
    state = json.loads(script.with_suffix(".json").read_text())
    assert state["callbacks"] == 0
    assert state["patched"]
    assert (result.stdout + result.stderr).count(
        "auto-instrumentation found an incompatible"
    ) == 1


def test_disabled_launcher_does_not_patch_trainer(tmp_path: Path) -> None:
    script = tmp_path / "disabled.py"
    script.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "from transformers import Trainer\n"
        "from traceml_ai.integrations import huggingface as hf\n"
        "config = hf.init()\n"
        'Path(__file__).with_suffix(".json").write_text(json.dumps({'
        '"disabled": config.disabled, '
        '"patched": bool(getattr(Trainer._inner_training_loop, '
        '"_traceml_lifecycle_guard", False)), '
        '"hook": any(type(f).__name__ == "_ImportFinder" '
        "for f in sys.meta_path)}))\n",
        encoding="utf-8",
    )
    result = _run(tmp_path, script, disabled=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(script.with_suffix(".json").read_text()) == {
        "disabled": True,
        "patched": False,
        "hook": False,
    }
