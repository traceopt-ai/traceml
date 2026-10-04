"""Launcher boundaries for automatic Hugging Face Trainer attachment."""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("accelerate")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def _port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _run(
    tmp_path: Path,
    script: Path,
    *,
    disabled: bool = False,
    profile: str = "run",
):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(SRC), env.get("PYTHONPATH", "")) if part
    )
    command = [
        sys.executable,
        "-m",
        "traceml_ai.launcher.cli",
        profile,
        "--mode",
        "summary",
        "--logs-dir",
        str(tmp_path / "logs"),
        "--run-name",
        "hf-auto",
        "--aggregator-port",
        _port(),
        "--master-port",
        _port(),
        "--finalize-timeout-sec",
        "60",
    ]
    if disabled:
        command.append("--disable-traceml")
    command.append(str(script))
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


_WORKLOAD = """
import json
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

if MANUAL:
    from traceml_ai.integrations import huggingface as traceml_hf
    traceml_hf.init()
    callbacks = [traceml_hf.TraceMLTrainerCallback()]
else:
    callbacks = []

trainer = Trainer(
    model=Model(),
    args=TrainingArguments(
        output_dir=str(Path(__file__).parent / "output"), max_steps=2,
        per_device_train_batch_size=2, report_to=[], logging_strategy="no",
        save_strategy="no", disable_tqdm=True,
    ),
    train_dataset=Dataset(), callbacks=callbacks,
)
trainer.train()
from traceml_ai.integrations.huggingface import TraceMLTrainerCallback
Path(__file__).with_suffix(".json").write_text(json.dumps({
    "callbacks": sum(isinstance(cb, TraceMLTrainerCallback)
                     for cb in trainer.callback_handler.callbacks),
    "transformers_loaded": "transformers.trainer" in sys.modules,
    "patched": bool(getattr(Trainer._inner_training_loop,
                            "_traceml_lifecycle_guard", False)),
    "hook": any(type(f).__name__ == "_TrainerFinder" for f in sys.meta_path),
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
        '"hook": any(type(f).__name__ == "_TrainerFinder" '
        "for f in sys.meta_path)}))\n",
        encoding="utf-8",
    )

    result = _run(tmp_path, script, profile="watch")

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
        '"hook": any(type(f).__name__ == "_TrainerFinder" '
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
