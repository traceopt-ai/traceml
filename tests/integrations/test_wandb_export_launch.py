"""Live W&B export through the real launcher, against a stand-in wandb.

A fake ``wandb`` package on ``PYTHONPATH`` appends one JSON line per call
to a log file, so the aggregator process runs the real exporter and the
real export worker without any network or credentials.
"""

import json
import os
import sys
from pathlib import Path

import pytest

from tests.integrations.launcher_utils import SRC, _run

pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="The launcher smoke path is not verified on Windows.",
)

_STEPS = 4
_RUN_NAME = "wandb-launch"

_FAKE_WANDB = '''
"""Stand-in wandb that records every call as one JSON line."""

import json
import os
import threading
import types

__version__ = "0.31.0"
_LOCK = threading.Lock()


def _record(call, **fields):
    with _LOCK:
        with open(os.environ["FAKE_WANDB_LOG"], "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"call": call, **fields}) + "\\n")


class Settings:
    def __init__(self, **kwargs):
        self.__dict__["fields"] = dict(kwargs)

    def __setattr__(self, name, value):
        self.fields[name] = value

    def __getattr__(self, name):
        try:
            return self.__dict__["fields"][name]
        except KeyError:
            raise AttributeError(name) from None


class _Summary(dict):
    def __init__(self, run_id):
        super().__init__()
        self._run_id = run_id

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        _record("summary", id=self._run_id, key=key, value=value)


class _Run:
    def __init__(self, kwargs):
        self.id = kwargs["id"]
        self.name = kwargs["name"]
        self.project_url = "https://wandb.test/team/proj"
        self.url = self.project_url + "/runs/" + self.id
        self.summary = _Summary(self.id)

    def define_metric(self, name, **kwargs):
        _record("define_metric", id=self.id, name=name, kwargs=kwargs)

    def log(self, data, **kwargs):
        _record("log", id=self.id, data=data, kwargs=kwargs)

    def finish(self, exit_code=None):
        _record("finish", id=self.id, exit_code=exit_code)


class UsageError(Exception):
    pass


errors = types.SimpleNamespace(UsageError=UsageError)
sdk = types.SimpleNamespace(wandb_setup=types.SimpleNamespace(_singleton=None))
util = types.SimpleNamespace(generate_id=lambda length=8: "fixedid0")


def setup(settings=None):
    _record("setup", fields=sorted(settings.fields))
    return types.SimpleNamespace(settings=types.SimpleNamespace(mode="online"))


def login(key=None, relogin=None, prompt=True, verify=True):
    _record("login", prompt=prompt, verify=verify, key=key)
    return True


def init(**kwargs):
    if os.environ.get("FAKE_WANDB_INIT_FAIL") == "1":
        _record("init_failed", name=kwargs["name"])
        raise RuntimeError("Failed to get resume status (fake)")
    _record(
        "init",
        **{key: value for key, value in kwargs.items() if key != "settings"},
    )
    return _Run(kwargs)
'''

_WORKLOAD = """
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import traceml_ai as traceml

Path(os.environ["TRAINING_MARKER"]).write_text("started")
STEPS = NUM_STEPS
torch.manual_seed(0)
traceml.init(mode="auto")
model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loader = DataLoader(
    TensorDataset(torch.randn(STEPS * 4, 8), torch.randn(STEPS * 4, 1)),
    batch_size=4,
)
for x, y in loader:
    with traceml.trace_step(model):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
"""


@pytest.fixture
def fake_wandb(tmp_path, monkeypatch):
    """Write the stand-in package and clear inherited W&B settings."""
    for name in list(os.environ):
        if name.startswith("WANDB_"):
            monkeypatch.delenv(name)
    package = tmp_path / "fake_site" / "wandb"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(_FAKE_WANDB, encoding="utf-8")
    return tmp_path / "fake_site"


def _launch(tmp_path, fake_site, *, mode, nproc=1, fail=False):
    script = tmp_path / "train.py"
    script.write_text(_WORKLOAD.replace("NUM_STEPS", str(_STEPS)))
    env = {
        "TRACEML_EXPORTER": "wandb",
        "PYTHONPATH": os.pathsep.join([str(fake_site), str(SRC)]),
        "FAKE_WANDB_LOG": str(tmp_path / "wandb_calls.jsonl"),
        "TRAINING_MARKER": str(tmp_path / "training_started"),
        "WANDB_ENTITY": "team",
        "WANDB_PROJECT": "proj",
        "WANDB_API_KEY": "x" * 40,
    }
    if fail:
        env["FAKE_WANDB_INIT_FAIL"] = "1"
    if mode == "summary":
        env["TRACEML_HISTORY_ENABLED"] = "1"
    return _run(
        tmp_path,
        script,
        run_name=_RUN_NAME,
        nproc_per_node=nproc,
        mode=mode,
        extra_env=env,
    )


def _calls(tmp_path):
    path = tmp_path / "wandb_calls.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _all_output(tmp_path: Path, result) -> str:
    logs = [
        path.read_text(errors="replace")
        for path in (tmp_path / "logs").rglob("*.log")
    ]
    return "\n".join([result.stdout, result.stderr, *logs])


def test_telemetry_startup_failure_does_not_start_training(
    tmp_path, fake_wandb
):
    result = _launch(tmp_path, fake_wandb, mode="telemetry", fail=True)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "training was not started" in result.stderr
    assert not (tmp_path / "training_started").exists()
    calls = _calls(tmp_path)
    assert [call["call"] for call in calls][:3] == [
        "setup",
        "login",
        "init_failed",
    ]
    assert "wandb.init() failed for team/proj" in _all_output(tmp_path, result)


def test_summary_startup_failure_lets_training_finish(tmp_path, fake_wandb):
    result = _launch(tmp_path, fake_wandb, mode="summary", fail=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "training_started").exists()
    session_root = tmp_path / "logs" / _RUN_NAME
    warning = json.loads(
        (session_root / "aggregator" / "finalization_warning.json").read_text()
    )
    assert warning["export"]["exporter_failed"] is True
    assert warning["export"]["written_records"] == 0


def test_two_rank_telemetry_run_logs_into_two_grouped_runs(
    tmp_path, fake_wandb
):
    result = _launch(tmp_path, fake_wandb, mode="telemetry", nproc=2)

    assert result.returncode == 0, result.stdout + result.stderr
    calls = _calls(tmp_path)

    (login,) = [call for call in calls if call["call"] == "login"]
    assert login["prompt"] is False
    assert login["key"] is None
    inits = [call for call in calls if call["call"] == "init"]
    assert sorted(call["name"] for call in inits) == [
        f"{_RUN_NAME} rank 0",
        f"{_RUN_NAME} rank 1",
    ]
    for call in inits:
        assert call["group"] == _RUN_NAME
        assert call["job_type"] == "traceml"
        assert call["mode"] == "online"
        assert call["id"] != "fixedid0"
    rank_of = {call["id"]: int(call["name"][-1]) for call in inits}

    for run_id in rank_of:
        defined = [
            call["name"]
            for call in calls
            if call["call"] == "define_metric" and call["id"] == run_id
        ]
        assert defined == ["traceml/step", "traceml/*"]

    steps = {0: [], 1: []}
    for call in calls:
        if call["call"] != "log":
            continue
        assert call["kwargs"] == {"commit": True}
        data = call["data"]
        assert "traceml/forward_cpu_ms" in data
        assert not any("rank" in key for key in data)
        steps[rank_of[call["id"]]].append(data["traceml/step"])
    first = min(steps[0])
    for rank_steps in steps.values():
        assert sorted(set(rank_steps)) == list(range(first, first + _STEPS))

    finishes = [call for call in calls if call["call"] == "finish"]
    assert sorted(call["id"] for call in finishes) == sorted(rank_of)
    assert all(call["exit_code"] == 0 for call in finishes)
    written = {
        call["id"]: call["value"]
        for call in calls
        if call["call"] == "summary"
        and call["key"] == "traceml/export_written_records"
    }
    assert written == {
        run_id: len(steps[rank]) for run_id, rank in rank_of.items()
    }

    # The aggregator prints the line once; the launcher mirrors it.
    group_lines = [
        line
        for line in result.stderr.splitlines()
        if "[TraceML] W&B group:" in line
    ]
    assert len(group_lines) == 1
    assert group_lines[0].endswith(
        f"[TraceML] W&B group: https://wandb.test/team/proj/groups/{_RUN_NAME}"
    )
    assert "x" * 40 not in _all_output(tmp_path, result)
    session_root = tmp_path / "logs" / _RUN_NAME
    assert not list((session_root / "aggregator").glob("finalization_*.json"))
    assert list((session_root / "aggregator" / "export").iterdir()) == []
