"""Validate example launch topology and checkpoint safety without RF-DETR."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/integrations/rfdetr_minimal.py"
)
spec = importlib.util.spec_from_file_location("rfdetr_example", EXAMPLE)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


@pytest.mark.parametrize(
    "env,expected",
    [
        ({}, (1, 1, 0)),
        ({"WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2", "RANK": "1"}, (2, 1, 1)),
        (
            {
                "WORLD_SIZE": "8",
                "LOCAL_WORLD_SIZE": "4",
                "TRACEML_NNODES": "2",
                "RANK": "5",
            },
            (4, 2, 5),
        ),
        ({"WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "2"}, (2, 2, 0)),
    ],
)
def test_topology_follows_torchrun(env, expected):
    assert example.launch_topology(env) == expected


@pytest.mark.parametrize(
    "env",
    [
        {"WORLD_SIZE": "3", "LOCAL_WORLD_SIZE": "2"},
        {"LOCAL_WORLD_SIZE": "0"},
        {"TRACEML_NNODES": "2"},
        {"RANK": "1"},
        {"WORLD_SIZE": "invalid"},
    ],
)
def test_inconsistent_topology_is_rejected(env):
    with pytest.raises(ValueError):
        example.launch_topology(env)


@pytest.fixture
def example_run(tmp_path, monkeypatch):
    for name in ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "TRACEML_NNODES", "RANK"):
        monkeypatch.delenv(name, raising=False)
    for split in ("train", "valid", "test"):
        folder = tmp_path / "dataset" / split
        folder.mkdir(parents=True)
        (folder / "_annotations.coco.json").write_text("{}")

    calls = []

    class FakeNano:
        def __init__(self, **kwargs):
            calls.append(("model", kwargs))

        def train(self, **kwargs):
            calls.append(("train", kwargs))

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.manual_seed = lambda seed: calls.append(("seed", seed))
    rfdetr = ModuleType("rfdetr")
    rfdetr.RFDETRNano = FakeNano
    integration = ModuleType("traceml_ai.integrations.rfdetr")
    integration.init = lambda: calls.append(("init", None))
    integrations = ModuleType("traceml_ai.integrations")
    integrations.rfdetr = integration
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "rfdetr", rfdetr)
    monkeypatch.setitem(sys.modules, "traceml_ai.integrations", integrations)
    monkeypatch.setitem(
        sys.modules, "traceml_ai.integrations.rfdetr", integration
    )
    args = [
        "--dataset-dir",
        str(tmp_path / "dataset"),
        "--output-dir",
        str(tmp_path / "checkpoints"),
    ]
    return args, calls, tmp_path / "checkpoints"


def test_default_training_keeps_fixed_comparison_settings(example_run):
    args, calls, output = example_run
    example.main(args)
    assert output.is_dir()
    assert [name for name, _ in calls] == ["seed", "init", "model", "train"]
    model = dict(calls)["model"]
    training = dict(calls)["train"]
    assert model == {"device": "cpu", "resolution": 384, "compile": False}
    assert training["seed"] == 42
    assert training["multi_scale"] is False
    assert training["expanded_scales"] is False
    assert training["grad_accum_steps"] == 1
    assert training["devices"] == training["num_nodes"] == 1
    assert training["strategy"] == "auto"


def test_existing_checkpoint_directory_is_never_overwritten(example_run):
    args, calls, output = example_run
    output.mkdir()
    sentinel = output / "last.ckpt"
    sentinel.write_bytes(b"keep this checkpoint")
    with pytest.raises(SystemExit, match="2"):
        example.main(args)
    assert sentinel.read_bytes() == b"keep this checkpoint"
    assert not calls


def test_nonzero_rank_accepts_directory_created_by_rank_zero(
    example_run, monkeypatch
):
    args, calls, output = example_run
    output.mkdir()
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("TRACEML_NNODES", "2")
    monkeypatch.setenv("RANK", "3")
    example.main(args + ["--epochs", "3", "--num-workers", "2"])
    training = dict(calls)["train"]
    assert training["devices"] == training["num_nodes"] == 2
    assert training["strategy"] == "ddp"
    assert training["epochs"] == 3
    assert training["num_workers"] == 2
    assert ("init", None) in calls


def test_cuda_request_without_cuda_fails_before_creating_output(example_run):
    args, calls, output = example_run
    with pytest.raises(SystemExit, match="2"):
        example.main(args + ["--accelerator", "cuda"])
    assert not output.exists()
    assert not calls
