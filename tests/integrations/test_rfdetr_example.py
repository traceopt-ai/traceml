"""Validate example launch topology and checkpoint safety without RF-DETR."""

from __future__ import annotations

import importlib.util
import json
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
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "rfdetr", rfdetr)
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
    assert [name for name, _ in calls] == ["seed", "model", "train"]
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


def test_cuda_request_without_cuda_fails_before_creating_output(example_run):
    args, calls, output = example_run
    with pytest.raises(SystemExit, match="2"):
        example.main(args + ["--accelerator", "cuda"])
    assert not output.exists()
    assert not calls


@pytest.mark.parametrize(
    "data_args", [[], ["--demo", "--dataset-dir", "data"]]
)
def test_requires_one_data_source(data_args):
    with pytest.raises(SystemExit, match="2"):
        example.build_parser().parse_args(
            ["--output-dir", "checkpoints/demo", *data_args]
        )


def test_demo_annotations_match_generated_images(tmp_path):
    from PIL import Image

    example.write_demo_dataset(tmp_path)
    for split, count in (("train", 32), ("valid", 4), ("test", 4)):
        folder = tmp_path / split
        payload = json.loads((folder / "_annotations.coco.json").read_text())
        assert len(payload["images"]) == len(payload["annotations"]) == count
        images = {row["id"]: row for row in payload["images"]}
        categories = {row["id"] for row in payload["categories"]}
        for annotation in payload["annotations"]:
            row = images[annotation["image_id"]]
            with Image.open(folder / row["file_name"]) as image:
                assert image.size == (row["width"], row["height"])
            x, y, width, height = annotation["bbox"]
            assert 0 <= x < x + width <= row["width"]
            assert 0 <= y < y + height <= row["height"]
            assert annotation["area"] == width * height
            assert annotation["category_id"] in categories


@pytest.mark.parametrize("fail", [False, True])
def test_demo_data_lives_through_training_and_is_cleaned(
    example_run, monkeypatch, fail
):
    args, calls, output = example_run
    datasets = []

    def train(self, **kwargs):
        dataset = Path(kwargs["dataset_dir"])
        datasets.append(dataset)
        assert (dataset / "train/_annotations.coco.json").is_file()
        assert len(list((dataset / "train").glob("*.jpg"))) == 32
        if fail:
            raise RuntimeError("training failed")

    monkeypatch.setattr(sys.modules["rfdetr"].RFDETRNano, "train", train)
    demo_args = ["--demo", *args[2:], "--epochs", "1"]
    if fail:
        with pytest.raises(RuntimeError, match="training failed"):
            example.main(demo_args)
    else:
        example.main(demo_args)
    assert len(datasets) == 1
    assert not datasets[0].exists()
    assert output.is_dir()
