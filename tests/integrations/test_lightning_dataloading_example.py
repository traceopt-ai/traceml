"""The Lightning data-loading example: smoke path, profiles, notebook copy."""

import importlib
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
pytest.importorskip("torchvision")

from traceml_ai.instrumentation.hooks.optimizer_hooks import (  # noqa: E402
    reset_optimizer_timing,
)
from traceml_ai.instrumentation.step_events import (  # noqa: E402
    abort_step_capture,
    begin_step_capture,
    drain_step_memory_events,
    drain_step_time_batches,
)
from traceml_ai.runtime.state import (  # noqa: E402
    configure_trace_recording,
    reset_trace_session_state,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "integrations"
EXAMPLE = EXAMPLE / "lightning_dataloading_bottleneck.py"
NOTEBOOK = ROOT / "notebooks" / "lightning_dataloading_bottleneck.ipynb"

STEP = "_traceml_internal:step_time"
FETCH = "_traceml_internal:dataloader_next"
FORWARD = "_traceml_internal:forward_time"
BACKWARD = "_traceml_internal:backward_time"
OPTIMIZER = "_traceml_internal:optimizer_step"


@pytest.fixture(autouse=True)
def _reset_traceml():
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


def _example():
    # Import under the file's real name from its own directory so DataLoader
    # worker processes started with "spawn" can unpickle the dataset class.
    if str(EXAMPLE.parent) not in sys.path:
        sys.path.insert(0, str(EXAMPLE.parent))
    return importlib.import_module(EXAMPLE.stem)


def _counts(batches, name):
    return [sum(e.name == name for e in b.events) for b in batches]


def test_smoke_profile_publishes_one_step_per_batch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    example = _example()

    example.main(["--smoke", "--max-steps", "4", "--batch-size", "2"])
    batches = drain_step_time_batches()

    assert [b.step for b in batches] == [1, 2, 3, 4]
    for name in (FETCH, STEP, FORWARD, BACKWARD, OPTIMIZER):
        assert _counts(batches, name) == [1, 1, 1, 1], name
    assert not list(tmp_path.iterdir()), "smoke mode must download nothing"


def test_smoke_steps_across_an_epoch_boundary_keep_one_fetch_each(
    tmp_path, monkeypatch
):
    # 32 synthetic samples at batch 4 is an 8-batch epoch; 12 steps cross
    # the boundary, where Lightning re-creates the loader iterator.
    monkeypatch.chdir(tmp_path)
    example = _example()

    example.main(["--smoke", "--max-steps", "12", "--batch-size", "4"])
    batches = drain_step_time_batches()

    assert [b.step for b in batches] == list(range(1, 13))
    assert _counts(batches, FETCH) == [1] * 12
    assert _counts(batches, STEP) == [1] * 12


def test_smoke_with_workers_publishes_one_step_per_batch(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    example = _example()

    example.main(
        [
            "--smoke",
            "--num-workers",
            "2",
            "--max-steps",
            "4",
            "--batch-size",
            "2",
        ]
    )
    batches = drain_step_time_batches()

    assert [b.step for b in batches] == [1, 2, 3, 4]
    assert _counts(batches, FETCH) == [1, 1, 1, 1]


def test_parser_maps_the_persistent_workers_flags():
    parser = _example().build_parser()

    assert parser.parse_args([]).persistent_workers is None
    assert parser.parse_args(["--persistent-workers"]).persistent_workers
    assert (
        parser.parse_args(["--no-persistent-workers"]).persistent_workers
        is False
    )


@pytest.mark.parametrize(
    "profile, kwargs, expected",
    [
        ("baseline", {}, (0, False, False)),
        ("optimized", {"num_workers": 4}, (4, True, True)),
        (
            "optimized",
            {"num_workers": 4, "persistent_workers": False},
            (4, True, False),
        ),
        ("optimized", {"num_workers": 0}, (0, True, False)),
        ("baseline", {"num_workers": 2}, (2, False, False)),
        ("optimized", {"smoke": True}, (0, False, False)),
        ("optimized", {"smoke": True, "num_workers": 2}, (2, False, True)),
    ],
)
def test_loader_settings_follow_the_profile_table(profile, kwargs, expected):
    settings = _example().loader_settings(profile, **kwargs)

    assert (
        settings["num_workers"],
        settings["pin_memory"],
        settings["persistent_workers"],
    ) == expected


def test_optimized_worker_count_matches_cores_up_to_four(monkeypatch):
    example = _example()
    monkeypatch.setattr(example.os, "cpu_count", lambda: 2)
    assert example.loader_settings("optimized")["num_workers"] == 2
    monkeypatch.setattr(example.os, "cpu_count", lambda: 16)
    assert example.loader_settings("optimized")["num_workers"] == 4


def test_notebook_script_cell_is_the_example_file():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    magic = "%%writefile lightning_dataloading_bottleneck.py\n"
    scripts = [
        "".join(c["source"])
        for c in cells
        if c["cell_type"] == "code" and "".join(c["source"]).startswith(magic)
    ]

    assert len(scripts) == 1, "exactly one cell writes the training script"
    assert scripts[0][len(magic) :] == EXAMPLE.read_text(encoding="utf-8")
