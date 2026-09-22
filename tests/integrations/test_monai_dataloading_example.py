"""The MONAI spleen example: smoke path, run table, notebook copy."""

import importlib
import json
import re
import shlex
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")
pytest.importorskip("ignite")

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
EXAMPLE = EXAMPLE / "monai_dataloading_bottleneck.py"
NOTEBOOK = ROOT / "notebooks" / "monai_dataloading_bottleneck.ipynb"

STEP = "_traceml_internal:step_time"
FETCH = "_traceml_internal:dataloader_next"
FORWARD = "_traceml_internal:forward_time"
BACKWARD = "_traceml_internal:backward_time"
OPTIMIZER = "_traceml_internal:optimizer_step"

# The case study's run table: each run changes one setting from the run
# before it, as (dataset, num_workers, loader, amp).
RUNS = [
    ("plain", 0, "torch", False),
    ("plain", 4, "torch", False),
    ("cache", 4, "torch", False),
    ("cache", 0, "torch", False),
    ("cache", 0, "thread", False),
    ("cache", 0, "thread", True),
]
DATASETS = {"plain": "Dataset", "cache": "CacheDataset"}
LOADERS = {"torch": "DataLoader", "thread": "ThreadDataLoader"}


@pytest.fixture(autouse=True)
def _reset_traceml(monkeypatch):
    # Same isolation as test_monai.py: an earlier file's init() may have left
    # the process-wide fetch patch on, which would count Input Wait twice.
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
    reset_optimizer_timing()


def _example():
    if str(EXAMPLE.parent) not in sys.path:
        sys.path.insert(0, str(EXAMPLE.parent))
    return importlib.import_module(EXAMPLE.stem)


def _counts(batches, name):
    return [sum(e.name == name for e in b.events) for b in batches]


def _flags(dataset, num_workers, loader, amp):
    flags = ["--dataset", dataset, "--num-workers", str(num_workers)]
    flags += ["--loader", loader]
    return flags + (["--amp"] if amp else [])


def _settings(args):
    return (args.dataset, args.num_workers, args.loader, args.amp)


def _notebook_code():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    return ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]


# On CPU, SupervisedTrainer disables its CUDA autocast and grad scaler, so
# run 6 checks only that --amp reaches the trainer and the steps stay whole.
@pytest.mark.parametrize("run", RUNS, ids=[f"run{i}" for i in range(1, 7)])
def test_each_run_publishes_one_complete_step_per_iteration(
    run, tmp_path, monkeypatch
):
    pytest.importorskip("nibabel")
    monkeypatch.chdir(tmp_path)
    example = _example()

    def no_download(data_dir):
        raise AssertionError("smoke mode reached the spleen download")

    monkeypatch.setattr(example, "spleen_files", no_download)
    # Four smoke volumes at batch 2 is two iterations per epoch.
    record = example.main(["--smoke", "--epochs", "2", *_flags(*run)])
    batches = drain_step_time_batches()

    assert [b.step for b in batches] == [1, 2, 3, 4]
    for name in (FETCH, STEP, FORWARD, BACKWARD, OPTIMIZER):
        assert _counts(batches, name) == [1, 1, 1, 1], name
    assert record["steps"] == 4
    # The record reads the settings back from the trainer, so this checks
    # that each flag reached the objects that ran.
    dataset, num_workers, loader, amp = run
    assert record["dataset"] == DATASETS[dataset]
    assert record["loader"] == LOADERS[loader]
    assert (record["num_workers"], record["amp"]) == (num_workers, amp)
    assert record["volumes"] == 4
    assert record["cached_volumes"] == (4 if dataset == "cache" else None)
    assert not list(tmp_path.iterdir()), "smoke mode must download nothing"


def test_the_run_record_holds_every_field_the_study_reports():
    pytest.importorskip("nibabel")
    record = _example().main(["--smoke", "--epochs", "1"])

    assert {
        "argv",
        "gpu",
        "driver",
        "torch",
        "monai",
        "ignite",
        "traceml",
        "seed",
        "dataset",
        "loader",
        "amp",
        "epochs",
        "patches_per_volume",
        "cache_rate",
        "cached_volumes",
        "volumes",
        "num_workers",
        "batch_size",
        "patch_size",
        "steps",
        "dataset_ready_s",
        "train_s",
    } <= record.keys()
    assert record["argv"] == ["--smoke", "--epochs", "1"]
    assert record["cache_rate"] is None


def test_the_notebook_runs_change_one_setting_at_a_time():
    parser = _example().build_parser()
    commands = [
        line.split("--args", 1)[1]
        for code in _notebook_code()
        for line in code.splitlines()
        if "!traceml run" in line and "--smoke" not in line
    ]
    parsed = [parser.parse_args(shlex.split(c)) for c in commands]

    assert [_settings(a) for a in parsed] == RUNS
    for before, after in zip(RUNS, RUNS[1:]):
        assert sum(b != a for b, a in zip(before, after)) == 1
    held = {(a.batch_size, a.epochs, a.data_dir) for a in parsed}
    assert len(held) == 1, "every run keeps batch, epochs and data fixed"


def test_the_notebook_compares_each_adjacent_pair():
    names = [
        re.search(r"--run-name (\S+)", line).group(1)
        for code in _notebook_code()
        for line in code.splitlines()
        if "!traceml run" in line and "--smoke" not in line
    ]
    compares = [
        re.findall(r"logs/(\S+?)/final_summary\.json", line)
        for code in _notebook_code()
        for line in code.splitlines()
        if "!traceml compare" in line
    ]

    assert len(set(names)) == len(names), "each run writes its own logs"
    assert compares == [list(pair) for pair in zip(names, names[1:])]


def test_notebook_script_cell_is_the_example_file():
    magic = "%%writefile monai_dataloading_bottleneck.py\n"
    scripts = [c for c in _notebook_code() if c.startswith(magic)]

    assert len(scripts) == 1, "exactly one cell writes the training script"
    assert scripts[0][len(magic) :] == EXAMPLE.read_text(encoding="utf-8")


def test_a_partial_spleen_directory_fails_instead_of_training_on_less(
    tmp_path,
):
    # An interrupted extraction leaves the directory, which skips the
    # download; the script must refuse it rather than train on fewer volumes.
    root = tmp_path / "Task09_Spleen"
    for folder in ("imagesTr", "labelsTr"):
        (root / folder).mkdir(parents=True)
        for index in range(10):
            (root / folder / f"spleen_{index}.nii.gz").touch()

    with pytest.raises(RuntimeError, match="Expected 41 labelled volumes"):
        _example().spleen_files(str(tmp_path))
