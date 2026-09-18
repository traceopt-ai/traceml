"""Check the RF-DETR case study's timing window and reporting contract."""

import importlib.util
import json
import sqlite3
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

CASE = (
    Path(__file__).resolve().parents[2]
    / "examples/case_studies/rfdetr_nano_training"
)


def load(name):
    spec = importlib.util.spec_from_file_location(
        "case_" + name, CASE / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


training = load("train")
reporting = load("summarize")


def telemetry(path, *, ranks=1, missing=None):
    from traceml_ai.step_time.model import STEP_TIME_EVENT_NAMES

    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE step_time_samples (id INTEGER PRIMARY KEY, global_rank INTEGER, step INTEGER, events_json TEXT)"
        )
        for rank in range(ranks):
            for step in range(1, 51):
                if (rank, step) == missing:
                    continue
                times = dict(
                    input_wait=1,
                    h2d=1,
                    forward=2,
                    backward=3,
                    optimizer_step=1,
                    traced_step_time=9,
                )
                events = {
                    STEP_TIME_EVENT_NAMES[key]: {
                        "cpu": {"cpu_ms": value if step > 10 else value * 100}
                    }
                    for key, value in times.items()
                }
                conn.execute(
                    "INSERT INTO step_time_samples (global_rank, step, events_json) VALUES (?, ?, ?)",
                    (rank, step, json.dumps(events)),
                )


def run_files(path, mode, *, ranks=1):
    path.mkdir()
    run = {
        "mode": mode,
        "steps": 50,
        "warmup_steps": 10,
        "world_size": ranks,
        "train_config": {
            "output_dir": str(path / "training"),
            "batch_size": 4,
            "num_workers": 2,
            "seed": 42,
        },
        "model_config": {"resolution": 384},
        "configuration_overrides": {
            "model": {},
            "training": {"multi_scale": {"default": True, "value": False}},
            "trainer": {"max_steps": 50, "limit_val_batches": 0},
        },
        "trainer_precision": "16-mixed",
        "trace_db": "telemetry" if mode == "traced" else None,
        "sources": {
            "rfdetr": {"commit": training.RFDETR_COMMIT},
            "traceml": {"commit": "fixture"},
        },
        "environment": {
            "torch": "fixture",
            "cuda": "fixture",
            "gpu_driver": "test fixture, not a GPU result",
            "cpu": "fixture",
        },
    }
    training.write_json(path / "run.json", run)
    for rank in range(ranks):
        training.write_json(
            path / f"rank-{rank}.json",
            {
                "rank": rank,
                "completed_steps": 50,
                "measured_steps": 40,
                "elapsed_s": (4 if mode == "baseline" else 4.4) + rank,
                "final_loss": 1.0,
                "precision": "16-mixed",
                "gpu": "test fixture",
            },
        )
    if mode == "traced":
        telemetry(path / "telemetry", ranks=ranks)
    return run


def test_summary_excludes_warmup_and_computes_paired_overhead(tmp_path):
    base, trace = tmp_path / "base", tmp_path / "trace"
    run_files(base, "baseline")
    run = run_files(trace, "traced")
    window = reporting.phase_window(trace / "telemetry", run)
    assert window.steps == list(range(11, 51))
    assert window.rank_facts[0].average.forward_ms == 2
    text = reporting.make_report([(base, trace)])
    assert "100.000 | 110.000" in text
    assert "Median paired overhead: +10.00%" in text
    assert "Residual ms" in text
    assert "not a measurement of criterion/matcher" in text
    assert "training.multi_scale | `true` | `false`" in text
    assert "default multi-scale" in text
    assert text.index("## Training throughput") < text.index(
        "## TraceML phase"
    )
    assert text.index("## TraceML phase") < text.index(
        "## Measurement overhead"
    )


@pytest.mark.parametrize("missing", [(0, 11), (0, 25), (1, 50)])
def test_missing_telemetry_step_rejected(tmp_path, missing):
    path = tmp_path / "telemetry"
    telemetry(path, ranks=2, missing=missing)
    with pytest.raises(ValueError, match="incomplete telemetry"):
        reporting.phase_window(
            path, {"steps": 50, "warmup_steps": 10, "world_size": 2}
        )


def test_missing_telemetry_rank_rejected(tmp_path):
    path = tmp_path / "telemetry"
    telemetry(path, ranks=1)
    with pytest.raises(ValueError, match="missing ranks"):
        reporting.phase_window(
            path, {"steps": 50, "warmup_steps": 10, "world_size": 4}
        )


@pytest.mark.parametrize(
    "change", ["workers", "steps", "rank", "profiler", "repeat"]
)
def test_incompatible_or_incomplete_runs_rejected(tmp_path, change):
    base, trace = tmp_path / "base", tmp_path / "trace"
    run_files(base, "baseline")
    run = run_files(trace, "traced")
    pairs = [(base, trace)]
    if change == "workers":
        run["train_config"]["num_workers"] = 4
        training.write_json(trace / "run.json", run)
    elif change == "steps":
        path = trace / "rank-0.json"
        payload = json.loads(path.read_text())
        payload["completed_steps"] = 49
        training.write_json(path, payload)
    elif change == "rank":
        (trace / "rank-0.json").unlink()
    elif change == "profiler":
        run["mode"] = "profiler"
        training.write_json(trace / "run.json", run)
    else:
        pairs.append((base, trace))
    with pytest.raises(ValueError):
        reporting.make_report(pairs)


def test_ddp_uses_slowest_rank_wall_window(tmp_path):
    run_files(tmp_path / "base", "baseline", ranks=4)
    run, ranks = reporting.read_run(tmp_path / "base", "baseline")
    assert reporting.window_ms(run, ranks) == 175  # 7 seconds / 40 steps


@pytest.mark.parametrize("change", [None, "config", "rank", "window", "calls"])
def test_separate_profiler_attribution_validated(tmp_path, change):
    base, trace, profile = (
        tmp_path / name for name in ("base", "trace", "profile")
    )
    run_files(base, "baseline", ranks=2)
    run_files(trace, "traced", ranks=2)
    run = run_files(profile, "profiler", ranks=2)
    for rank in range(2):
        summary = {
            "rank": rank,
            "start_step": 26,
            "end_step": 35,
            "active_steps": 10,
            "scopes": {
                "rfdetr/criterion_including_matcher": {
                    "calls": 10,
                    "cpu_total_ms": 50,
                    "cuda_total_ms": 30,
                },
                "rfdetr/matcher": {
                    "calls": 10,
                    "cpu_total_ms": 20,
                    "cuda_total_ms": 10,
                },
            },
        }
        if change == "window":
            summary["active_steps"] = 9
        if change == "calls":
            summary["scopes"]["rfdetr/matcher"]["calls"] = 0
        training.write_json(
            profile / f"profile-summary-rank-{rank}.json", summary
        )
    if change == "config":
        run["train_config"]["num_workers"] = 4
        training.write_json(profile / "run.json", run)
    elif change == "rank":
        (profile / "profile-summary-rank-1.json").unlink()
    if change is not None:
        with pytest.raises(ValueError):
            reporting.make_report([(base, trace)], profile)
    else:
        text = reporting.make_report([(base, trace)], profile)
        assert "Criterion (including matcher) | 10 | 5.000 | 3.000" in text
        assert (
            "Matcher (batched and fallback calls) | 10 | 2.000 | 1.000" in text
        )
        assert "Median paired overhead: +8.00%" in text
        assert text.index("## Separate PyTorch") < text.index(
            "## Measurement overhead"
        )


def test_vcs_install_inside_another_repo_uses_package_commit(
    tmp_path, monkeypatch
):
    (tmp_path / ".git").mkdir()
    source = tmp_path / "venv/site-packages/rfdetr/__init__.py"
    module = SimpleNamespace(__file__=str(source), __name__="rfdetr")

    def untracked(*args):
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(training, "command", untracked)
    installed = SimpleNamespace(
        read_text=lambda name: json.dumps(
            {"vcs_info": {"commit_id": "package-commit"}}
        ),
        locate_file=lambda name: source,
    )
    monkeypatch.setattr(
        training.importlib.metadata, "distribution", lambda name: installed
    )
    assert (
        training.source_identity(module, "rfdetr")["commit"]
        == "package-commit"
    )
    # A PYTHONPATH shadow must not inherit the installed package's identity.
    module.__file__ = str(tmp_path / "shadow/rfdetr/__init__.py")
    assert training.source_identity(module, "rfdetr")["commit"] is None


def test_cpu_identity_excludes_dynamic_frequency(monkeypatch):
    rows = {
        "lscpu": [
            {"field": "Model name:", "data": "Example CPU"},
            {"field": "CPU(s) scaling MHz:", "data": "73%"},
        ]
    }
    monkeypatch.setattr(training, "command", lambda *args: json.dumps(rows))
    assert training.cpu_description() == "Model name: Example CPU"


@pytest.mark.parametrize("batch_size", [1, 4])
def test_native_profiler_covers_fast_path_and_fallback(
    tmp_path, monkeypatch, batch_size
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("rfdetr")
    from rfdetr.config import RFDETRNanoConfig, TrainConfig
    from rfdetr.models import build_criterion_from_config

    mc = RFDETRNanoConfig(pretrain_weights=None, device="cpu")
    criterion, _ = build_criterion_from_config(
        mc, TrainConfig(dataset_dir="/unused")
    )
    criterion.train()
    module = SimpleNamespace(criterion=criterion)
    matcher_type = type(criterion.matcher)
    if not hasattr(matcher_type, "_match_many"):
        pytest.skip(
            "Batched matcher requires the case study's pinned development revision"
        )
    original_forward = matcher_type.forward
    original_many = matcher_type._match_many
    batched_results = []

    def observe_batched(self, *args, **kwargs):
        result = original_many(self, *args, **kwargs)
        batched_results.append(result is not None)
        return result

    layer = {
        "pred_logits": torch.randn(
            batch_size, 4 * mc.group_detr, mc.num_classes + 1
        ),
        "pred_boxes": torch.rand(batch_size, 4 * mc.group_detr, 4),
    }
    outputs = {**layer, "aux_outputs": [layer]}
    targets = [
        {
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        }
        for _ in range(batch_size)
    ]
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        expected = criterion(outputs, targets)
        monkeypatch.setattr(matcher_type, "_match_many", observe_batched)
        with (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                schedule=torch.profiler.schedule(
                    wait=20, warmup=5, active=10, repeat=1
                ),
                on_trace_ready=lambda prof: training.export_profile(
                    prof, tmp_path, 0
                ),
            ) as profiler,
            training.criterion_scopes(module),
        ):
            for _ in range(50):
                actual = criterion(outputs, targets)
                profiler.step()
        torch.testing.assert_close(actual, expected)
        assert all(torch.isfinite(loss) for loss in actual.values())
        assert batched_results == [batch_size > 1] * 50
        assert matcher_type.forward is original_forward
        assert matcher_type._match_many is observe_batched
        assert "forward" not in criterion.__dict__
        assert "forward" not in criterion.matcher.__dict__
        summary = json.loads(
            (tmp_path / "profile-summary-rank-0.json").read_text()
        )
        metrics = summary["scopes"]
        assert metrics["rfdetr/criterion_including_matcher"]["calls"] == 10
        assert metrics["rfdetr/matcher"]["calls"] == (
            10 if batch_size > 1 else 30
        )
        assert metrics["rfdetr/matcher"]["cpu_total_ms"] > 0
        assert metrics["rfdetr/matcher"]["cuda_total_ms"] is None
        assert json.loads((tmp_path / "profile-rank-0.json").read_text())[
            "traceEvents"
        ]
        with pytest.raises(RuntimeError, match="interrupted"):
            with training.criterion_scopes(module):
                raise RuntimeError("interrupted")
        assert matcher_type._match_many is observe_batched
        assert "forward" not in criterion.__dict__
        assert "forward" not in criterion.matcher.__dict__
    finally:
        torch.set_num_threads(threads)


def test_native_configuration_differences_expose_workload_changes():
    pytest.importorskip("rfdetr")
    from rfdetr.config import TrainConfig

    defaults = TrainConfig(dataset_dir="/unused")
    actual = TrainConfig(
        dataset_dir="/data",
        output_dir="/output",
        multi_scale=False,
        expanded_scales=False,
        augmentation_backend="torchvision",
        tensorboard=False,
    )
    changes = training.config_changes(actual, defaults)
    assert changes["multi_scale"] == {"default": True, "value": False}
    assert changes["expanded_scales"] == {"default": True, "value": False}
    assert changes["augmentation_backend"] == {
        "default": "cpu",
        "value": "torchvision",
    }
    assert changes["tensorboard"] == {"default": True, "value": False}
    assert "dataset_dir" not in changes and "output_dir" not in changes
    assert "use_ema" not in changes


def test_timer_starts_after_warmup_and_includes_next_fetch(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    clock = iter([10.0, 14.0])
    monkeypatch.setattr(training.time, "perf_counter", lambda: next(clock))
    barriers = []
    trainer = SimpleNamespace(
        global_step=0,
        strategy=SimpleNamespace(barrier=lambda: barriers.append(True)),
    )
    module = SimpleNamespace(device=torch.device("cpu"))
    timer = training.make_window_callback(10, 50)
    timer.on_train_start(trainer, module)
    for step in range(1, 51):
        trainer.global_step = step
        timer.on_train_batch_end(
            trainer, module, {"loss": torch.tensor(1.0)}, None, step - 1
        )
        if step < 10:
            assert timer.started is None
    assert timer.elapsed_s == 4
    assert timer.completed_steps == 50
    assert barriers == [True]


@pytest.mark.parametrize("profile", [False, True])
def test_native_nano_cpu_smoke_preserves_callbacks(
    tmp_path, monkeypatch, profile
):
    """Tiny offline detector verifies wiring only; never a reported benchmark."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("rfdetr")
    from PIL import Image
    import rfdetr.training
    from rfdetr.config import RFDETRNanoConfig, TrainConfig
    from rfdetr.training.callbacks.ema import RFDETREMACallback
    from traceml_ai.integrations import rfdetr as tracing
    from traceml_ai.runtime.state import reset_trace_session_state
    from traceml_ai.instrumentation.step_events import (
        drain_step_time_batches,
        drain_step_memory_events,
    )

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("TRACEML_DISABLED", "1" if profile else "0")
    monkeypatch.setattr(
        rfdetr.training, "build_trainer", rfdetr.training.build_trainer
    )
    reset_trace_session_state()
    drain_step_time_batches()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    dataset = tmp_path / "coco"
    (dataset / "annotations").mkdir(parents=True)
    for split in ("train2017", "val2017"):
        (dataset / split).mkdir()
        images, annotations = [], []
        for i in range(4):
            Image.new("RGB", (64, 64), (80, 120, 160)).save(
                dataset / split / f"{i}.jpg"
            )
            images.append(
                {"id": i, "file_name": f"{i}.jpg", "height": 64, "width": 64}
            )
            annotations.append(
                {
                    "id": i,
                    "image_id": i,
                    "category_id": 1,
                    "bbox": [8, 8, 24, 24],
                    "area": 576,
                    "iscrowd": 0,
                }
            )
        training.write_json(
            dataset / "annotations" / f"instances_{split}.json",
            {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "object"}],
            },
        )
    try:
        tracing.init()
        mc = RFDETRNanoConfig(
            pretrain_weights=None,
            device="cpu",
            resolution=64,
            num_queries=4,
            num_select=4,
            group_detr=1,
        )
        tc = TrainConfig(
            dataset_file="coco",
            dataset_dir=str(dataset),
            output_dir=str(tmp_path / "out"),
            batch_size=1,
            num_workers=0,
            devices=1,
            accelerator="cpu",
            multi_scale=False,
            expanded_scales=False,
            augmentation_backend="torchvision",
            tensorboard=False,
            progress_bar=None,
        )
        module = rfdetr.training.RFDETRModelModule(mc, tc)
        data = rfdetr.training.RFDETRDataModule(mc, tc)
        trainer = rfdetr.training.build_trainer(
            tc,
            mc,
            max_steps=3,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            logger=False,
            enable_model_summary=False,
        )
        original = tuple(trainer.callbacks)
        profiler = (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            )
            if profile
            else None
        )
        timer = training.make_window_callback(1, 3, profiler)
        trainer.callbacks.append(timer)
        with profiler if profile else nullcontext():
            try:
                trainer.fit(module, datamodule=data)
            finally:
                timer.close_scopes()
        assert all(callback in trainer.callbacks for callback in original)
        assert any(isinstance(cb, RFDETREMACallback) for cb in original)
        assert sum(
            isinstance(cb, tracing._callback_class()) for cb in original
        ) == (0 if profile else 1)
        assert timer.result(trainer)["measured_steps"] == 2
        assert [event.step for event in drain_step_time_batches()] == (
            [] if profile else [1, 2, 3]
        )
        assert "forward" not in module.criterion.__dict__
        if profile:
            ema = next(
                cb for cb in original if isinstance(cb, RFDETREMACallback)
            )
            assert (
                "forward" not in ema._average_model.module.criterion.__dict__
            )
            assert set(training.PROFILE_SCOPES) <= {
                event.key for event in profiler.key_averages()
            }
    finally:
        torch.set_num_threads(threads)
        drain_step_time_batches()
        drain_step_memory_events()
        reset_trace_session_state()
