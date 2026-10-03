"""Check the RF-DETR release-regression case-study contract."""

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

CASE = (
    Path(__file__).resolve().parents[2]
    / "examples/case_studies/rfdetr_input_pipeline_regression"
)


def load(name):
    spec = importlib.util.spec_from_file_location(
        "rfdetr_regression_" + name, CASE / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dataset = load("generate_dataset")
analysis = load("analyze")
training = load("train")


@pytest.mark.parametrize(
    ("model_fields", "expected"),
    [
        (
            {"do_random_resize_via_padding": object()},
            {"do_random_resize_via_padding": False},
        ),
        ({"multi_scale": object()}, {}),
    ],
)
def test_version_specific_train_options(model_fields, expected):
    assert training.version_specific_train_options(model_fields) == expected


def test_dataset_is_deterministic_coco_and_refuses_replacement(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    options = {
        "image_format": "png",
        "image_size": 384,
        "train_images": 4,
        "val_images": 1,
        "seed": 1544,
    }
    first_manifest = dataset.generate(first, **options)
    second_manifest = dataset.generate(second, **options)

    assert first_manifest == second_manifest
    assert len(first_manifest["files"]) == 7
    annotation = json.loads(
        (first / "annotations/instances_train2017.json").read_text()
    )
    assert len(annotation["images"]) == 4
    assert len(annotation["annotations"]) == 4
    assert all(
        row["bbox"] == [76, 76, 232, 232] for row in annotation["annotations"]
    )

    with pytest.raises(FileExistsError, match="refusing to replace"):
        dataset.generate(first, **options)


def write_telemetry(path, *, input_wait, step_time):
    from traceml_ai.step_time.model import STEP_TIME_EVENT_NAMES

    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE step_time_samples ("
            "id INTEGER PRIMARY KEY, global_rank INTEGER, step INTEGER, "
            "events_json TEXT)"
        )
        for step in range(1, 51):
            durations = {
                "input_wait": input_wait,
                "h2d": 1.0,
                "forward": 2.0,
                "backward": 3.0,
                "optimizer_step": 1.0,
                "traced_step_time": step_time,
            }
            events = {
                STEP_TIME_EVENT_NAMES[name]: {
                    "cpu": {"cpu_ms": value},
                    "cuda:0": {"cpu_ms": value, "gpu_ms": value},
                }
                for name, value in durations.items()
            }
            connection.execute(
                "INSERT INTO step_time_samples "
                "(global_rank, step, events_json) VALUES (?, ?, ?)",
                (0, step, json.dumps(events)),
            )


def write_run(
    root, *, repeat, version, mode, native_ms, input_wait, step_time
):
    directory = root / "runs" / f"repeat-{repeat}" / version / mode
    directory.mkdir(parents=True)
    traced = mode == "traced"
    run = {
        "schema_version": 1,
        "mode": mode,
        "steps": 50,
        "warmup_steps": 10,
        "trace_db": "telemetry" if traced else None,
        "workload": {
            "model": "RF-DETR Nano",
            "resolution": 384,
            "batch_size": 4,
            "num_workers": 1,
            "seed": 1544,
            "precision_requested": "fp16",
            "precision_resolved": "16-mixed",
            "augmentation_backend": "torchvision",
            "multi_scale": False,
            "compile": False,
            "cuda_graphs": False,
        },
        "dataset": {
            "manifest_sha256": "dataset",
            "generator": {"image_format": "png"},
            "annotations": {"train2017": "train", "val2017": "val"},
        },
        "sources": {
            "rfdetr": {
                "version": version,
                "wheel_sha256": version * 8,
                "python_tree_sha256": "tree-" + version,
            },
            "traceml": {"commit": "fixture", "dirty": False},
            "script_sha256": "script",
        },
        "weights_sha256": "weights",
        "environment": {
            "python": "3.11",
            "torch": "fixture",
            "cuda": "12.8",
            "controlled_packages_sha256": "packages",
            "hostname": "fixture",
            "platform": "fixture",
            "cpu": "fixture",
            "cpu_count": 8,
            "cpu_affinity": list(range(8)),
            "omp_num_threads": "2",
            "cuda_visible_devices": "0",
            "gpu_driver": "fixture GPU",
        },
    }
    elapsed_ms = native_ms * (1.02 if traced else 1.0)
    result = {
        "completed_steps": 50,
        "measured_steps": 40,
        "elapsed_s": elapsed_ms * 40 / 1000,
        "final_loss": 1.0,
        "precision": "16-mixed",
        "gpu": "fixture GPU",
    }
    (directory / "run.json").write_text(json.dumps(run))
    (directory / "result.json").write_text(json.dumps(result))
    if traced:
        write_telemetry(
            directory / "telemetry",
            input_wait=input_wait,
            step_time=step_time,
        )


def test_complete_release_sequence_supports_only_declared_claim(tmp_path):
    fixture = {
        "1.10.1": {"native_ms": 20.0, "input_wait": 10.0, "step_time": 20.0},
        "1.11.0": {"native_ms": 24.0, "input_wait": 14.0, "step_time": 24.0},
        "1.11.1": {"native_ms": 20.5, "input_wait": 10.5, "step_time": 20.5},
    }
    for repeat in analysis.REPEATS:
        for version, values in fixture.items():
            for mode in analysis.MODES:
                write_run(
                    tmp_path,
                    repeat=repeat,
                    version=version,
                    mode=mode,
                    **values,
                )

    result = analysis.analyze(tmp_path)

    assert result["status"] == "supported"
    assert result["schema_version"] == 2
    assert result["evaluation_protocol_version"] == 2
    assert result["deltas_pct"]["regressed_native_pct"] == pytest.approx(20.0)
    assert result["deltas_pct"]["regressed_input_wait_pct"] == pytest.approx(
        40.0
    )
    assert all(check["passed"] for check in result["checks"])
    report = analysis.make_report(
        result["runs"],
        result["medians"],
        result["checks"],
        result["deltas_pct"],
    )
    assert "Evaluation status: SUPPORTED" in report
    assert "Evaluation protocol: 2" in report
    assert "not a claim of original discovery" in report


def test_status_is_inconclusive_when_compute_regresses():
    rows = []
    for repeat in analysis.REPEATS:
        for version, native, wait, compute in (
            ("1.10.1", 20.0, 10.0, 6.0),
            ("1.11.0", 24.0, 14.0, 8.0),
            ("1.11.1", 20.5, 10.5, 6.0),
        ):
            rows.append(
                {
                    "repeat": repeat,
                    "version": version,
                    "native_ms": native,
                    "traced_ms": native * 1.02,
                    "overhead_pct": 2.0,
                    "phases": {"input_wait_ms": wait, "compute_ms": compute},
                }
            )
    medians = analysis.median_metrics(rows)
    checks, deltas = analysis.evaluate(
        rows,
        medians,
        regression_threshold=10.0,
        stability_threshold=10.0,
        recovery_threshold=10.0,
        overhead_threshold=5.0,
    )
    compute_check = next(
        check
        for check in checks
        if "1.11.0 median compute regression" in check["name"]
    )
    assert not compute_check["passed"]
    report = analysis.make_report(rows, medians, checks, deltas)
    assert "Evaluation status: INCONCLUSIVE" in report
    assert "Failed checks:" in report
    assert compute_check["name"] in report


def test_missing_h2d_does_not_break_median_analysis():
    rows = []
    for repeat in analysis.REPEATS:
        for version in analysis.VERSIONS:
            rows.append(
                {
                    "repeat": repeat,
                    "version": version,
                    "native_ms": 20.0,
                    "traced_ms": 20.4,
                    "overhead_pct": 2.0,
                    "phases": {
                        "input_wait_ms": 10.0,
                        "compute_ms": 6.0,
                        "h2d_ms": None,
                    },
                }
            )

    medians = analysis.median_metrics(rows)

    assert all(
        medians[version]["h2d_ms"] is None for version in analysis.VERSIONS
    )


def test_compute_improvement_does_not_invalidate_localization():
    rows = []
    for repeat in analysis.REPEATS:
        for version, native, wait, compute in (
            ("1.10.1", 20.0, 10.0, 10.0),
            ("1.11.0", 24.0, 14.0, 8.9),
            ("1.11.1", 20.5, 10.5, 8.8),
        ):
            rows.append(
                {
                    "repeat": repeat,
                    "version": version,
                    "native_ms": native,
                    "traced_ms": native * 1.02,
                    "overhead_pct": 2.0,
                    "phases": {"input_wait_ms": wait, "compute_ms": compute},
                }
            )
    medians = analysis.median_metrics(rows)
    checks, deltas = analysis.evaluate(
        rows,
        medians,
        regression_threshold=10.0,
        stability_threshold=10.0,
        recovery_threshold=10.0,
        overhead_threshold=5.0,
    )

    assert deltas["regressed_compute_pct"] == pytest.approx(-11.0)
    assert deltas["fixed_vs_regressed_compute_pct"] == pytest.approx(
        -1.1235955
    )
    assert all(check["passed"] for check in checks)
