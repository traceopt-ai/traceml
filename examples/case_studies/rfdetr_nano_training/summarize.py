"""Validate paired RF-DETR runs and write a shareable measured-window report."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
from contextlib import closing
from copy import deepcopy
from pathlib import Path


def comparison_config(run):
    """Ignore output locations, retaining every workload/environment setting."""
    result = deepcopy(run)
    for key in ("mode", "trace_db"):
        result.pop(key, None)
    result["train_config"].pop("output_dir", None)
    return result


def read_run(directory, mode):
    directory = Path(directory).resolve()
    run = json.loads((directory / "run.json").read_text())
    if run["mode"] != mode:
        raise ValueError(f"{directory}: expected {mode}, found {run['mode']}")
    steps, warmup, world = run["steps"], run["warmup_steps"], run["world_size"]
    if not 0 <= warmup < steps or world < 1:
        raise ValueError(f"{directory}: invalid measurement window/topology")
    ranks = []
    for rank in range(world):
        path = directory / f"rank-{rank}.json"
        if not path.is_file():
            raise ValueError(f"{directory}: missing completed rank {rank}")
        result = json.loads(path.read_text())
        if (
            result["rank"],
            result["completed_steps"],
            result["measured_steps"],
        ) != (rank, steps, steps - warmup):
            raise ValueError(f"{path}: incorrect rank or completed step count")
        if (
            not math.isfinite(result["elapsed_s"])
            or result["elapsed_s"] <= 0
            or not math.isfinite(result["final_loss"])
        ):
            raise ValueError(f"{path}: invalid elapsed time or final loss")
        if result["precision"] != run["trainer_precision"]:
            raise ValueError(f"{path}: ranks resolved different precisions")
        ranks.append(result)
    return run, ranks


def phase_window(path, run):
    """Analyze the exact measured steps with TraceML's existing analyzer."""
    from traceml_ai.step_time.analysis import StepTimeAnalyzer
    from traceml_ai.step_time.model import StepTimeLoadRequest
    from traceml_ai.step_time.sqlite import SQLiteStepTimeRepository

    start, end = run["warmup_steps"] + 1, run["steps"]
    expected_steps = list(range(start, end + 1))
    expected_ranks = set(range(run["world_size"]))
    with closing(
        sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    ) as conn:
        snapshot = SQLiteStepTimeRepository(conn).load_summary(
            StepTimeLoadRequest(start_step=start, end_step=end)
        )
    if set(snapshot.global_ranks) != expected_ranks:
        raise ValueError(
            "Telemetry is missing ranks or contains an unexpected rank"
        )
    for rank in expected_ranks:
        observed = sorted(
            row.step for row in snapshot.rows if row.global_rank == rank
        )
        if observed != expected_steps:
            raise ValueError(
                f"Rank {rank}: incomplete telemetry window {start}–{end}"
            )
    window = StepTimeAnalyzer().analyze(snapshot, window_size=None)
    if window.steps != expected_steps:
        raise ValueError(
            "TraceML could not align the complete measurement window"
        )
    if window.clock != "gpu":
        raise ValueError("CUDA case-study phase timing requires GPU events")
    for rank in window.rank_facts:
        required = (
            "step_time_ms",
            "input_wait_ms",
            "forward_ms",
            "backward_ms",
            "optimizer_step_ms",
            "residual_ms",
        )
        if any(getattr(rank.average, key) is None for key in required):
            raise ValueError(
                f"Rank {rank.global_rank}: required phase timings are unavailable"
            )
    return window


def window_ms(run, ranks):
    return (
        1000
        * max(rank["elapsed_s"] for rank in ranks)
        / (run["steps"] - run["warmup_steps"])
    )


def raw_cpu_scope_total(directory, rank, name, calls):
    """Read CPU user-annotation duration from the portable Chrome trace."""
    path = Path(directory) / f"profile-rank-{rank}.json"
    try:
        events = json.loads(path.read_text())["traceEvents"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"Missing usable profiler trace: {path}") from exc
    durations = [
        event.get("dur")
        for event in events
        if event.get("name") == name
        and event.get("cat") == "user_annotation"
        and event.get("ph") == "X"
    ]
    if len(durations) != calls or any(
        not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in durations
    ):
        raise ValueError(f"{path}: incomplete CPU annotations for {name}")
    return sum(durations) / 1000


def profiler_table(directory, reference, reference_ranks):
    run, ranks = read_run(directory, "profiler")
    if comparison_config(run) != comparison_config(reference) or [
        rank["gpu"] for rank in ranks
    ] != [rank["gpu"] for rank in reference_ranks]:
        raise ValueError(
            "Profiler run differs from the benchmark configuration"
        )
    lines = [
        "",
        "## Separate PyTorch profiler attribution",
        "",
        "Steps 26–35 only (wait 20, warmup 5, active 10), without TraceML. "
        "Inclusive CPU scope time and GPU-side scope span per active step. "
        "Criterion includes matcher; do not add rows or CPU/CUDA columns. "
        "These overlapping timings are excluded from benchmark throughput and overhead.",
        "",
        "| Rank | Scope | Calls | CPU ms/active step | GPU span ms/active step |",
        "|---|---|---:|---:|---:|",
    ]
    names = {
        "rfdetr/criterion_including_matcher": "Criterion (including matcher)",
        "rfdetr/matcher": "Matcher (batched and fallback calls)",
    }
    for rank in range(run["world_size"]):
        path = Path(directory) / f"profile-summary-rank-{rank}.json"
        if not path.is_file():
            raise ValueError(f"Missing profiler summary for rank {rank}")
        summary = json.loads(path.read_text())
        if (
            summary["rank"],
            summary["start_step"],
            summary["end_step"],
            summary["active_steps"],
        ) != (rank, 26, 35, 10):
            raise ValueError(
                f"{path}: incorrect profiler rank or active window"
            )
        for name, label in names.items():
            metrics = summary["scopes"][name]
            calls = metrics["calls"]
            if calls < 10 or (
                name.endswith("including_matcher") and calls != 10
            ):
                raise ValueError(f"{path}: incomplete scope calls for {name}")
            cells = []
            for key in ("cpu_total_ms", "cuda_total_ms"):
                value = metrics[key]
                if key == "cpu_total_ms" and value == 0:
                    value = raw_cpu_scope_total(directory, rank, name, calls)
                if value is None and key == "cuda_total_ms":
                    cells.append("n/a")
                elif value is None or not math.isfinite(value) or value < 0:
                    raise ValueError(f"{path}: invalid {key} for {name}")
                else:
                    cells.append(f"{value / 10:.3f}")
            lines.append(
                f"| {rank} | {label} | {calls} | " + " | ".join(cells) + " |"
            )
    lines += [
        "",
        "Matcher covers _match_many/forward; other target preparation stays in criterion. "
        "The GPU span runs from the first to the last GPU activity in each scope, "
        "so it may include idle time while host work completes. Inspect profile-rank-*.json "
        "for host synchronizations and individual kernels.",
    ]
    return lines


def make_report(pairs, profile_dir=None):
    results = []
    common = None
    used_directories = set()
    for baseline_path, traced_path in pairs:
        paths = [Path(baseline_path).resolve(), Path(traced_path).resolve()]
        if (
            any(path in used_directories for path in paths)
            or paths[0] == paths[1]
        ):
            raise ValueError(
                "Each repeat must use fresh, distinct run directories"
            )
        used_directories.update(paths)
        baseline, baseline_ranks = read_run(paths[0], "baseline")
        traced, traced_ranks = read_run(paths[1], "traced")
        config = comparison_config(baseline)
        if config != comparison_config(traced) or (
            common is not None and common != config
        ):
            raise ValueError(
                "Paired runs/repeats differ in workload, source, weights, data, or environment"
            )
        common = config
        if [r["gpu"] for r in baseline_ranks] != [
            r["gpu"] for r in traced_ranks
        ]:
            raise ValueError("Paired ranks used different GPU models")
        database = paths[1] / traced["trace_db"]
        phases = phase_window(database, traced)
        base_ms = window_ms(baseline, baseline_ranks)
        trace_ms = window_ms(traced, traced_ranks)
        results.append(
            (base_ms, trace_ms, (trace_ms / base_ms - 1) * 100, phases)
        )

    run = traced
    env = run["environment"]
    tc = run["train_config"]
    measured = run["steps"] - run["warmup_steps"]
    global_batch = tc["batch_size"] * run["world_size"]
    overhead = [row[2] for row in results]
    lines = [
        "# RF-DETR Nano training baseline — issue #1410",
        "",
        f"Tested {run['world_size']} GPU(s): {', '.join(r['gpu'] for r in traced_ranks)}. "
        f"Torch {env['torch']}, CUDA {env['cuda']}, precision {run['trainer_precision']}.",
        f"Batch {tc['batch_size']}/GPU (global {global_batch}), workers {tc['num_workers']}/rank; "
        f"COCO train2017, eager Nano at {run['model_config']['resolution']}, seed {tc['seed']}.",
        "Fixed resolution with torchvision augmentation; default multi-scale training is disabled.",
        f"RF-DETR commit: `{run['sources']['rfdetr']['commit']}`. "
        f"TraceML source: `{run['sources']['traceml']['commit']}`.",
        "",
        f"Measured steps {run['warmup_steps'] + 1}–{run['steps']} ({measured} per rank), "
        f"excluding {run['warmup_steps']} warmup steps and validation. "
        "Wall time includes input loading, with CUDA synchronization only at window boundaries. "
        "DDP uses the maximum rank elapsed time after an aligned start.",
        "",
        "## Training throughput",
        "",
        "Untraced native training; each row is a separate run.",
        "",
        "| Repeat | Wall ms/step | Images/s |",
        "|---|---:|---:|",
    ]
    for index, (base_ms, _, _, _) in enumerate(results, 1):
        lines.append(
            f"| {index} | {base_ms:.3f} | {global_batch * 1000 / base_ms:.2f} |"
        )
    lines += [
        "",
        "## TraceML phase timing",
        "",
        "Per-rank event-clock means over the same steps, distinct from wall throughput. "
        "H2D is included in the displayed phase decomposition.",
        "",
        "| Repeat | Rank | Clock | Outer step ms | Input wait ms | H2D ms | Forward ms | Backward ms | Optimizer region ms | Residual ms |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, (_, _, _, window) in enumerate(results, 1):
        for rank in window.rank_facts:
            fields = (
                "step_time_ms",
                "input_wait_ms",
                "h2d_ms",
                "forward_ms",
                "backward_ms",
                "optimizer_step_ms",
                "residual_ms",
            )
            cells = [
                (
                    "n/a"
                    if getattr(rank.average, key) is None
                    else f"{getattr(rank.average, key):.3f}"
                )
                for key in fields
            ]
            lines.append(
                f"| {index} | {rank.global_rank} | {window.clock.upper()} | "
                + " | ".join(cells)
                + " |"
            )
    lines += [
        "",
        "Forward covers the detector; backward includes DDP communication. "
        "The optimizer region includes scheduler/EMA callbacks. Residual is unassigned "
        "time, not a measurement of criterion/matcher cost. Input wait measures exposed "
        "waiting, not total worker preprocessing.",
    ]
    if profile_dir is not None:
        lines += profiler_table(profile_dir, run, traced_ranks)
    else:
        lines += [
            "",
            "For criterion/matcher attribution, run --profile and supply --profile-dir.",
        ]
    lines += [
        "",
        "## Settings differing from native defaults",
        "",
        "Relative to pinned RFDETRNanoConfig/TrainConfig defaults; paths are in run.json.",
        "",
        "| Setting | Native default | This run |",
        "|---|---|---|",
    ]
    overrides = run["configuration_overrides"]
    for section in ("model", "training"):
        for key, values in sorted(overrides[section].items()):
            lines.append(
                f"| {section}.{key} | `{json.dumps(values['default'])}` | "
                f"`{json.dumps(values['value'])}` |"
            )
    lines += [
        "",
        "Explicit native trainer overrides:",
        "```json",
        json.dumps(overrides["trainer"], indent=2),
        "```",
        "",
        "## Measurement overhead",
        "",
        "Paired CUDA-synchronized wall windows; profiler runs are excluded.",
        "",
        "| Repeat | Native wall ms/step | Traced wall ms/step | Delta ms/step | TraceML overhead |",
        "|---|---:|---:|---:|---:|",
    ]
    for index, (base_ms, trace_ms, delta, _) in enumerate(results, 1):
        lines.append(
            f"| {index} | {base_ms:.3f} | {trace_ms:.3f} | "
            f"{trace_ms - base_ms:+.3f} | {delta:+.2f}% |"
        )
    lines += [
        "",
        f"Median paired overhead: {statistics.median(overhead):+.2f}% "
        f"(range {min(overhead):+.2f}% to {max(overhead):+.2f}%). "
        "Negative deltas may reflect run variation.",
        "",
        "## Environment and reproducibility",
        "",
        "GPU inventory (name, UUID, driver, memory):",
        "```text",
        env["gpu_driver"],
        "```",
        "CPU:",
        "```text",
        env["cpu"],
        "```",
        "",
        "Share the code commit, run.json (configs, sources and data/weight checksums), "
        "environment.txt, rank results, telemetry and profiler traces.",
        "",
        "Eager baseline only; no compile, CUDA-graph or accuracy comparison. "
        "Any later optimization needs before/after mAP on the same evaluation set.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        required=True,
        metavar=("BASELINE", "TRACED"),
    )
    parser.add_argument(
        "--profile-dir", type=Path, help="Separate matching --profile run"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = make_report(args.pair, args.profile_dir)
    except (ValueError, KeyError, OSError, sqlite3.Error) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
