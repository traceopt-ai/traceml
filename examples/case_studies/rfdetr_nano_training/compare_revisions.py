"""Compare RF-DETR PR #1489 before/after runs without conflating revisions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sqlite3
import statistics
from copy import deepcopy
from pathlib import Path

BEFORE_COMMIT = "1ac74e7a25edf0d771f075a414e86cb8dae085ee"
AFTER_COMMIT = "5b39cbc0a0ef75ab9c9571ccb28b780c7da51432"
MEASURED_STEPS = 40
WORLD_SIZE = 4
NATIVE_PAIRS = 5
TRACED_PAIRS = 3


def _load_summarizer():
    path = Path(__file__).with_name("summarize.py")
    spec = importlib.util.spec_from_file_location("rfdetr_case_summary", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summary = _load_summarizer()


def _comparison_config(run, *, ignore_revision=False):
    """Remove output identity while preserving the measured workload."""
    result = deepcopy(run)
    for key in ("mode", "trace_db"):
        result.pop(key, None)
    result["train_config"].pop("output_dir", None)
    if ignore_revision:
        result["sources"]["rfdetr"]["commit"] = "<revision>"
    return result


def _validate_workload(run, *, multi_scale=None):
    tc = run["train_config"]
    mc = run["model_config"]
    expected = {
        "steps": run["steps"] == 50,
        "warmup_steps": run["warmup_steps"] == 10,
        "world_size": run["world_size"] == WORLD_SIZE,
        "batch_size": tc["batch_size"] == 4,
        "num_workers": tc["num_workers"] == 2,
        "seed": tc["seed"] == 42,
        "resolution": mc["resolution"] == 384,
        "precision": run["trainer_precision"] == "16-mixed",
    }
    failed = [name for name, valid in expected.items() if not valid]
    if failed:
        raise ValueError(
            "Run does not match the four-T4 validation workload: "
            + ", ".join(failed)
        )
    resolved_multi_scale = bool(tc["multi_scale"])
    if multi_scale is not None and resolved_multi_scale != multi_scale:
        raise ValueError("Runs mix fixed-resolution and multi-scale workloads")
    return resolved_multi_scale


def _read(directory, mode, commit):
    run, ranks = summary.read_run(directory, mode)
    source = run["sources"]["rfdetr"]
    if source.get("commit") != commit or source.get("dirty"):
        raise ValueError(
            f"{directory}: expected clean RF-DETR {commit}, found {source}"
        )
    _validate_workload(run)
    for rank in ranks:
        for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
            value = rank.get(key)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{directory}: invalid {key} for one rank")
    return run, ranks


def _check_pair(before, after):
    if _comparison_config(before, ignore_revision=True) != _comparison_config(
        after, ignore_revision=True
    ):
        raise ValueError(
            "Before/after runs differ beyond the RF-DETR source revision"
        )


def _fresh_paths(pairs):
    paths = [Path(path).resolve() for pair in pairs for path in pair]
    if len(paths) != len(set(paths)):
        raise ValueError("Every comparison run must use a fresh directory")
    return paths


def _native_results(pairs):
    if len(pairs) != NATIVE_PAIRS:
        raise ValueError(f"Expected exactly {NATIVE_PAIRS} native pairs")
    _fresh_paths(pairs)
    results = []
    common = None
    multi_scale = None
    for before_path, after_path in pairs:
        before, before_ranks = _read(before_path, "baseline", BEFORE_COMMIT)
        after, after_ranks = _read(after_path, "baseline", AFTER_COMMIT)
        _check_pair(before, after)
        resolved = _validate_workload(before, multi_scale=multi_scale)
        multi_scale = resolved if multi_scale is None else multi_scale
        config = _comparison_config(before, ignore_revision=True)
        if common is not None and common != config:
            raise ValueError("Native repeats do not use one common workload")
        common = config
        before_ms = summary.window_ms(before, before_ranks)
        after_ms = summary.window_ms(after, after_ranks)
        results.append(
            {
                "before_ms": before_ms,
                "after_ms": after_ms,
                "delta_pct": (after_ms / before_ms - 1) * 100,
                "before_memory": _memory_max(before_ranks),
                "after_memory": _memory_max(after_ranks),
                "before_run": before,
                "after_run": after,
                "before_ranks": before_ranks,
                "after_ranks": after_ranks,
            }
        )
    return results, bool(multi_scale)


def _memory_max(ranks):
    return {
        key: max(float(rank[key]) for rank in ranks)
        for key in ("peak_allocated_bytes", "peak_reserved_bytes")
    }


def classify(deltas):
    """Classify paired step-time changes; negative means faster."""
    median = statistics.median(deltas)
    if all(delta < 0 for delta in deltas) and median <= -3:
        return "IMPROVEMENT"
    if all(delta > 0 for delta in deltas) and median >= 3:
        return "REGRESSION"
    return "INCONCLUSIVE"


def _phase_values(window):
    fields = (
        "step_time_ms",
        "input_wait_ms",
        "h2d_ms",
        "forward_ms",
        "backward_ms",
        "optimizer_step_ms",
        "residual_ms",
    )
    values = {}
    for field in fields:
        present = [
            getattr(rank.average, field)
            for rank in window.rank_facts
            if getattr(rank.average, field) is not None
        ]
        values[field] = statistics.mean(present) if present else None
    step_values = [
        rank.average.step_time_ms
        for rank in window.rank_facts
        if rank.average.step_time_ms is not None
    ]
    values["rank_spread_ms"] = max(step_values) - min(step_values)
    values["clock"] = window.clock
    return values


def _traced_results(pairs, native_reference):
    if not pairs:
        return []
    if len(pairs) != TRACED_PAIRS:
        raise ValueError(f"Expected exactly {TRACED_PAIRS} traced pairs")
    _fresh_paths(pairs)
    reference = native_reference[0]
    results = []
    for before_path, after_path in pairs:
        before, _ = _read(before_path, "traced", BEFORE_COMMIT)
        after, _ = _read(after_path, "traced", AFTER_COMMIT)
        _check_pair(before, after)
        if _comparison_config(
            before, ignore_revision=True
        ) != _comparison_config(reference["before_run"], ignore_revision=True):
            raise ValueError("Traced runs do not match the native workload")
        before_window = summary.phase_window(
            Path(before_path) / before["trace_db"], before
        )
        after_window = summary.phase_window(
            Path(after_path) / after["trace_db"], after
        )
        results.append(
            {
                "before": _phase_values(before_window),
                "after": _phase_values(after_window),
            }
        )
    return results


def _profiler_summary(directory, commit, native_run):
    run, _ = _read(directory, "profiler", commit)
    if _comparison_config(run, ignore_revision=True) != _comparison_config(
        native_run, ignore_revision=True
    ):
        raise ValueError("Profiler run does not match the native workload")
    scopes = {}
    collectives = []
    expected_scopes = {
        "rfdetr/criterion_including_matcher",
        "rfdetr/matcher",
        "rfdetr/optimizer_step",
        "rfdetr/ema_update",
        "rfdetr/lr_scheduler_step",
    }
    for rank in range(WORLD_SIZE):
        path = Path(directory) / f"profile-summary-rank-{rank}.json"
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Missing usable profiler summary: {path}"
            ) from exc
        if (
            payload.get("rank") != rank
            or payload.get("start_step") != 26
            or payload.get("end_step") != 35
            or payload.get("active_steps") != 10
        ):
            raise ValueError(f"{path}: incorrect profiler window or rank")
        missing = expected_scopes - set(payload.get("scopes", {}))
        if missing:
            raise ValueError(
                f"{path}: missing profiler scopes {sorted(missing)}"
            )
        for name in expected_scopes:
            metrics = payload["scopes"][name]
            if metrics["calls"] < 10:
                raise ValueError(f"{path}: incomplete profiler scope {name}")
            scopes.setdefault(name, []).append(
                {
                    "cpu_ms": metrics["cpu_total_ms"] / 10,
                    "cuda_ms": (
                        None
                        if metrics["cuda_total_ms"] is None
                        else metrics["cuda_total_ms"] / 10
                    ),
                }
            )
        collective = payload.get("collectives")
        if not isinstance(collective, dict) or "available" not in collective:
            raise ValueError(f"{path}: missing collective availability")
        collectives.append(collective)
    return {"scopes": scopes, "collectives": collectives}


def _median(values):
    finite = [value for value in values if value is not None]
    return statistics.median(finite) if finite else None


def _mib(value):
    return value / (1024 * 1024)


def _fmt(value, digits=3):
    return "unavailable" if value is None else f"{value:.{digits}f}"


def make_report(
    native_pairs,
    traced_pairs=(),
    profile_before=None,
    profile_after=None,
):
    native, multi_scale = _native_results(native_pairs)
    traced = _traced_results(traced_pairs, native)
    if (profile_before is None) != (profile_after is None):
        raise ValueError("Provide both profiler directories or neither")
    profiles = None
    if profile_before is not None:
        profiles = {
            "before": _profiler_summary(
                profile_before, BEFORE_COMMIT, native[0]["before_run"]
            ),
            "after": _profiler_summary(
                profile_after, AFTER_COMMIT, native[0]["after_run"]
            ),
        }

    deltas = [result["delta_pct"] for result in native]
    verdict = classify(deltas)
    run = native[0]["after_run"]
    tc = run["train_config"]
    lines = [
        "# RF-DETR PR #1489 four-T4 validation",
        "",
        f"Result: **{verdict}**.",
        "",
        f"Compared `{BEFORE_COMMIT}` with `{AFTER_COMMIT}` on four T4 GPUs "
        f"using real COCO train2017, FP16 and batch {tc['batch_size']} per rank. "
        f"The workload used {'native multi-scale' if multi_scale else 'fixed 384px'} training.",
        "",
        "## Native throughput",
        "",
        "Negative change means the revision after PR #1489 was faster.",
        "",
        "| Pair | Before ms/step | After ms/step | Change |",
        "|---:|---:|---:|---:|",
    ]
    for index, result in enumerate(native, 1):
        lines.append(
            f"| {index} | {result['before_ms']:.3f} | "
            f"{result['after_ms']:.3f} | {result['delta_pct']:+.2f}% |"
        )
    lines += [
        "",
        f"Median paired change: **{statistics.median(deltas):+.2f}%** "
        f"(range {min(deltas):+.2f}% to {max(deltas):+.2f}%).",
        "",
        "## Measurement-window CUDA memory",
        "",
        "Maximum rank value in each run, followed by the median across runs.",
        "",
        "| Metric | Before | After | Change |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("peak_allocated_bytes", "Peak allocated"),
        ("peak_reserved_bytes", "Peak reserved"),
    ):
        before = statistics.median(
            _mib(result["before_memory"][key]) for result in native
        )
        after = statistics.median(
            _mib(result["after_memory"][key]) for result in native
        )
        lines.append(
            f"| {label} | {before:.1f} MiB | {after:.1f} MiB | "
            f"{after - before:+.1f} MiB |"
        )

    if traced:
        labels = (
            ("step_time_ms", "Step envelope"),
            ("input_wait_ms", "Input wait"),
            ("h2d_ms", "H2D"),
            ("forward_ms", "Forward"),
            ("backward_ms", "Backward"),
            ("optimizer_step_ms", "Broad optimizer region"),
            ("residual_ms", "Residual"),
            ("rank_spread_ms", "Rank spread"),
        )
        lines += [
            "",
            "## TraceML phase attribution",
            "",
            "Median of the per-run four-rank means. Backward can include DDP "
            "communication; this is not direct NCCL timing. The optimizer region "
            "also includes scheduler, EMA and later batch-end callbacks.",
            "",
            "| Region | Before ms | After ms | Change |",
            "|---|---:|---:|---:|",
        ]
        for key, label in labels:
            before = statistics.median(row["before"][key] for row in traced)
            after = statistics.median(row["after"][key] for row in traced)
            lines.append(
                f"| {label} | {before:.3f} | {after:.3f} | "
                f"{after - before:+.3f} |"
            )

    if profiles is not None:
        labels = {
            "rfdetr/criterion_including_matcher": "Criterion, including matcher",
            "rfdetr/matcher": "Matcher",
            "rfdetr/optimizer_step": "Optimizer step",
            "rfdetr/ema_update": "EMA update",
            "rfdetr/lr_scheduler_step": "LR scheduler step",
        }
        lines += [
            "",
            "## Separate PyTorch Profiler attribution",
            "",
            "Median per-rank time over active steps 26–35. Scopes can overlap. "
            "Profiler runs are excluded from throughput results.",
            "",
            "| Scope | Before CPU ms | After CPU ms | Before CUDA ms | After CUDA ms |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, label in labels.items():
            before = profiles["before"]["scopes"][name]
            after = profiles["after"]["scopes"][name]
            lines.append(
                f"| {label} | {_fmt(_median([row['cpu_ms'] for row in before]))} | "
                f"{_fmt(_median([row['cpu_ms'] for row in after]))} | "
                f"{_fmt(_median([row['cuda_ms'] for row in before]))} | "
                f"{_fmt(_median([row['cuda_ms'] for row in after]))} |"
            )
        lines += [
            "",
            "### Explicit collective kernels",
            "",
            "These are CUDA kernels whose profiler names explicitly identify NCCL "
            "or a collective. Summed kernel time can overlap compute and is not "
            "collective wall time.",
            "",
            "| Revision | Calls/active step | CUDA ms/active step |",
            "|---|---:|---:|",
        ]
        for arm in ("before", "after"):
            entries = profiles[arm]["collectives"]
            if not all(entry["available"] for entry in entries):
                calls = cuda_ms = None
            else:
                calls = _median([entry["calls"] / 10 for entry in entries])
                cuda_ms = _median(
                    [entry["cuda_total_ms"] / 10 for entry in entries]
                )
            lines.append(
                f"| {arm.title()} | {_fmt(calls, 1)} | {_fmt(cuda_ms)} |"
            )

    lines += [
        "",
        "## Limits",
        "",
        "This experiment measures short-run throughput and attribution. It does "
        "not test mAP, convergence or long-run accuracy. TraceML does not measure "
        "NCCL collectives directly; collective details above come only from the "
        "separate PyTorch Profiler traces.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native-pair",
        nargs=2,
        action="append",
        required=True,
        metavar=("BEFORE", "AFTER"),
    )
    parser.add_argument(
        "--traced-pair",
        nargs=2,
        action="append",
        default=[],
        metavar=("BEFORE", "AFTER"),
    )
    parser.add_argument("--profile-before", type=Path)
    parser.add_argument("--profile-after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = make_report(
            args.native_pair,
            args.traced_pair,
            args.profile_before,
            args.profile_after,
        )
    except (ValueError, KeyError, OSError, sqlite3.Error) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
