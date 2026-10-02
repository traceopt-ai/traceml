"""Validate RF-DETR release runs and produce a publication-oriented report."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
from contextlib import closing
from copy import deepcopy
from pathlib import Path

VERSIONS = ("1.10.1", "1.11.0", "1.11.1")
MODES = ("native", "traced")
REPEATS = (1, 2, 3)


def percent_delta(candidate: float, reference: float) -> float:
    if reference <= 0:
        raise ValueError("comparison reference must be positive")
    return (candidate / reference - 1.0) * 100.0


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"missing or invalid JSON: {path}") from exc


def comparison_identity(run: dict) -> dict:
    """Return all controlled facts while excluding the intended release delta."""
    identity = {
        "schema_version": run["schema_version"],
        "steps": run["steps"],
        "warmup_steps": run["warmup_steps"],
        "workload": run["workload"],
        "dataset": run["dataset"],
        "weights_sha256": run["weights_sha256"],
        "traceml": run["sources"]["traceml"],
        "script_sha256": run["sources"]["script_sha256"],
        "environment": deepcopy(run["environment"]),
    }
    return identity


def read_measurement(
    directory: Path, expected_mode: str, version: str
) -> dict:
    run = load_json(directory / "run.json")
    result = load_json(directory / "result.json")
    if run.get("schema_version") != 1:
        raise ValueError(f"{directory}: unsupported run schema")
    if run.get("mode") != expected_mode:
        raise ValueError(
            f"{directory}: expected {expected_mode}, found {run.get('mode')}"
        )
    observed_version = run.get("sources", {}).get("rfdetr", {}).get("version")
    if observed_version != version:
        raise ValueError(
            f"{directory}: expected RF-DETR {version}, found {observed_version}"
        )
    steps = run.get("steps")
    warmup = run.get("warmup_steps")
    if not isinstance(steps, int) or not isinstance(warmup, int):
        raise ValueError(f"{directory}: invalid measurement window")
    if not 0 <= warmup < steps:
        raise ValueError(f"{directory}: invalid measurement window")
    if (
        result.get("completed_steps") != steps
        or result.get("measured_steps") != steps - warmup
    ):
        raise ValueError(f"{directory}: incomplete measurement")
    elapsed = result.get("elapsed_s")
    loss = result.get("final_loss")
    if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed):
        raise ValueError(f"{directory}: invalid elapsed time")
    if (
        elapsed <= 0
        or not isinstance(loss, (int, float))
        or not math.isfinite(loss)
    ):
        raise ValueError(f"{directory}: invalid elapsed time or final loss")
    if result.get("precision") != run["workload"]["precision_resolved"]:
        raise ValueError(f"{directory}: resolved precision changed")
    return {"run": run, "result": result, "directory": directory}


def phase_metrics(path: Path, run: dict) -> dict[str, float]:
    from traceml_ai.step_time.analysis import StepTimeAnalyzer
    from traceml_ai.step_time.model import StepTimeLoadRequest
    from traceml_ai.step_time.sqlite import SQLiteStepTimeRepository

    start, end = run["warmup_steps"] + 1, run["steps"]
    expected_steps = list(range(start, end + 1))
    with closing(
        sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    ) as conn:
        snapshot = SQLiteStepTimeRepository(conn).load_summary(
            StepTimeLoadRequest(start_step=start, end_step=end)
        )
    if list(snapshot.global_ranks) != [0]:
        raise ValueError("traced run must contain exactly global rank 0")
    observed = sorted(
        row.step for row in snapshot.rows if row.global_rank == 0
    )
    if observed != expected_steps:
        raise ValueError(f"telemetry does not cover every step {start}-{end}")
    window = StepTimeAnalyzer().analyze(snapshot, window_size=None)
    if window.steps != expected_steps or window.clock != "gpu":
        raise ValueError("TraceML did not produce a complete GPU-clock window")
    if len(window.rank_facts) != 1:
        raise ValueError("TraceML result must contain exactly one rank")
    average = window.rank_facts[0].average
    names = (
        "step_time_ms",
        "input_wait_ms",
        "h2d_ms",
        "forward_ms",
        "backward_ms",
        "optimizer_step_ms",
        "residual_ms",
    )
    metrics = {name: getattr(average, name) for name in names}
    required = {name for name in names if name != "h2d_ms"}
    if any(
        metrics[name] is None
        or not math.isfinite(metrics[name])
        or metrics[name] < 0
        for name in required
    ):
        raise ValueError("TraceML phase metrics are missing or invalid")
    metrics["compute_ms"] = sum(
        metrics[name]
        for name in ("forward_ms", "backward_ms", "optimizer_step_ms")
    )
    return metrics


def collect(experiment_dir: Path) -> list[dict]:
    rows = []
    identity = None
    rfdetr_identities: dict[str, dict] = {}
    seen_directories = set()
    for repeat in REPEATS:
        for version in VERSIONS:
            pair = {}
            for mode in MODES:
                directory = (
                    experiment_dir
                    / "runs"
                    / f"repeat-{repeat}"
                    / version
                    / mode
                ).resolve()
                if directory in seen_directories:
                    raise ValueError("run directories must be distinct")
                seen_directories.add(directory)
                pair[mode] = read_measurement(directory, mode, version)
                observed_identity = comparison_identity(pair[mode]["run"])
                if identity is None:
                    identity = observed_identity
                elif observed_identity != identity:
                    raise ValueError(
                        f"controlled workload or environment changed at {directory}"
                    )
                source = pair[mode]["run"]["sources"]["rfdetr"]
                source_identity = {
                    "wheel_sha256": source["wheel_sha256"],
                    "python_tree_sha256": source["python_tree_sha256"],
                }
                previous = rfdetr_identities.setdefault(
                    version, source_identity
                )
                if source_identity != previous:
                    raise ValueError(
                        f"RF-DETR {version} source identity changed between runs"
                    )

            native_run = pair["native"]["run"]
            native_result = pair["native"]["result"]
            traced_result = pair["traced"]["result"]
            measured_steps = native_run["steps"] - native_run["warmup_steps"]
            native_ms = 1000.0 * native_result["elapsed_s"] / measured_steps
            traced_ms = 1000.0 * traced_result["elapsed_s"] / measured_steps
            trace_path = (
                pair["traced"]["directory"] / pair["traced"]["run"]["trace_db"]
            )
            rows.append(
                {
                    "repeat": repeat,
                    "version": version,
                    "native_ms": native_ms,
                    "traced_ms": traced_ms,
                    "overhead_pct": percent_delta(traced_ms, native_ms),
                    "phases": phase_metrics(trace_path, pair["traced"]["run"]),
                }
            )
    if (
        len({value["wheel_sha256"] for value in rfdetr_identities.values()})
        != 3
    ):
        raise ValueError("the three RF-DETR releases must use distinct wheels")
    return rows


def median_metrics(rows: list[dict]) -> dict[str, dict[str, float]]:
    result = {}
    for version in VERSIONS:
        selected = [row for row in rows if row["version"] == version]
        result[version] = {
            "native_ms": statistics.median(
                row["native_ms"] for row in selected
            ),
            "traced_ms": statistics.median(
                row["traced_ms"] for row in selected
            ),
            "overhead_pct": statistics.median(
                row["overhead_pct"] for row in selected
            ),
        }
        for key in selected[0]["phases"]:
            result[version][key] = statistics.median(
                row["phases"][key] for row in selected
            )
    return result


def evaluate(
    rows: list[dict],
    medians: dict[str, dict[str, float]],
    *,
    regression_threshold: float,
    stability_threshold: float,
    recovery_threshold: float,
    overhead_threshold: float,
) -> tuple[list[dict], dict[str, float]]:
    baseline = medians["1.10.1"]
    regressed = medians["1.11.0"]
    fixed = medians["1.11.1"]
    deltas = {
        "regressed_native_pct": percent_delta(
            regressed["native_ms"], baseline["native_ms"]
        ),
        "regressed_input_wait_pct": percent_delta(
            regressed["input_wait_ms"], baseline["input_wait_ms"]
        ),
        "regressed_compute_pct": percent_delta(
            regressed["compute_ms"], baseline["compute_ms"]
        ),
        "fixed_native_pct": percent_delta(
            fixed["native_ms"], baseline["native_ms"]
        ),
        "fixed_input_wait_pct": percent_delta(
            fixed["input_wait_ms"], baseline["input_wait_ms"]
        ),
    }
    by_repeat = {(row["repeat"], row["version"]): row for row in rows}
    checks = [
        {
            "name": "1.11.0 native wall time is slower in every repeat",
            "passed": all(
                by_repeat[(repeat, "1.11.0")]["native_ms"]
                > by_repeat[(repeat, "1.10.1")]["native_ms"]
                for repeat in REPEATS
            ),
        },
        {
            "name": "1.11.0 input wait is higher in every repeat",
            "passed": all(
                by_repeat[(repeat, "1.11.0")]["phases"]["input_wait_ms"]
                > by_repeat[(repeat, "1.10.1")]["phases"]["input_wait_ms"]
                for repeat in REPEATS
            ),
        },
        {
            "name": f"median native regression is at least {regression_threshold:.1f}%",
            "passed": deltas["regressed_native_pct"] >= regression_threshold,
        },
        {
            "name": f"median input-wait regression is at least {regression_threshold:.1f}%",
            "passed": deltas["regressed_input_wait_pct"]
            >= regression_threshold,
        },
        {
            "name": f"median compute change is within {stability_threshold:.1f}%",
            "passed": abs(deltas["regressed_compute_pct"])
            <= stability_threshold,
        },
        {
            "name": f"1.11.1 native wall time is within {recovery_threshold:.1f}% of baseline",
            "passed": abs(deltas["fixed_native_pct"]) <= recovery_threshold,
        },
        {
            "name": f"1.11.1 input wait is within {recovery_threshold:.1f}% of baseline",
            "passed": abs(deltas["fixed_input_wait_pct"])
            <= recovery_threshold,
        },
    ]
    checks.extend(
        {
            "name": f"{version} median TraceML overhead is at most {overhead_threshold:.1f}%",
            "passed": medians[version]["overhead_pct"] <= overhead_threshold,
        }
        for version in VERSIONS
    )
    return checks, deltas


def make_report(
    rows: list[dict],
    medians: dict[str, dict[str, float]],
    checks: list[dict],
    deltas: dict[str, float],
) -> str:
    supported = all(check["passed"] for check in checks)
    status = "SUPPORTED" if supported else "INCONCLUSIVE"
    lines = [
        "# RF-DETR non-JPEG input-pipeline regression",
        "",
        f"**Publication status: {status}.**",
        "",
    ]
    if supported:
        lines += [
            "The measurements support the following statement:",
            "",
            "> TraceML reproduced a released RF-DETR training regression, localized "
            "the change to input wait rather than GPU computation, and verified that "
            "the following release restored the baseline behavior.",
        ]
    else:
        lines += [
            "These measurements do not satisfy every predeclared publication check. "
            "They should not be presented as proof of the regression on this host.",
        ]
    lines += [
        "",
        "## Individual measurements",
        "",
        "| Repeat | RF-DETR | Native ms/step | Traced ms/step | TraceML overhead | Input wait ms | Compute ms |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['repeat']} | {row['version']} | {row['native_ms']:.3f} | "
            f"{row['traced_ms']:.3f} | {row['overhead_pct']:+.2f}% | "
            f"{row['phases']['input_wait_ms']:.3f} | "
            f"{row['phases']['compute_ms']:.3f} |"
        )
    lines += [
        "",
        "Compute is the sum of TraceML forward, backward and optimizer regions. "
        "Input wait is exposed waiting at the training-process boundary; it is not "
        "the total CPU preprocessing cost across worker processes.",
        "",
        "## Median comparison",
        "",
        "| RF-DETR | Native ms/step | Input wait ms | Compute ms | TraceML overhead |",
        "|---|---:|---:|---:|---:|",
    ]
    for version in VERSIONS:
        metric = medians[version]
        lines.append(
            f"| {version} | {metric['native_ms']:.3f} | "
            f"{metric['input_wait_ms']:.3f} | {metric['compute_ms']:.3f} | "
            f"{metric['overhead_pct']:+.2f}% |"
        )
    lines += [
        "",
        "Relative to RF-DETR 1.10.1:",
        "",
        f"- RF-DETR 1.11.0 native wall time: {deltas['regressed_native_pct']:+.2f}%",
        f"- RF-DETR 1.11.0 input wait: {deltas['regressed_input_wait_pct']:+.2f}%",
        f"- RF-DETR 1.11.0 compute regions: {deltas['regressed_compute_pct']:+.2f}%",
        f"- RF-DETR 1.11.1 native wall time: {deltas['fixed_native_pct']:+.2f}%",
        f"- RF-DETR 1.11.1 input wait: {deltas['fixed_input_wait_pct']:+.2f}%",
        "",
        "## Publication checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for check in checks:
        lines.append(
            f"| {check['name']} | {'PASS' if check['passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "This is a reproduction of a known public regression, not a claim of original "
        "discovery. It evaluates one generated non-JPEG workload on one host and does "
        "not establish model-accuracy changes or behavior for other image formats, "
        "hardware, worker counts or RF-DETR training modes.",
        "",
    ]
    return "\n".join(lines)


def analyze(
    experiment_dir: Path,
    *,
    regression_threshold: float = 10.0,
    stability_threshold: float = 10.0,
    recovery_threshold: float = 10.0,
    overhead_threshold: float = 5.0,
) -> dict:
    rows = collect(experiment_dir)
    medians = median_metrics(rows)
    checks, deltas = evaluate(
        rows,
        medians,
        regression_threshold=regression_threshold,
        stability_threshold=stability_threshold,
        recovery_threshold=recovery_threshold,
        overhead_threshold=overhead_threshold,
    )
    return {
        "schema_version": 1,
        "status": (
            "supported"
            if all(row["passed"] for row in checks)
            else "inconclusive"
        ),
        "thresholds_pct": {
            "minimum_regression": regression_threshold,
            "maximum_compute_change": stability_threshold,
            "maximum_recovery_delta": recovery_threshold,
            "maximum_traceml_overhead": overhead_threshold,
        },
        "runs": rows,
        "medians": medians,
        "deltas_pct": deltas,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    experiment_dir = args.experiment_dir.expanduser().resolve()
    try:
        result = analyze(experiment_dir)
    except (KeyError, OSError, ValueError, sqlite3.Error) as exc:
        parser.error(str(exc))
    output = experiment_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    serializable = deepcopy(result)
    for row in serializable["runs"]:
        row.pop("directory", None)
    (output / "analysis.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    report = make_report(
        result["runs"],
        result["medians"],
        result["checks"],
        result["deltas_pct"],
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
