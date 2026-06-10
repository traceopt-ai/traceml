"""Run paired TraceML overhead trials for the synthetic MLP workload.

This runner handles single-node benchmark cells. Multi-node runs need a
scheduler or one launcher per node; see ``benchmarks/README.md`` for that
workflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = REPO_ROOT / "src"
DEFAULT_WORKLOAD = (
    REPO_ROOT / "benchmarks" / "workloads" / "ddp_mlp_overhead.py"
)
LAUNCHER_SNIPPET = "from traceml_ai.launcher.cli import main; main()"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired native-vs-TraceML overhead trials."
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results",
    )
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--input-dim", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--output-dim", type=int, default=1000)
    parser.add_argument("--target-gpu-mem-frac", type=float, default=0.30)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--aggregator-port", type=int, default=29765)
    return parser.parse_args()


def python_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(LOCAL_SRC)
        if not existing
        else f"{LOCAL_SRC}{os.pathsep}{existing}"
    )
    return env


def build_command(
    *,
    args: argparse.Namespace,
    run_name: str,
    metrics_file: Path,
    disabled: bool,
    repeat_index: int,
) -> list[str]:
    master_port = args.master_port + repeat_index * 10 + (0 if disabled else 1)
    aggregator_port = (
        args.aggregator_port + repeat_index * 10 + (0 if disabled else 1)
    )
    cmd = [
        sys.executable,
        "-c",
        LAUNCHER_SNIPPET,
        "run",
        str(args.workload),
        "--mode=summary",
        "--logs-dir",
        str(args.output_dir / "logs"),
        "--run-name",
        run_name,
        "--nproc-per-node",
        str(args.nproc_per_node),
        "--master-port",
        str(master_port),
        "--aggregator-port",
        str(aggregator_port),
    ]
    if disabled:
        cmd.append("--disable-traceml")

    cmd.extend(
        [
            "--args",
            "--steps",
            str(args.steps),
            "--warmup-steps",
            str(args.warmup_steps),
            "--batch-size",
            str(args.batch_size),
            "--input-dim",
            str(args.input_dim),
            "--hidden-dim",
            str(args.hidden_dim),
            "--hidden-layers",
            str(args.hidden_layers),
            "--output-dim",
            str(args.output_dim),
            "--target-gpu-mem-frac",
            str(args.target_gpu_mem_frac),
            "--metrics-file",
            str(metrics_file),
        ]
    )
    return cmd


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def run_trial(
    *,
    args: argparse.Namespace,
    repeat_index: int,
    mode: str,
) -> dict[str, Any]:
    disabled = mode == "native"
    run_name = f"overhead_r{repeat_index:02d}_{mode}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = run_dir / "workload_metrics.json"
    stdout_file = run_dir / "stdout_stderr.txt"

    cmd = build_command(
        args=args,
        run_name=run_name,
        metrics_file=metrics_file,
        disabled=disabled,
        repeat_index=repeat_index,
    )

    started = time.perf_counter()
    with stdout_file.open("w", encoding="utf-8") as out:
        out.write("$ " + " ".join(cmd) + "\n\n")
        out.flush()
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=python_env(),
            stdout=out,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed_s = time.perf_counter() - started

    metrics = load_json(metrics_file)
    final_summary = load_json(
        args.output_dir / "logs" / run_name / "final_summary.json"
    )
    primary_step_ms = None
    if metrics is not None:
        primary_step_ms = metrics.get("global", {}).get("primary_step_ms")

    return {
        "repeat": repeat_index,
        "mode": mode,
        "run_name": run_name,
        "returncode": completed.returncode,
        "elapsed_s": elapsed_s,
        "primary_step_ms": primary_step_ms,
        "metrics_file": str(metrics_file),
        "stdout_file": str(stdout_file),
        "final_summary_file": str(
            args.output_dir / "logs" / run_name / "final_summary.json"
        ),
        "has_final_summary": final_summary is not None,
    }


def paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_repeat: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_repeat.setdefault(int(row["repeat"]), {})[str(row["mode"])] = row

    pairs: list[dict[str, Any]] = []
    for repeat, modes in sorted(by_repeat.items()):
        native = modes.get("native")
        traceml = modes.get("traceml")
        if not native or not traceml:
            continue
        native_ms = native.get("primary_step_ms")
        traceml_ms = traceml.get("primary_step_ms")
        overhead_pct = None
        if (
            native.get("returncode") == 0
            and traceml.get("returncode") == 0
            and isinstance(native_ms, (int, float))
            and isinstance(traceml_ms, (int, float))
            and native_ms > 0
        ):
            overhead_pct = ((traceml_ms / native_ms) - 1.0) * 100.0
        pairs.append(
            {
                "repeat": repeat,
                "native_step_ms": native_ms,
                "traceml_step_ms": traceml_ms,
                "overhead_pct": overhead_pct,
                "native_returncode": native.get("returncode"),
                "traceml_returncode": traceml.get("returncode"),
            }
        )
    return pairs


def write_outputs(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runs.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    (output_dir / "pairs.json").write_text(
        json.dumps(pairs, indent=2),
        encoding="utf-8",
    )

    with (output_dir / "pairs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "repeat",
                "native_step_ms",
                "traceml_step_ms",
                "overhead_pct",
                "native_returncode",
                "traceml_returncode",
            ],
        )
        writer.writeheader()
        writer.writerows(pairs)

    overheads = [
        float(pair["overhead_pct"])
        for pair in pairs
        if isinstance(pair.get("overhead_pct"), (int, float))
    ]
    lines = ["# TraceML Overhead Results", ""]
    if overheads:
        lines.extend(
            [
                f"- successful pairs: {len(overheads)}/{len(pairs)}",
                f"- median overhead: {statistics.median(overheads):.2f}%",
                f"- min overhead: {min(overheads):.2f}%",
                f"- max overhead: {max(overheads):.2f}%",
                "",
            ]
        )
    else:
        lines.extend(["- successful pairs: 0", ""])

    lines.extend(
        [
            "| repeat | native step ms | TraceML step ms | overhead |",
            "|---:|---:|---:|---:|",
        ]
    )
    for pair in pairs:
        overhead = pair.get("overhead_pct")
        overhead_text = "" if overhead is None else f"{float(overhead):.2f}%"
        native_text = (
            ""
            if pair.get("native_step_ms") is None
            else f"{float(pair['native_step_ms']):.3f}"
        )
        traceml_text = (
            ""
            if pair.get("traceml_step_ms") is None
            else f"{float(pair['traceml_step_ms']):.3f}"
        )
        lines.append(
            f"| {pair['repeat']} | {native_text} | "
            f"{traceml_text} | {overhead_text} |"
        )
    (output_dir / "results.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.nproc_per_node <= 0:
        raise SystemExit("--nproc-per-node must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for repeat in range(1, args.repeats + 1):
        order = ["native", "traceml"]
        if repeat % 2 == 0:
            order.reverse()
        for mode in order:
            print(f"[benchmark] repeat={repeat} mode={mode}")
            row = run_trial(args=args, repeat_index=repeat, mode=mode)
            rows.append(row)
            print(
                "[benchmark] "
                f"{row['run_name']} rc={row['returncode']} "
                f"primary_step_ms={row['primary_step_ms']}"
            )

    pairs = paired_rows(rows)
    write_outputs(output_dir=args.output_dir, rows=rows, pairs=pairs)
    print(f"[benchmark] wrote {args.output_dir / 'results.md'}")


if __name__ == "__main__":
    main()
