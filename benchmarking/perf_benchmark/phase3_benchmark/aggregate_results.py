"""Aggregate Phase 3 outputs into JSON, CSV, and one public Markdown report."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from common.io_utils import write_json
from common.stats import (
    baseline_noise_floor,
    fmt,
    overhead_label,
    pct,
    summary_stats,
)


PHASE_KEYS = [
    "trace_context_enter_ms",
    "dataloader_ms",
    "h2d_ms",
    "zero_grad_ms",
    "forward_ms",
    "backward_ms",
    "optimizer_step_ms",
    "trace_context_exit_ms",
    "inter_step_idle_ms",
    "total_step_ms",
]


def load_rank_files(results_dir: Path) -> list[dict]:
    payloads = []
    for path in sorted((results_dir / "runs").glob("*/repeat_*/rank_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_path"] = str(path)
        payload["_repeat"] = Path(path).parent.name
        payloads.append(payload)
    return payloads


def workload_key(payload: dict) -> str:
    model = payload["model"]
    return json.dumps(
        {
            "model": model.get("name"),
            "batch_size": model.get("batch_size"),
            "dataloader": model.get("dataloader"),
            "input_dim": model.get("input_dim"),
            "hidden_dim": model.get("hidden_dim"),
            "layers": model.get("layers"),
            "num_classes": model.get("num_classes"),
            "seq_len": model.get("seq_len"),
            "vocab_size": model.get("vocab_size"),
        },
        sort_keys=True,
    )


def group_payloads(
    payloads: list[dict],
) -> dict[tuple[str, str, str], list[dict]]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for payload in payloads:
        timing = str(payload.get("timing_mode", "phase"))
        grouped[(payload["cell_name"], timing, workload_key(payload))].append(
            payload
        )
    return grouped


def collect_phase_values(payloads: list[dict], phase: str) -> list[float]:
    values: list[float] = []
    for payload in payloads:
        for record in payload.get("records", []):
            if record.get("is_warmup"):
                continue
            values.append(float(record["phases_ms"][phase]))
    return values


def collect_repeat_medians(payloads: list[dict], phase: str) -> list[dict]:
    """Summarize one independent process repeat at a time.

    The primary rows intentionally pool all measured samples. This companion
    view keeps each ``repeat_XX`` separate so a report can expose variation
    across fresh process launches rather than implying that every step sample
    is an independent benchmark replicate.
    """
    by_repeat: dict[str, list[float]] = defaultdict(list)
    for payload in payloads:
        repeat = str(payload.get("_repeat", "repeat_unknown"))
        by_repeat[repeat].extend(
            float(record["phases_ms"][phase])
            for record in payload.get("records", [])
            if not record.get("is_warmup")
        )
    return [
        {"repeat": repeat, **summary_stats(values)}
        for repeat, values in sorted(by_repeat.items())
    ]


def collect_rank_rows(payloads: list[dict], phase: str) -> list[dict]:
    rows = []
    for payload in payloads:
        rank = int(payload["rank"]["rank"])
        values = [
            float(record["phases_ms"][phase])
            for record in payload.get("records", [])
            if not record.get("is_warmup")
        ]
        rows.append({"rank": rank, **summary_stats(values)})
    return sorted(rows, key=lambda row: row["rank"])


def collect_skew_values(payloads: list[dict], phase: str) -> list[float]:
    by_step: dict[tuple[str, int], list[float]] = defaultdict(list)
    for payload in payloads:
        repeat = str(payload.get("_repeat", "repeat_unknown"))
        for record in payload.get("records", []):
            if record.get("is_warmup"):
                continue
            measured = record.get("measured_step_index")
            if measured is None:
                continue
            by_step[(repeat, int(measured))].append(
                float(record["phases_ms"][phase])
            )
    return [
        max(values) - min(values)
        for values in by_step.values()
        if len(values) >= 2
    ]


def collect_gil_values(payloads: list[dict]) -> list[float]:
    values: list[float] = []
    for payload in payloads:
        probe = payload.get("gil_probe", {})
        if not probe.get("enabled"):
            continue
        for sample in probe.get("retained_samples", []):
            values.append(float(sample["chunk_ms"]))
    return values


def collect_network_rows(payloads: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for payload in payloads:
        network = payload.get("network", {})
        if not network.get("available"):
            continue
        rows.append(
            {
                "repeat": str(payload.get("_repeat", "repeat_unknown")),
                "rank": int(payload["rank"]["rank"]),
                "interface": network.get("interface"),
                "scope": network.get("scope"),
                "rx_bytes": int(network["rx_bytes"]),
                "tx_bytes": int(network["tx_bytes"]),
                "total_bytes": int(network["total_bytes"]),
                "estimated_bytes_per_collector_interval": network.get(
                    "estimated_bytes_per_collector_interval"
                ),
            }
        )
    return rows


def sqlite_row_counts(db_path: Path) -> dict[str, int]:
    if not db_path.is_file():
        return {}
    tables = [
        "step_time_samples",
        "step_memory_samples",
        "process_samples",
        "system_samples",
        "system_gpu_samples",
        "runtime_environment_samples",
        "stdout_stderr_samples",
    ]
    counts: dict[str, int] = {}
    try:
        with sqlite3.connect(str(db_path)) as conn:
            existing = {
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table';"
                ).fetchall()
            }
            for table in tables:
                if table in existing:
                    counts[table] = int(
                        conn.execute(
                            f"SELECT COUNT(*) FROM {table};"
                        ).fetchone()[0]
                    )
    except sqlite3.Error as exc:
        counts["sqlite_error"] = str(exc)  # type: ignore[assignment]
    return counts


def artifact_metrics(results_dir: Path) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    logs_dir = results_dir / "traceml-logs"
    if not logs_dir.exists():
        return metrics
    for session_dir in sorted(
        path for path in logs_dir.iterdir() if path.is_dir()
    ):
        total = 0
        files = 0
        json_bytes = 0
        for root, _, filenames in os.walk(session_dir):
            for filename in filenames:
                path = Path(root) / filename
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                total += size
                files += 1
                if filename.endswith(".json"):
                    json_bytes += size
        db_path = session_dir / "aggregator" / "telemetry"
        metrics[session_dir.name] = {
            "artifact_total_bytes": total,
            "artifact_file_count": files,
            "json_artifact_bytes": json_bytes,
            "telemetry_sqlite_bytes": (
                db_path.stat().st_size if db_path.exists() else 0
            ),
            "sqlite_row_counts": sqlite_row_counts(db_path),
            "final_summary_present": (
                session_dir / "final_summary.json"
            ).is_file(),
        }
    return metrics


def build_summary(results_dir: Path) -> dict:
    payloads = load_rank_files(results_dir)
    grouped = group_payloads(payloads)
    baselines: dict[tuple[str, str], dict[str, dict]] = {}
    network_baselines: dict[tuple[str, str, str, int], int] = {}
    for (cell_name, timing, key), group in grouped.items():
        if cell_name.startswith("never_init"):
            baselines[(timing, key)] = {
                phase: summary_stats(collect_phase_values(group, phase))
                for phase in PHASE_KEYS
            }
            for network_row in collect_network_rows(group):
                network_baselines[
                    (timing, key, network_row["repeat"], network_row["rank"])
                ] = network_row["total_bytes"]

    rows = []
    repeat_rows = []
    repeat_summary_rows = []
    rank_rows = []
    skew_rows = []
    gil_rows = []
    network_rows = []
    for (cell_name, timing, key), group in sorted(grouped.items()):
        workload = json.loads(key)
        baseline = baselines.get((timing, key), {})
        ranks = sorted({int(payload["rank"]["rank"]) for payload in group})
        repeats = sorted({str(payload.get("_repeat")) for payload in group})
        for phase in PHASE_KEYS:
            current = summary_stats(collect_phase_values(group, phase))
            base = baseline.get(phase, {})
            base_median = base.get("median_ms")
            median = current.get("median_ms")
            delta = (
                float(median) - float(base_median)
                if median is not None and base_median is not None
                else None
            )
            noise = baseline_noise_floor(base)
            rows.append(
                {
                    "cell_name": cell_name,
                    "timing_mode": timing,
                    "phase": phase,
                    "workload": workload,
                    "ranks": ranks,
                    "repeat_count": len(repeats),
                    **current,
                    "baseline_median_ms": base_median,
                    "overhead_median_ms": delta,
                    "overhead_median_pct": pct(delta, base_median),
                    "baseline_noise_floor_ms": noise,
                    "within_baseline_noise": (
                        delta is not None
                        and noise is not None
                        and abs(delta) <= float(noise)
                    ),
                }
            )
            per_repeat = collect_repeat_medians(group, phase)
            repeat_medians = [row["median_ms"] for row in per_repeat]
            if repeat_medians:
                repeat_stats = summary_stats(repeat_medians)
                repeat_summary_rows.append(
                    {
                        "cell_name": cell_name,
                        "timing_mode": timing,
                        "phase": phase,
                        "workload": workload,
                        "repeat_count": len(per_repeat),
                        "median_of_repeat_medians_ms": repeat_stats[
                            "median_ms"
                        ],
                        "p95_of_repeat_medians_ms": repeat_stats["p95_ms"],
                        "p99_of_repeat_medians_ms": repeat_stats["p99_ms"],
                        "mean_of_repeat_medians_ms": repeat_stats["mean_ms"],
                        "std_of_repeat_medians_ms": repeat_stats["std_ms"],
                        "min_of_repeat_medians_ms": repeat_stats["min_ms"],
                        "max_of_repeat_medians_ms": repeat_stats["max_ms"],
                    }
                )
                for repeat_row in per_repeat:
                    repeat_rows.append(
                        {
                            "cell_name": cell_name,
                            "timing_mode": timing,
                            "phase": phase,
                            "workload": workload,
                            "repeat": repeat_row["repeat"],
                            "sample_count": repeat_row["n"],
                            "repeat_median_ms": repeat_row["median_ms"],
                            "repeat_p95_ms": repeat_row["p95_ms"],
                            "repeat_p99_ms": repeat_row["p99_ms"],
                        }
                    )
            for rank_row in collect_rank_rows(group, phase):
                rank_rows.append(
                    {
                        "cell_name": cell_name,
                        "timing_mode": timing,
                        "phase": phase,
                        "workload": workload,
                        **rank_row,
                    }
                )
        skew = collect_skew_values(group, "total_step_ms")
        if skew:
            skew_rows.append(
                {
                    "cell_name": cell_name,
                    "timing_mode": timing,
                    "phase": "total_step_ms",
                    "workload": workload,
                    **summary_stats(skew),
                }
            )
        gil = collect_gil_values(group) if timing == "step" else []
        if gil:
            gil_rows.append(
                {
                    "cell_name": cell_name,
                    "timing_mode": timing,
                    "workload": workload,
                    **summary_stats(gil),
                }
            )
        for network_row in collect_network_rows(group):
            baseline_total = network_baselines.get(
                (timing, key, network_row["repeat"], network_row["rank"])
            )
            network_rows.append(
                {
                    "cell_name": cell_name,
                    "timing_mode": timing,
                    "workload": workload,
                    "baseline_total_bytes": baseline_total,
                    "overhead_total_bytes": (
                        network_row["total_bytes"] - baseline_total
                        if baseline_total is not None
                        else None
                    ),
                    **network_row,
                }
            )

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(results_dir),
        "rank_file_count": len(payloads),
        "rows": rows,
        "repeat_rows": repeat_rows,
        "repeat_summary_rows": repeat_summary_rows,
        "rank_rows": rank_rows,
        "skew_rows": skew_rows,
        "gil_rows": gil_rows,
        "network_rows": network_rows,
        "artifact_metrics": artifact_metrics(results_dir),
    }


def write_csv(summary: dict, path: Path) -> None:
    fieldnames = [
        "cell_name",
        "timing_mode",
        "phase",
        "model",
        "batch_size",
        "dataloader",
        "repeat_count",
        "ranks",
        "n",
        "median_ms",
        "p95_ms",
        "p99_ms",
        "std_ms",
        "baseline_median_ms",
        "overhead_median_ms",
        "overhead_median_pct",
        "baseline_noise_floor_ms",
        "within_baseline_noise",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["rows"]:
            workload = row["workload"]
            writer.writerow(
                {
                    "cell_name": row["cell_name"],
                    "timing_mode": row["timing_mode"],
                    "phase": row["phase"],
                    "model": workload.get("model"),
                    "batch_size": workload.get("batch_size"),
                    "dataloader": workload.get("dataloader"),
                    "repeat_count": row["repeat_count"],
                    "ranks": ",".join(str(rank) for rank in row["ranks"]),
                    "n": row["n"],
                    "median_ms": row["median_ms"],
                    "p95_ms": row["p95_ms"],
                    "p99_ms": row["p99_ms"],
                    "std_ms": row["std_ms"],
                    "baseline_median_ms": row["baseline_median_ms"],
                    "overhead_median_ms": row["overhead_median_ms"],
                    "overhead_median_pct": row["overhead_median_pct"],
                    "baseline_noise_floor_ms": row["baseline_noise_floor_ms"],
                    "within_baseline_noise": row["within_baseline_noise"],
                }
            )


def write_repeat_csv(summary: dict, path: Path) -> None:
    fieldnames = [
        "cell_name",
        "timing_mode",
        "phase",
        "model",
        "batch_size",
        "dataloader",
        "repeat",
        "sample_count",
        "repeat_median_ms",
        "repeat_p95_ms",
        "repeat_p99_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["repeat_rows"]:
            workload = row["workload"]
            writer.writerow(
                {
                    "cell_name": row["cell_name"],
                    "timing_mode": row["timing_mode"],
                    "phase": row["phase"],
                    "model": workload.get("model"),
                    "batch_size": workload.get("batch_size"),
                    "dataloader": workload.get("dataloader"),
                    "repeat": row["repeat"],
                    "sample_count": row["sample_count"],
                    "repeat_median_ms": row["repeat_median_ms"],
                    "repeat_p95_ms": row["repeat_p95_ms"],
                    "repeat_p99_ms": row["repeat_p99_ms"],
                }
            )


def workload_label(workload: dict) -> str:
    return (
        f"{workload.get('model')} bs={workload.get('batch_size')} "
        f"loader={workload.get('dataloader')}"
    )


def write_report(summary: dict, results_dir: Path, path: Path) -> None:
    total_rows = [
        row for row in summary["rows"] if row["phase"] == "total_step_ms"
    ]
    phase_rows = [
        row
        for row in summary["rows"]
        if row["timing_mode"] == "phase"
        and row["phase"]
        in {
            "trace_context_enter_ms",
            "dataloader_ms",
            "h2d_ms",
            "forward_ms",
            "backward_ms",
            "optimizer_step_ms",
            "trace_context_exit_ms",
        }
    ]
    lines = [
        "# TraceML Runtime Overhead Benchmark",
        "",
        f"Generated: `{summary['generated_at_utc']}`",
        f"Results directory: `{results_dir}`",
        "",
        "This self-run benchmark is reproducible from the public scripts and raw per-rank JSON.",
        "Deltas inside baseline noise are reported as bounds, not zero.",
        "",
        "## Total Step",
        "",
        "| Cell | Timing | Workload | Ranks | N | Median ms | p95 ms | p99 ms | Overhead Bound |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in total_rows:
        lines.append(
            "| {cell} | {timing} | {workload} | {ranks} | {n} | {median} | "
            "{p95} | {p99} | {overhead} |".format(
                cell=row["cell_name"],
                timing=row["timing_mode"],
                workload=workload_label(row["workload"]),
                ranks=",".join(str(rank) for rank in row["ranks"]),
                n=row["n"],
                median=fmt(row["median_ms"]),
                p95=fmt(row["p95_ms"]),
                p99=fmt(row["p99_ms"]),
                overhead=overhead_label(row),
            )
        )

    total_repeat_rows = [
        row
        for row in summary["repeat_summary_rows"]
        if row["phase"] == "total_step_ms"
    ]
    if total_repeat_rows:
        lines.extend(
            [
                "",
                "## Independent Repeat Medians — Total Step",
                "",
                "Each value below is the median from one fresh process "
                "repeat. The table summarizes variation across those "
                "independent repeat medians, rather than across pooled "
                "per-step samples.",
                "",
                "| Cell | Timing | Workload | Repeats | Median of repeat medians (ms) | Std dev (ms) | Min (ms) | Max (ms) |",
                "|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in total_repeat_rows:
            lines.append(
                "| {cell} | {timing} | {workload} | {count} | {median} | "
                "{std} | {minimum} | {maximum} |".format(
                    cell=row["cell_name"],
                    timing=row["timing_mode"],
                    workload=workload_label(row["workload"]),
                    count=row["repeat_count"],
                    median=fmt(row["median_of_repeat_medians_ms"]),
                    std=fmt(row["std_of_repeat_medians_ms"]),
                    minimum=fmt(row["min_of_repeat_medians_ms"]),
                    maximum=fmt(row["max_of_repeat_medians_ms"]),
                )
            )

    lines.extend(
        [
            "",
            "## Phase Attribution",
            "",
            "| Cell | Phase | Workload | Median ms | p95 ms | p99 ms | Overhead Bound |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in phase_rows:
        lines.append(
            "| {cell} | `{phase}` | {workload} | {median} | {p95} | {p99} | "
            "{overhead} |".format(
                cell=row["cell_name"],
                phase=row["phase"],
                workload=workload_label(row["workload"]),
                median=fmt(row["median_ms"]),
                p95=fmt(row["p95_ms"]),
                p99=fmt(row["p99_ms"]),
                overhead=overhead_label(row),
            )
        )

    if summary["skew_rows"]:
        lines.extend(
            [
                "",
                "## Rank Skew",
                "",
                "| Cell | Timing | Workload | N | Median skew ms | p95 ms | p99 ms |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in summary["skew_rows"]:
            lines.append(
                "| {cell} | {timing} | {workload} | {n} | {median} | {p95} | {p99} |".format(
                    cell=row["cell_name"],
                    timing=row["timing_mode"],
                    workload=workload_label(row["workload"]),
                    n=row["n"],
                    median=fmt(row["median_ms"]),
                    p95=fmt(row["p95_ms"]),
                    p99=fmt(row["p99_ms"]),
                )
            )

    if summary["gil_rows"]:
        lines.extend(
            [
                "",
                "## GIL Victim Probe",
                "",
                "| Cell | Timing | Workload | Samples | Median ms | p95 ms | p99 ms |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in summary["gil_rows"]:
            lines.append(
                "| {cell} | {timing} | {workload} | {n} | {median} | {p95} | {p99} |".format(
                    cell=row["cell_name"],
                    timing=row["timing_mode"],
                    workload=workload_label(row["workload"]),
                    n=row["n"],
                    median=fmt(row["median_ms"]),
                    p95=fmt(row["p95_ms"]),
                    p99=fmt(row["p99_ms"]),
                )
            )

    if summary["network_rows"]:
        lines.extend(
            [
                "",
                "## Network Counters",
                "",
                "Node-interface bytes include DDP and TraceML traffic. For one-rank-per-node runs, rows are rank-scoped; compare cells with `never_init` for TraceML-attributable deltas.",
                "",
                "| Cell | Timing | Repeat | Rank | Interface | RX bytes | TX bytes | Delta vs baseline bytes | Est. bytes / collector interval |",
                "|---|---|---|---:|---|---:|---:|---:|---:|",
            ]
        )
        for row in summary["network_rows"]:
            lines.append(
                "| {cell} | {timing} | {repeat} | {rank} | {interface} | {rx} | {tx} | {delta} | {per_interval} |".format(
                    cell=row["cell_name"],
                    timing=row["timing_mode"],
                    repeat=row["repeat"],
                    rank=row["rank"],
                    interface=row["interface"],
                    rx=row["rx_bytes"],
                    tx=row["tx_bytes"],
                    delta=row["overhead_total_bytes"],
                    per_interval=fmt(
                        row["estimated_bytes_per_collector_interval"]
                    ),
                )
            )

    if summary["artifact_metrics"]:
        lines.extend(
            [
                "",
                "## TraceML Artifacts",
                "",
                "| Session | Total bytes | Telemetry DB bytes | Step rows | Process rows | System rows | Final summary |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for session, metrics in sorted(summary["artifact_metrics"].items()):
            counts = metrics.get("sqlite_row_counts", {})
            lines.append(
                "| {session} | {total} | {sqlite} | {step} | {process} | {system} | {final} |".format(
                    session=session,
                    total=metrics["artifact_total_bytes"],
                    sqlite=metrics["telemetry_sqlite_bytes"],
                    step=counts.get("step_time_samples", 0),
                    process=counts.get("process_samples", 0),
                    system=counts.get("system_samples", 0),
                    final=(
                        "yes" if metrics.get("final_summary_present") else "no"
                    ),
                )
            )

    lines.extend(
        [
            "",
            "## Raw Artifacts",
            "",
            "- Raw rank files: `runs/<cell>/repeat_<n>/rank_<rank>.json`",
            "- Aggregate JSON: `summary.json`",
            "- Flat table: `summary.csv`",
            "- Per-repeat medians: `repeat_medians.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    summary = build_summary(results_dir)
    write_json(results_dir / "summary.json", summary)
    write_csv(summary, results_dir / "summary.csv")
    write_repeat_csv(summary, results_dir / "repeat_medians.csv")
    write_report(summary, results_dir, results_dir / "report.md")
    print(f"[phase3-aggregate] wrote {results_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
