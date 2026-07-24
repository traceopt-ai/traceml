#!/usr/bin/env python3
"""Static source audit for the TraceML Phase 1 cost model."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class PatternHit:
    file: str
    line: int
    pattern: str
    text: str


HOOK_POINTS = [
    {
        "name": "trace_step context manager",
        "source": "src/traceml_ai/sdk/instrumentation.py",
        "function": "trace_step",
        "critical_path": True,
        "expected_cost": (
            "StepMemoryTracker construction/reset, whole-step timed_region, "
            "auto-timer context managers, optimizer-hook installation check, "
            "step advance, memory record, queue flush."
        ),
    },
    {
        "name": "DataLoader.__iter__ monkeypatch",
        "source": "src/traceml_ai/instrumentation/patches/dataloader_patch.py",
        "function": "patch_dataloader",
        "critical_path": True,
        "expected_cost": "One timed region around each DataLoader next().",
    },
    {
        "name": "nn.Module.__call__ monkeypatch",
        "source": (
            "src/traceml_ai/instrumentation/patches/"
            "forward_auto_timer_patch.py"
        ),
        "function": "patch_forward",
        "critical_path": True,
        "expected_cost": (
            "Global __call__ wrapper, TLS/depth checks, one outer forward "
            "timed region for the target model."
        ),
    },
    {
        "name": "Tensor.backward/autograd.backward monkeypatch",
        "source": (
            "src/traceml_ai/instrumentation/patches/"
            "backward_auto_timer_patch.py"
        ),
        "function": "patch_backward",
        "critical_path": True,
        "expected_cost": "Global backward wrapper and one timed region.",
    },
    {
        "name": "Tensor.to monkeypatch",
        "source": "src/traceml_ai/instrumentation/patches/h2d_auto_timer_patch.py",
        "function": "patch_h2d",
        "critical_path": True,
        "expected_cost": (
            "Global Tensor.to wrapper, TLS check, CPU-to-CUDA filter, timed "
            "region for H2D transfers."
        ),
    },
    {
        "name": "Optimizer step hooks",
        "source": "src/traceml_ai/instrumentation/hooks/optimizer_hooks.py",
        "function": "ensure_optimizer_timing_installed",
        "critical_path": True,
        "expected_cost": (
            "Global optimizer pre/post hook dispatch, optional CUDA event "
            "recording, TimeEvent queue append."
        ),
    },
    {
        "name": "Runtime sampler thread",
        "source": "src/traceml_ai/runtime/runtime.py",
        "function": "TraceMLRuntime._sampler_loop",
        "critical_path": False,
        "expected_cost": (
            "Off-thread sampler wakeups, queue drain, local DB aggregation, "
            "publisher flush."
        ),
    },
    {
        "name": "Step time sampler",
        "source": "src/traceml_ai/samplers/step_time_sampler.py",
        "function": "StepTimeSampler.sample",
        "critical_path": False,
        "expected_cost": (
            "Off-thread CUDA event query/elapsed_time, dict/defaultdict "
            "aggregation, pending deque retention until events resolve."
        ),
    },
    {
        "name": "TCP telemetry transport",
        "source": "src/traceml_ai/transport/tcp_transport.py",
        "function": "TCPClient.send_batch",
        "critical_path": False,
        "expected_cost": "Msgpack/json encoding and socket.sendall.",
    },
    {
        "name": "SQLite aggregator writer",
        "source": "src/traceml_ai/aggregator/sqlite_writer.py",
        "function": "SQLiteWriterSimple._loop",
        "critical_path": False,
        "expected_cost": (
            "Aggregator-process queue drain, projection conversion, SQLite "
            "transaction, retention pruning, WAL checkpoint at finalize."
        ),
    },
]

HOT_PATH_FILES = [
    "src/traceml_ai/sdk/instrumentation.py",
    "src/traceml_ai/utils/timing.py",
    "src/traceml_ai/utils/step_memory.py",
    "src/traceml_ai/utils/flush_buffers.py",
    "src/traceml_ai/instrumentation/patches/dataloader_patch.py",
    "src/traceml_ai/instrumentation/patches/forward_auto_timer_patch.py",
    "src/traceml_ai/instrumentation/patches/backward_auto_timer_patch.py",
    "src/traceml_ai/instrumentation/patches/h2d_auto_timer_patch.py",
    "src/traceml_ai/instrumentation/hooks/optimizer_hooks.py",
]

SYNC_RISK_PATTERNS = [
    ".item(",
    ".cpu(",
    ".numpy(",
    ".tolist(",
    "synchronize(",
    "torch.cuda.synchronize",
]

COST_PATTERNS = [
    "torch.cuda.Event",
    ".record(",
    ".query(",
    "elapsed_time",
    "reset_peak_memory_stats",
    "max_memory_allocated",
    "max_memory_reserved",
    "queue.put_nowait",
    "Queue(",
    "deque",
    "threading.Lock",
    "RLock",
    "psutil",
    "pynvml",
    "socket.",
    "sendall",
    "json.dumps",
    "sqlite3",
]


def _scan_patterns(
    repo_root: Path, files: list[str], patterns: list[str]
) -> list[PatternHit]:
    hits: list[PatternHit] = []
    for rel in files:
        path = repo_root / rel
        if not path.is_file():
            continue
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            for pattern in patterns:
                if pattern in line:
                    hits.append(
                        PatternHit(
                            file=rel,
                            line=line_no,
                            pattern=pattern,
                            text=line.strip(),
                        )
                    )
    return hits


def _write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# TraceML Phase 1 Static Cost Audit",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "## Instrumentation Points",
        "",
        "| Point | Source | Critical Path | Expected Cost |",
        "|---|---:|---:|---|",
    ]
    for point in payload["hook_points"]:
        lines.append(
            "| {name} | `{source}` / `{function}` | {critical_path} | {expected_cost} |".format(
                **point
            )
        )

    lines.extend(
        [
            "",
            "## CUDA Sync / D2H Risk Hits On Hot Path",
            "",
        ]
    )
    if payload["sync_risk_hits"]:
        lines.extend(
            ["| File | Line | Pattern | Source Text |", "|---|---:|---|---|"]
        )
        for hit in payload["sync_risk_hits"]:
            lines.append(
                f"| `{hit['file']}` | {hit['line']} | `{hit['pattern']}` | "
                f"`{hit['text'].replace('|', '&#124;')}` |"
            )
    else:
        lines.append(
            "No `.item()`, `.cpu()`, `.numpy()`, `.tolist()`, or explicit "
            "`torch.cuda.synchronize()` calls were found in the configured "
            "hot-path files."
        )

    lines.extend(
        [
            "",
            "## Cost-Relevant Pattern Hits",
            "",
            "| File | Line | Pattern | Source Text |",
            "|---|---:|---|---|",
        ]
    )
    for hit in payload["cost_pattern_hits"]:
        lines.append(
            f"| `{hit['file']}` | {hit['line']} | `{hit['pattern']}` | "
            f"`{hit['text'].replace('|', '&#124;')}` |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(repo_root: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    files_for_costs = sorted(
        set(
            HOT_PATH_FILES
            + [
                "src/traceml_ai/runtime/runtime.py",
                "src/traceml_ai/runtime/sender.py",
                "src/traceml_ai/database/database_sender.py",
                "src/traceml_ai/transport/tcp_transport.py",
                "src/traceml_ai/samplers/step_time_sampler.py",
                "src/traceml_ai/samplers/step_memory_sampler.py",
                "src/traceml_ai/samplers/system_sampler.py",
                "src/traceml_ai/samplers/process_sampler.py",
                "src/traceml_ai/aggregator/trace_aggregator.py",
                "src/traceml_ai/aggregator/sqlite_writer.py",
                "src/traceml_ai/aggregator/sqlite_writers/step_time.py",
                "src/traceml_ai/utils/msgpack_codec.py",
            ]
        )
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "hook_points": HOOK_POINTS,
        "hot_path_files": HOT_PATH_FILES,
        "sync_risk_patterns": SYNC_RISK_PATTERNS,
        "sync_risk_hits": [
            asdict(hit)
            for hit in _scan_patterns(
                repo_root, HOT_PATH_FILES, SYNC_RISK_PATTERNS
            )
        ],
        "cost_patterns": COST_PATTERNS,
        "cost_pattern_hits": [
            asdict(hit)
            for hit in _scan_patterns(
                repo_root, files_for_costs, COST_PATTERNS
            )
        ],
    }
    json_path = output_dir / "static_cost_audit.json"
    md_path = output_dir / "static_cost_audit.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_markdown(payload, md_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_audit(args.repo_root.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
