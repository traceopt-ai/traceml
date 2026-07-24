"""Run the Phase 3 benchmark with TraceML CLI + torchrun.

Single-node and multi-node runs use the same config-driven launcher. For
multi-node, start this script on every node with the same config, matching
--master-addr, and the node's own --node-rank.
"""

from __future__ import annotations

import argparse
import copy
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from common.env_capture import capture_environment
from common.io_utils import read_json, write_json, write_text
from common.launch_utils import (
    benchmark_env,
    command_text,
    run_capture,
    stable_cell_port,
)


WORKER = SCRIPT_DIR / "train_worker.py"
AGGREGATE = SCRIPT_DIR / "aggregate_results.py"


CELL_DEFS = {
    "never_init": {"trace_mode": "never_init", "launcher": "disabled"},
    "trace_manual": {"trace_mode": "trace_manual", "launcher": "enabled"},
    "trace_auto": {"trace_mode": "trace_auto", "launcher": "enabled"},
    "residual_hooks_optimizer_active": {
        "trace_mode": "residual_hooks_optimizer_active",
        "launcher": "disabled",
    },
    "trace_selective_h2d_only": {
        "trace_mode": "trace_selective_h2d_only",
        "launcher": "enabled",
    },
    "trace_selective_no_h2d": {
        "trace_mode": "trace_selective_no_h2d",
        "launcher": "enabled",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--node-rank", type=int, default=None)
    parser.add_argument("--master-addr", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--aggregate-after", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a 20-step, two-cell rendezvous smoke check; not publishable.",
    )
    return parser.parse_args()


def clean_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def require_config(config: dict[str, Any], args: argparse.Namespace) -> None:
    launch = config.get("launch", {})
    nnodes = int(launch.get("nnodes", 1))
    if nnodes > 1:
        if not int(launch.get("master_port", 0)):
            raise SystemExit("Multi-node config must set launch.master_port.")
        if not int(launch.get("aggregator_port", 0)):
            raise SystemExit(
                "Multi-node config must set launch.aggregator_port."
            )
        if not int(launch.get("barrier_port", 0)):
            raise SystemExit("Multi-node config must set launch.barrier_port.")
        if float(launch.get("rendezvous_timeout_sec", 0.0)) <= 0.0:
            raise SystemExit(
                "Multi-node config must set launch.rendezvous_timeout_sec."
            )
        run_id = args.run_id.strip() or str(config.get("run_id", "")).strip()
        if not run_id:
            raise SystemExit(
                "Multi-node config must set run_id or pass --run-id."
            )
        configured_addr = str(launch.get("master_addr", "")).strip()
        if not args.master_addr and configured_addr in {"", "<NODE0_IP>"}:
            raise SystemExit(
                "Multi-node runs must pass --master-addr <NODE0_IP>."
            )
        if args.keep_going:
            raise SystemExit(
                "--keep-going is not supported for multi-node benchmark runs."
            )
    cells = config.get("cells", [])
    bad = [cell for cell in cells if cell not in CELL_DEFS]
    if bad:
        raise SystemExit(f"Unknown cells in config: {bad}")
    for mode in config.get("timing_modes", []):
        if mode not in {"step", "phase"}:
            raise SystemExit(f"Unknown timing mode: {mode}")


def result_dir(config: dict[str, Any], args: argparse.Namespace) -> Path:
    output_root = (
        args.output_root
        or Path(
            str(
                config.get(
                    "output_root", "benchmarking/perf_benchmark/results"
                )
            )
        )
    ).resolve()
    run_id = (
        args.run_id.strip()
        or str(config.get("run_id", "")).strip()
        or datetime.now().strftime("phase3_%Y%m%d_%H%M%S")
    )
    path = output_root / run_id
    if path.exists() and any(path.iterdir()):
        raise SystemExit(
            f"Refusing to reuse non-empty result directory: {path}. "
            "Pass a new --run-id."
        )
    path.mkdir(parents=True, exist_ok=True)
    (path / "runs").mkdir(exist_ok=True)
    (path / "traceml-logs").mkdir(exist_ok=True)
    return path


def worker_args(
    config: dict[str, Any],
    *,
    workload: dict[str, Any],
    cell_name: str,
    trace_mode: str,
    timing_mode: str,
    run_dir: Path,
    repeat: int,
) -> list[str]:
    defaults = config.get("defaults", {})
    args = [
        "--output-dir",
        str(run_dir),
        "--cell-name",
        cell_name,
        "--trace-mode",
        trace_mode,
        "--timing-mode",
        timing_mode,
        "--steps",
        str(int(config.get("steps", defaults.get("steps", 1000)))),
        "--warmup",
        str(int(config.get("warmup", defaults.get("warmup", 100)))),
        "--model",
        str(workload["model"]),
        "--batch-size",
        str(int(workload["batch_size"])),
        "--dataloader",
        str(workload.get("dataloader", "synthetic")),
        "--num-classes",
        str(
            int(workload.get("num_classes", defaults.get("num_classes", 1000)))
        ),
        "--seq-len",
        str(int(workload.get("seq_len", defaults.get("seq_len", 128)))),
        "--vocab-size",
        str(
            int(workload.get("vocab_size", defaults.get("vocab_size", 50257)))
        ),
        "--realistic-num-workers",
        str(
            int(
                workload.get(
                    "realistic_num_workers",
                    defaults.get("realistic_num_workers", 2),
                )
            )
        ),
        "--device",
        str(config.get("device", "auto")),
        "--seed",
        str(int(config.get("seed", 1337)) + repeat * 7919),
        "--inter-step-sleep-ms",
        str(float(config.get("inter_step_sleep_ms", 0.0))),
        "--gil-inner-iters",
        str(int(config.get("gil_inner_iters", 1000))),
        "--gil-max-samples",
        str(int(config.get("gil_max_samples", 200000))),
        "--collector-interval-sec",
        str(float(config.get("launch", {}).get("interval", 1.0))),
    ]
    for key, flag in (
        ("input_dim", "--input-dim"),
        ("hidden_dim", "--hidden-dim"),
        ("layers", "--layers"),
    ):
        if workload.get(key) is not None:
            args.extend([flag, str(workload[key])])
    if bool(workload.get("pin_memory", defaults.get("pin_memory", True))):
        args.append("--pin-memory")
    if bool(config.get("require_cuda", True)):
        args.append("--require-cuda")
    # Opt-in only: GilVictim is an active GIL-contention stress injector
    # (unthrottled CPU loop), not a passive probe. Running it by default
    # contaminated the 2026-07-21 campaign's baselines (~1.4ms/step became
    # ~221ms/step); see PR #230 discussion.
    if bool(config.get("gil_probe", False)):
        args.append("--gil-probe")
    return args


def build_command(
    config: dict[str, Any],
    *,
    args: argparse.Namespace,
    case_index: int,
    repeat: int,
    workload: dict[str, Any],
    cell_name: str,
    timing_mode: str,
    run_dir: Path,
    results_dir: Path,
) -> list[str]:
    launch = config.get("launch", {})
    cell = CELL_DEFS[cell_name]
    nnodes = int(launch.get("nnodes", 1))
    nproc = int(launch.get("nproc_per_node", 1))
    node_rank = int(
        args.node_rank
        if args.node_rank is not None
        else launch.get("node_rank", 0)
    )
    master_addr = args.master_addr or str(
        launch.get("master_addr", "127.0.0.1")
    )
    master_port = stable_cell_port(int(launch["master_port"]), case_index)
    aggregator_port = stable_cell_port(
        int(launch["aggregator_port"]), case_index
    )
    session = clean_name(
        f"{cell_name}_{timing_mode}_{workload['model']}_{workload.get('dataloader', 'synthetic')}_r{repeat:02d}"
    )

    cmd = [
        sys.executable,
        "-c",
        "from traceml_ai.launcher.cli import main; main()",
        "run",
        str(WORKER),
        "--mode",
        "summary",
        "--logs-dir",
        str(results_dir / "traceml-logs"),
        "--run-name",
        session,
        "--nproc-per-node",
        str(nproc),
        "--nnodes",
        str(nnodes),
        "--node-rank",
        str(node_rank),
        "--master-addr",
        master_addr,
        "--master-port",
        str(master_port),
        "--aggregator-port",
        str(aggregator_port),
        "--finalize-timeout-sec",
        str(float(launch.get("finalize_timeout_sec", 300.0))),
        "--summary-window-rows",
        str(int(launch.get("summary_window_rows", 2000))),
    ]
    if launch.get("aggregator_host"):
        cmd.extend(["--aggregator-host", str(launch["aggregator_host"])])
    elif nnodes > 1:
        cmd.extend(["--aggregator-host", master_addr])
    if launch.get("aggregator_bind_host"):
        cmd.extend(
            ["--aggregator-bind-host", str(launch["aggregator_bind_host"])]
        )
    elif nnodes > 1 and node_rank == 0:
        cmd.extend(["--aggregator-bind-host", "0.0.0.0"])
    cmd.extend(["--interval", str(float(launch.get("interval", 1.0)))])
    if cell["launcher"] == "disabled":
        cmd.append("--disable-traceml")
    cmd.extend(
        [
            "--args",
            *worker_args(
                config,
                workload=workload,
                cell_name=cell_name,
                trace_mode=str(cell["trace_mode"]),
                timing_mode=timing_mode,
                run_dir=run_dir,
                repeat=repeat,
            ),
        ]
    )
    return cmd


def cases(
    config: dict[str, Any]
) -> list[tuple[str, str, dict[str, Any], int]]:
    items = []
    repeats = int(config.get("repeats", 3))
    for timing in config.get("timing_modes", ["step", "phase"]):
        for workload in config.get("workloads", []):
            for cell_name in config.get("cells", []):
                for repeat in range(repeats):
                    items.append((cell_name, timing, workload, repeat))
    return items


def quick_config(config: dict[str, Any]) -> dict[str, Any]:
    """Create the smallest useful multi-node connectivity and artifact smoke run."""
    quick = copy.deepcopy(config)
    quick["steps"] = 20
    quick["warmup"] = 5
    quick["repeats"] = 1
    quick["timing_modes"] = ["step"]
    quick["cells"] = ["never_init", "trace_auto"]
    quick["workloads"] = quick.get("workloads", [])[:1]
    quick["run_id"] = f"{quick.get('run_id', 'phase3')}_quick"
    return quick


def cross_node_barrier(
    config: dict[str, Any], args: argparse.Namespace, case_index: int
) -> float:
    """Keep nodes on the same case; timeout before torchrun can block for hours."""
    launch = config.get("launch", {})
    nnodes = int(launch.get("nnodes", 1))
    if nnodes <= 1:
        return 0.0
    node_rank = int(
        args.node_rank
        if args.node_rank is not None
        else launch.get("node_rank", 0)
    )
    host = args.master_addr or str(launch["master_addr"])
    port = stable_cell_port(int(launch["barrier_port"]), case_index)
    timeout = float(launch["rendezvous_timeout_sec"])
    deadline = time.monotonic() + timeout
    started = time.monotonic()
    token = f"{case_index}:{node_rank}".encode("ascii")

    if node_rank == 0:
        peers: list[socket.socket] = []
        seen = {0}
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind(("0.0.0.0", port))
                server.listen(nnodes - 1)
                while len(seen) < nnodes:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("timed out waiting for peer nodes")
                    server.settimeout(remaining)
                    conn, _ = server.accept()
                    conn.settimeout(remaining)
                    message = conn.recv(64).decode("ascii").strip()
                    try:
                        peer_case, peer_rank = (
                            int(value) for value in message.split(":")
                        )
                    except ValueError as exc:
                        conn.close()
                        raise RuntimeError(
                            f"invalid peer barrier token: {message!r}"
                        ) from exc
                    if (
                        peer_case != case_index
                        or peer_rank in seen
                        or not 0 < peer_rank < nnodes
                    ):
                        conn.close()
                        raise RuntimeError(
                            f"unexpected peer barrier token: {message!r}"
                        )
                    seen.add(peer_rank)
                    peers.append(conn)
                for peer in peers:
                    peer.sendall(b"ready\n")
            return time.monotonic() - started
        finally:
            for peer in peers:
                peer.close()

    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            remaining = max(0.1, deadline - time.monotonic())
            with socket.create_connection(
                (host, port), timeout=min(2.0, remaining)
            ) as conn:
                conn.settimeout(remaining)
                conn.sendall(token + b"\n")
                if conn.recv(16) == b"ready\n":
                    return time.monotonic() - started
                raise RuntimeError(
                    "barrier owner closed without releasing this node"
                )
        except OSError as exc:
            last_error = exc
            time.sleep(0.2)
    raise TimeoutError(
        f"timed out joining node-0 barrier at {host}:{port} after {timeout:.0f}s"
    ) from last_error


def verify_local_rank_outputs(
    run_dir: Path, expected_local_ranks: int
) -> None:
    rank_files = sorted(run_dir.glob("rank_*.json"))
    if len(rank_files) < expected_local_ranks:
        raise RuntimeError(
            f"Expected at least {expected_local_ranks} rank files in {run_dir}, "
            f"found {len(rank_files)}."
        )
    for path in rank_files:
        payload = read_json(path)
        expected = int(payload["args"]["steps"]) + int(
            payload["args"]["warmup"]
        )
        if len(payload.get("records", [])) != expected:
            raise RuntimeError(f"{path} has a mismatched step count.")


def run_from_config(config: dict[str, Any], args: argparse.Namespace) -> Path:
    require_config(config, args)
    results_dir = result_dir(config, args)
    capture_environment(results_dir, REPO_ROOT)
    env = benchmark_env(REPO_ROOT)
    launch = config.get("launch", {})
    nproc = int(launch.get("nproc_per_node", 1))
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "results_dir": str(results_dir),
        "runs": [],
    }
    plan = cases(config)
    for case_index, (cell_name, timing, workload, repeat) in enumerate(plan):
        run_dir = (
            results_dir
            / "runs"
            / clean_name(
                f"{cell_name}_{timing}_{workload['model']}_{workload.get('dataloader', 'synthetic')}"
            )
            / f"repeat_{repeat:02d}"
        )
        barrier_wait_sec = cross_node_barrier(config, args, case_index)
        run_dir.mkdir(parents=True, exist_ok=False)
        cmd = build_command(
            config,
            args=args,
            case_index=case_index,
            repeat=repeat,
            workload=workload,
            cell_name=cell_name,
            timing_mode=timing,
            run_dir=run_dir,
            results_dir=results_dir,
        )
        print(f"[phase3-runner] {command_text(cmd)}", flush=True)
        result = run_capture(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            timeout=float(launch.get("per_cell_timeout_sec", 3600.0)),
        )
        write_text(run_dir / "launcher_stdout.txt", result["stdout"])
        write_text(run_dir / "launcher_stderr.txt", result["stderr"])
        entry = {
            "cell": cell_name,
            "timing_mode": timing,
            "workload": workload,
            "repeat": repeat,
            "run_dir": str(run_dir),
            "command": cmd,
            "returncode": result["returncode"],
            "wall_time_sec": result["wall_time_sec"],
            "timed_out": result["timed_out"],
            "barrier_wait_sec": barrier_wait_sec,
        }
        manifest["runs"].append(entry)
        write_json(results_dir / "suite_manifest.json", manifest)
        if result["returncode"] != 0:
            print(result["stdout"][-4000:], file=sys.stdout)
            print(result["stderr"][-4000:], file=sys.stderr)
            if not args.keep_going:
                raise SystemExit(
                    f"Case failed: {cell_name}/{timing}/{workload['model']}/r{repeat}"
                )
            continue
        verify_local_rank_outputs(run_dir, nproc)

    if args.aggregate_after:
        subprocess.run(
            [
                sys.executable,
                str(AGGREGATE),
                "--results-dir",
                str(results_dir),
            ],
            cwd=str(REPO_ROOT),
            check=True,
        )
    return results_dir


def main() -> int:
    args = parse_args()
    config = read_json(args.config.resolve())
    if args.quick:
        config = quick_config(config)
        if args.run_id:
            args.run_id = f"{args.run_id}_quick"
    results_dir = run_from_config(config, args)
    print(f"[phase3-runner] done: {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
