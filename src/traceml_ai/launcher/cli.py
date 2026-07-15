# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Argparse wiring for the TraceML launcher."""

from __future__ import annotations

import argparse

from traceml_ai.launcher.commands import (
    run_compare,
    run_inspect,
    run_serve,
    run_view,
    run_with_tracing,
    validate_launch_args,
)
from traceml_ai.reporting.config import DEFAULT_SUMMARY_WINDOW_ROWS
from traceml_ai.runtime.settings import DEFAULT_FINALIZE_TIMEOUT_SEC


def _add_launch_args(parser: argparse.ArgumentParser) -> None:
    """Add shared launch arguments for TraceML run commands."""
    parser.add_argument(
        "script", help="Path to the target Python training script."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["cli", "dashboard", "summary"],
        help=(
            "TraceML mode. "
            "'summary': end-of-run report, supports single-node and "
            "multi-node multi-GPU. "
            "'cli' and 'dashboard': live views, intended for single-node "
            "runs, including single-node multi-GPU. "
            "'dashboard' requires the dashboard extra: "
            "pip install 'traceml-ai[dashboard]'. Default: cli on "
            "single-node runs, summary on multi-node runs."
        ),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--enable-logging",
        action="store_const",
        const=True,
        default=None,
        help="Enable TraceML logging output.",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=None,
        help="Directory for TraceML session logs.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help=(
            "Human-readable TraceML run name. Determines the output folder "
            "under --logs-dir. Required for multi-node runs unless "
            "--session-id is used."
        ),
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default="",
        help=(
            "Backward-compatible alias for --run-name. Existing scripts may "
            "continue to use this flag."
        ),
    )
    parser.add_argument(
        "--aggregator-port",
        type=int,
        default=29765,
        help="TraceML aggregator port.",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=None,
        help="Port for the dashboard web UI (mode=dashboard). Default: 8765.",
    )
    parser.add_argument(
        "--no-dashboard-auto-open",
        action="store_const",
        const=True,
        default=None,
        help=(
            "Do not auto-open a browser for the dashboard UI. Use on "
            "headless or remote machines."
        ),
    )
    parser.add_argument(
        "--summary-window-rows",
        type=int,
        default=DEFAULT_SUMMARY_WINDOW_ROWS,
        help=(
            "Rows used per node/rank for final summaries. SQLite retains "
            "1.5x this value for alignment. Default: "
            f"{DEFAULT_SUMMARY_WINDOW_ROWS}."
        ),
    )
    parser.add_argument(
        "--finalize-timeout-sec",
        type=float,
        default=None,
        help=(
            "Maximum seconds TraceML waits at end of run to settle telemetry, "
            "flush SQLite, checkpoint WAL, and write final summary artifacts. "
            f"Default: {DEFAULT_FINALIZE_TIMEOUT_SEC:g}."
        ),
    )
    parser.add_argument(
        "--trace-max-steps",
        type=int,
        default=None,
        help=(
            "Record TraceML telemetry for only the first N TraceML steps. "
            "Training continues after recording stops."
        ),
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="torchrun nproc_per_node value.",
    )
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="torchrun nnodes value.",
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="torchrun node_rank value for this machine.",
    )
    parser.add_argument(
        "--master-addr",
        type=str,
        default="127.0.0.1",
        help="torchrun master_addr value, usually node 0's address.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="torchrun master_port value.",
    )
    parser.add_argument(
        "--aggregator-host",
        type=str,
        default=None,
        help=(
            "Address workers use to send TraceML telemetry. Defaults to "
            "--master-addr."
        ),
    )
    parser.add_argument(
        "--aggregator-bind-host",
        type=str,
        default=None,
        help=(
            "Address the owner node binds the TraceML aggregator to. Use "
            "0.0.0.0 when other nodes must connect."
        ),
    )
    parser.add_argument(
        "--args",
        nargs=argparse.REMAINDER,
        help=(
            "Arguments forwarded to the target training script. "
            "Usage: traceml <watch|run> <script> --args <script args>"
        ),
    )
    parser.add_argument(
        "--no-history",
        action="store_const",
        const=True,
        default=None,
        help="Disable history saving (live view only; summaries and comparisons unavailable).",
    )
    parser.add_argument(
        "--disable-traceml",
        action="store_const",
        const=True,
        default=None,
        help="Disable TraceML telemetry and run the script natively via torchrun.",
    )
    parser.add_argument(
        "--html-report",
        action="store_true",
        help=(
            "Also write a self-contained final_summary.html next to the "
            "JSON/TXT artifacts at end of run."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level TraceML CLI parser."""
    parser = argparse.ArgumentParser(
        "traceml",
        description=(
            "Run TraceML around a training script.\n\n"
            "Examples:\n"
            "  traceml watch train.py\n"
            "  traceml run train.py --args --epochs 10 --lr 1e-3\n"
            "  traceml view logs/my_run/final_summary.json\n"
            "  traceml compare run_a.json run_b.json"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    watch_parser = sub.add_parser(
        "watch",
        help="Run a script in lightweight watch mode (system and process telemetry only).",
    )
    _add_launch_args(watch_parser)

    run_parser = sub.add_parser(
        "run",
        help="Run a script with TraceML bottleneck instrumentation.",
    )
    _add_launch_args(run_parser)

    serve_parser = sub.add_parser(
        "serve",
        help=(
            "Start the TraceML aggregator standalone. Use with scripts "
            "launched directly via `python` or `torchrun` that call "
            "traceml.init(...)."
        ),
    )
    serve_parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["cli", "dashboard", "summary"],
        help="TraceML display mode for the aggregator. Default: summary.",
    )
    serve_parser.add_argument(
        "--logs-dir",
        type=str,
        default=None,
        help="Directory for TraceML session logs.",
    )
    serve_parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help=(
            "Human-readable TraceML run name. Determines the output folder "
            "under --logs-dir. Workers must use the same run name."
        ),
    )
    serve_parser.add_argument(
        "--session-id",
        type=str,
        default="",
        help="Backward-compatible alias for --run-name.",
    )
    serve_parser.add_argument(
        "--nnodes",
        type=int,
        default=None,
        help=(
            "Total nodes in the training job. With --nproc-per-node, sets the "
            "expected worker count so the aggregator waits for ALL ranks "
            "before finalizing (and warns about ranks that never report). "
            "Without it the aggregator finalizes after the first rank."
        ),
    )
    serve_parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=None,
        help="Workers per node. Multiplied by --nnodes for the rank count.",
    )
    serve_parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Aggregator render/polling interval in seconds.",
    )
    serve_parser.add_argument(
        "--enable-logging",
        action="store_const",
        const=True,
        default=None,
        help="Enable TraceML logging output.",
    )
    serve_parser.add_argument(
        "--aggregator-host",
        type=str,
        default=None,
        help=(
            "Address workers use to connect to this aggregator. Defaults to "
            "127.0.0.1. Set to node 0's reachable address for multi-node."
        ),
    )
    serve_parser.add_argument(
        "--aggregator-bind-host",
        type=str,
        default=None,
        help=(
            "Address the aggregator binds to. Defaults to 127.0.0.1. Use "
            "0.0.0.0 when workers on other nodes must connect."
        ),
    )
    serve_parser.add_argument(
        "--aggregator-port",
        type=int,
        default=29765,
        help="TraceML aggregator port.",
    )

    compare_parser = sub.add_parser(
        "compare",
        help="Compare two TraceML final summary JSON files.",
    )
    compare_parser.add_argument(
        "left",
        help="Path to the left-hand TraceML final summary JSON file.",
    )
    compare_parser.add_argument(
        "right",
        help="Path to the right-hand TraceML final summary JSON file.",
    )
    compare_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional output base path. "
            "Writes both <base>.json and <base>.txt. "
            "Default: compare/<left>_vs_<right> in the current directory."
        ),
    )

    view_parser = sub.add_parser(
        "view",
        help="Print the stored text from a TraceML summary JSON file.",
    )
    view_parser.add_argument(
        "summary",
        help="Path to a TraceML summary JSON file.",
    )
    view_parser.add_argument(
        "--html",
        nargs="?",
        const="",
        default=None,
        metavar="OUT",
        help=(
            "Render an HTML report from the summary JSON instead of printing "
            "text. Optional OUT path; defaults to <summary>.html next to the "
            "input. Usage: traceml view <summary.json> --html [OUT]."
        ),
    )

    inspect_parser = sub.add_parser(
        "inspect", help="Inspect binary .msgpack logs."
    )
    inspect_parser.add_argument("file", help="Path to a .msgpack file.")

    return parser


def main() -> None:
    """CLI entrypoint for the TraceML launcher."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command in {"watch", "run"}:
        validate_launch_args(args)

    if args.command == "watch":
        run_with_tracing(args, profile="watch")
    elif args.command == "run":
        run_with_tracing(args, profile="run")
    elif args.command == "serve":
        run_serve(args)
    elif args.command == "compare":
        run_compare(args)
    elif args.command == "view":
        run_view(args)
    elif args.command == "inspect":
        run_inspect(args)
    else:
        parser.print_help()
        raise SystemExit(1)
