"""Argparse wiring for the TraceML launcher."""

from __future__ import annotations

import argparse

from traceml.launcher.commands import (
    run_compare,
    run_inspect,
    run_with_tracing,
    validate_launch_args,
)


def _add_launch_args(parser: argparse.ArgumentParser) -> None:
    """Add shared launch arguments for TraceML run commands."""
    parser.add_argument(
        "script", help="Path to the target Python training script."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="cli",
        choices=["cli", "dashboard", "summary"],
        help=(
            "TraceML display mode to launch. "
            "Use 'summary' for final-summary-only runs. Default: cli."
        ),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--enable-logging",
        action="store_true",
        help="Enable TraceML logging output.",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="./logs",
        help="Directory for TraceML session logs.",
    )
    parser.add_argument(
        "--num-display-layers",
        type=int,
        default=5,
        help="Maximum number of model layers to display in the live UI.",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default="",
        help="Optional explicit session id.",
    )
    parser.add_argument(
        "--tcp-host",
        type=str,
        default="127.0.0.1",
        help="Aggregator bind host.",
    )
    parser.add_argument(
        "--tcp-port", type=int, default=29765, help="Aggregator bind port."
    )
    parser.add_argument(
        "--remote-max-rows",
        type=int,
        default=200,
        help="Maximum number of rows returned by remote telemetry queries.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="torchrun nproc_per_node value.",
    )
    parser.add_argument(
        "--args",
        nargs=argparse.REMAINDER,
        help=(
            "Arguments forwarded to the target training script. "
            "Usage: traceml <watch|run|deep> <script> --args <script args>"
        ),
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable history saving (live view only; summaries and comparisons unavailable).",
    )
    parser.add_argument(
        "--disable-traceml",
        action="store_true",
        help="Disable TraceML telemetry and run the script natively via torchrun.",
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
            "  traceml deep train.py --args --config config.yaml\n"
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

    deep_parser = sub.add_parser(
        "deep",
        help="Run a script with TraceML deep layerwise instrumentation.",
    )
    _add_launch_args(deep_parser)

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

    inspect_parser = sub.add_parser(
        "inspect", help="Inspect binary .msgpack logs."
    )
    inspect_parser.add_argument("file", help="Path to a .msgpack file.")

    return parser


def main() -> None:
    """CLI entrypoint for the TraceML launcher."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command in {"watch", "run", "deep"}:
        validate_launch_args(args)

    if args.command == "watch":
        run_with_tracing(args, profile="watch")
    elif args.command == "run":
        run_with_tracing(args, profile="run")
    elif args.command == "deep":
        run_with_tracing(args, profile="deep")
    elif args.command == "compare":
        run_compare(args)
    elif args.command == "inspect":
        run_inspect(args)
    else:
        parser.print_help()
        raise SystemExit(1)
