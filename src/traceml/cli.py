"""Public console entrypoint for TraceML.

The launcher implementation lives under :mod:`traceml.launcher`.  This module
stays intentionally small so packaging entrypoints and existing developer
imports can continue to target ``traceml.cli`` without pulling launcher
concerns back into one large file.
"""

from traceml.launcher.cli import build_parser, main
from traceml.launcher.commands import (
    launch_process,
    resolve_existing_script_path,
    run_compare,
    run_inspect,
    run_with_tracing,
    validate_launch_args,
)
from traceml.launcher.manifest import (
    collect_existing_artifacts,
    load_json_or_warn,
    update_run_manifest,
    utc_now_iso,
    write_code_manifest,
    write_json_atomic,
    write_run_manifest,
)
from traceml.launcher.process import (
    DEFAULT_SHUTDOWN_TIMEOUT_SEC,
    DEFAULT_TCP_READY_TIMEOUT_SEC,
    INTERRUPTED_EXIT_CODE,
    build_torchrun_base_cmd,
    install_shutdown_handlers,
    start_aggregator_process,
    start_training_process,
    terminate_process_group,
    wait_for_tcp_listen,
)

__all__ = [
    "DEFAULT_SHUTDOWN_TIMEOUT_SEC",
    "DEFAULT_TCP_READY_TIMEOUT_SEC",
    "INTERRUPTED_EXIT_CODE",
    "build_parser",
    "build_torchrun_base_cmd",
    "collect_existing_artifacts",
    "install_shutdown_handlers",
    "launch_process",
    "load_json_or_warn",
    "main",
    "resolve_existing_script_path",
    "run_compare",
    "run_inspect",
    "run_with_tracing",
    "start_aggregator_process",
    "start_training_process",
    "terminate_process_group",
    "update_run_manifest",
    "utc_now_iso",
    "validate_launch_args",
    "wait_for_tcp_listen",
    "write_code_manifest",
    "write_json_atomic",
    "write_run_manifest",
]

# Historical private helper names are kept as aliases inside this public
# entrypoint module. New code should import from ``traceml.launcher.*``.
_build_torchrun_base_cmd = build_torchrun_base_cmd
_collect_existing_artifacts = collect_existing_artifacts
_load_json_or_warn = load_json_or_warn
_resolve_existing_script_path = resolve_existing_script_path
_utc_now_iso = utc_now_iso
_validate_launch_args = validate_launch_args
_write_json_atomic = write_json_atomic


if __name__ == "__main__":
    main()
