"""
TraceML executor

This module is the execution wrapper used by TraceML to run user scripts
in a controlled environment.

Responsibilities:
- Read TraceML configuration from environment variables
- Start and stop the TraceML runtime lifecycle
- Execute the user script in-process via runpy
- Capture crashes and persist error reports to session log files

This module intentionally runs in the same Python process as the user
script so that hooks, stack traces, and execution context remain accurate.

Error reporting policy
----------------------
- User-script failures are written to: torchrun_error.log
- TraceML runtime / executor internal failures are written to: runtime_error.log

This keeps user-code failures separate from TraceML infrastructure failures
and avoids relying on terminal output that may be overwritten by the Rich UI.
"""

import os
import runpy
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from traceml.runtime.runtime import TraceMLRuntime
from traceml.runtime.settings import TraceMLSettings, TraceMLTCPSettings
from traceml.utils.shared_utils import EXECUTION_LAYER

INTERRUPTED_EXIT_CODE = 130
DEFAULT_LOGS_DIR = "./logs"
DEFAULT_TCP_HOST = "127.0.0.1"
DEFAULT_TCP_PORT = 29765
DEFAULT_PROFILE = "run"
DEFAULT_UI_MODE = "cli"
DEFAULT_INTERVAL_SEC = 1.0
DEFAULT_NUM_DISPLAY_LAYERS = 20
DEFAULT_REMOTE_MAX_ROWS = 200
USER_ERROR_LOG_NAME = "torchrun_error.log"
RUNTIME_ERROR_LOG_NAME = "runtime_error.log"


def _utc_now_iso() -> str:
    """Return the current UTC time as a string."""
    return datetime.now(timezone.utc).isoformat()


def _get_session_dir(cfg: Dict[str, Any]) -> Path:
    """
    Return the TraceML session directory for the current run.

    The directory is:
        <logs_dir>/<session_id>

    If no session id is available, a fallback directory name is used so that
    error logging can still succeed.
    """
    logs_dir = Path(str(cfg.get("logs_dir", DEFAULT_LOGS_DIR)))
    session_id = str(cfg.get("session_id", "") or "no_session")
    return logs_dir / session_id


def _append_error_log(
    cfg: Dict[str, Any],
    filename: str,
    header: str,
    error: Optional[BaseException] = None,
    include_execution_layer: bool = False,
) -> None:
    """
    Append an error report to a session log file.

    Parameters
    ----------
    cfg:
        Runtime configuration dictionary.
    filename:
        Target log filename under the session directory.
    header:
        Short human-readable header describing the failure.
    error:
        Optional exception to serialize as a traceback.
    include_execution_layer:
        Whether to include the last known execution layer in the report.

    Notes
    -----
    This function must never raise. Logging failures should not change
    the outcome of the user's run.
    """
    try:
        out_dir = _get_session_dir(cfg)
        out_dir.mkdir(parents=True, exist_ok=True)

        path = out_dir / filename
        with open(path, "a", encoding="utf-8", errors="replace") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"{_utc_now_iso()}  {header}\n")

            if include_execution_layer:
                current_execution_layer = getattr(
                    EXECUTION_LAYER, "current", None
                )
                if current_execution_layer is not None:
                    f.write(
                        "[TraceML] Last execution point: "
                        f"{current_execution_layer}\n"
                    )

            if error is not None:
                traceback.print_exception(
                    type(error),
                    error,
                    error.__traceback__,
                    file=f,
                )

            f.flush()
    except Exception:
        # Never break the user's run because best-effort logging failed.
        pass


def write_user_error_log(
    cfg: Dict[str, Any],
    header: str,
    error: Optional[BaseException] = None,
) -> None:
    """
    Append a user-script crash or exit report to torchrun_error.log.

    This log is reserved for failures originating from the executed user
    script, including unhandled exceptions, KeyboardInterrupt, and nonzero
    SystemExit outcomes.
    """
    _append_error_log(
        cfg=cfg,
        filename=USER_ERROR_LOG_NAME,
        header=header,
        error=error,
        include_execution_layer=True,
    )


def write_runtime_error_log(
    cfg: Dict[str, Any],
    header: str,
    error: Optional[BaseException] = None,
) -> None:
    """
    Append a TraceML internal runtime or executor failure report to
    runtime_error.log.

    This log is reserved for TraceML infrastructure problems such as runtime
    startup failure, runtime shutdown failure, and executor-internal errors.
    """
    _append_error_log(
        cfg=cfg,
        filename=RUNTIME_ERROR_LOG_NAME,
        header=header,
        error=error,
        include_execution_layer=False,
    )


class NoOpRuntime:
    """
    Fallback runtime used when TraceML is disabled or runtime startup fails.

    This preserves the executor control flow and avoids additional branching
    in the main execution path.
    """

    def start(self) -> None:
        """No-op start hook."""
        return None

    def stop(self) -> None:
        """No-op stop hook."""
        return None


def read_traceml_env() -> Dict[str, Any]:
    """
    Read TraceML configuration injected by the CLI launcher.

    All configuration is passed through environment variables so that:
    - user scripts remain untouched
    - the executor runs in a clean child process
    - runtime behavior is centrally controlled by the launcher

    Backward compatibility:
    - TRACEML_UI_MODE is preferred
    - TRACEML_MODE is still accepted

     Supported display modes:
    - cli
    - dashboard
    - summary
    """
    ui_mode = os.environ.get(
        "TRACEML_UI_MODE",
        os.environ.get("TRACEML_MODE", DEFAULT_UI_MODE),
    )

    return {
        "script_path": os.environ["TRACEML_SCRIPT_PATH"],
        "profile": os.environ.get("TRACEML_PROFILE", DEFAULT_PROFILE),
        "mode": ui_mode,
        "interval": float(
            os.environ.get("TRACEML_INTERVAL", str(DEFAULT_INTERVAL_SEC))
        ),
        "enable_logging": (
            os.environ.get("TRACEML_ENABLE_LOGGING", "0") == "1"
        ),
        "logs_dir": os.environ.get("TRACEML_LOGS_DIR", DEFAULT_LOGS_DIR),
        "num_display_layers": int(
            os.environ.get(
                "TRACEML_NUM_DISPLAY_LAYERS",
                str(DEFAULT_NUM_DISPLAY_LAYERS),
            )
        ),
        "tcp_host": os.environ.get("TRACEML_TCP_HOST", DEFAULT_TCP_HOST),
        "tcp_port": int(
            os.environ.get("TRACEML_TCP_PORT", str(DEFAULT_TCP_PORT))
        ),
        "remote_max_rows": int(
            os.environ.get(
                "TRACEML_REMOTE_MAX_ROWS",
                str(DEFAULT_REMOTE_MAX_ROWS),
            )
        ),
        "session_id": os.environ.get("TRACEML_SESSION_ID", ""),
        "disable_traceml": os.environ.get("TRACEML_DISABLED", "0") == "1",
    }


def extract_script_args() -> list[str]:
    """
    Extract arguments intended for the user script.

    Convention:
        traceml run train.py --args --epochs 10 --lr 1e-3

    Everything after '--' is forwarded to the target script.
    If '--' is absent, no extra script arguments are forwarded.
    """
    try:
        separator_index = sys.argv.index("--")
        return sys.argv[separator_index + 1 :]
    except ValueError:
        return []


def build_runtime_settings(cfg: Dict[str, Any]) -> TraceMLSettings:
    """
    Build TraceML runtime settings from the executor configuration.

    Keeping settings creation separate makes runtime startup easier to test
    and keeps the lifecycle code focused on orchestration.
    """
    return TraceMLSettings(
        mode=str(cfg["mode"]),
        profile=str(cfg["profile"]),
        sampler_interval_sec=float(cfg["interval"]),
        enable_logging=bool(cfg["enable_logging"]),
        logs_dir=str(cfg["logs_dir"]),
        session_id=str(cfg["session_id"]),
        tcp=TraceMLTCPSettings(
            host=str(cfg["tcp_host"]),
            port=int(cfg["tcp_port"]),
        ),
    )


def start_runtime(cfg: Dict[str, Any]) -> Union[TraceMLRuntime, NoOpRuntime]:
    """
    Initialize and start the TraceML runtime.

    Runtime startup must happen before user code executes so TraceML can:
    - attach hooks
    - start background samplers
    - observe early runtime behavior

    Returns
    -------
    TraceMLRuntime | NoOpRuntime
        A started TraceMLRuntime on success, otherwise a NoOpRuntime.

    Notes
    -----
    This function is fail-open by design. If TraceML cannot initialize, the
    user script still runs.
    """
    if bool(cfg.get("disable_traceml")):
        return NoOpRuntime()

    try:
        settings = build_runtime_settings(cfg)
        runtime = TraceMLRuntime(settings=settings)
        runtime.start()
        return runtime

    except Exception as error:
        write_runtime_error_log(
            cfg,
            header="Failed to start TraceMLRuntime",
            error=error,
        )
        return NoOpRuntime()


def stop_runtime(
    runtime: Union[TraceMLRuntime, NoOpRuntime],
    cfg: Dict[str, Any],
) -> None:
    """
    Stop the TraceML runtime.

    This is best-effort and must never raise. Runtime shutdown failures should
    not mask user-script failures or alter exit behavior.
    """
    try:
        runtime.stop()

        # Summaries are intentionally disabled here for now.
        # TODO: Re-enable once summary behavior is finalized.
        # runtime.log_summaries(path=None)

    except Exception as error:
        write_runtime_error_log(
            cfg,
            header="Error during TraceML runtime shutdown",
            error=error,
        )


def run_user_script(script_path: str, script_args: list[str]) -> None:
    """
    Execute the user script in-process using runpy.

    This intentionally does not spawn another subprocess so that:
    - hooks attach to the real Python objects
    - stack traces remain meaningful
    - execution context matches the user's script as closely as possible

    Global ``sys.argv`` and ``sys.path`` are restored afterward to avoid
    leaking state.
    """
    resolved_script_path = str(Path(script_path).resolve())
    script_dir = str(Path(resolved_script_path).parent)
    old_argv = sys.argv[:]
    old_path = sys.path[:]
    try:
        sys.argv = [resolved_script_path, *script_args]

        # Match normal ``python script.py`` semantics so sibling imports and
        # path-relative module discovery work the same way under TraceML.
        if sys.path:
            sys.path[0] = script_dir
        else:
            sys.path = [script_dir]

        runpy.run_path(resolved_script_path, run_name="__main__")
    finally:
        sys.argv = old_argv
        sys.path = old_path


def report_crash(cfg: Dict[str, Any], error: BaseException) -> None:
    """
    Persist an enriched user-script crash report.

    The crash report is written to torchrun_error.log and includes the last
    known execution layer when available.

    This function intentionally does not print large tracebacks to the terminal,
    because the aggregator may own the terminal UI and overwrite them.
    """
    write_user_error_log(
        cfg,
        header="Unhandled exception in user script",
        error=error,
    )


def _coerce_exit_code(code: Any) -> int:
    """
    Normalize SystemExit.code into a process exit code.

    Rules
    -----
    - None -> 0
    - int -> int
    - anything else -> 1
    """
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1


def main() -> None:
    """
    TraceML executor entrypoint.

    Execution flow
    --------------
    1. Read TraceML configuration from environment variables
    2. Start the TraceML runtime
    3. Execute the user script in-process
    4. Stop the runtime
    5. Persist enriched crash context if needed
    6. Exit with the user script's resulting exit code

    Error handling policy
    ---------------------
    - User script failures are written to torchrun_error.log
    - TraceML internal failures are written to runtime_error.log
    - Large error reports are not printed to the terminal because the
      aggregator may control the terminal UI
    """
    cfg = read_traceml_env()
    script_args = extract_script_args()

    runtime = start_runtime(cfg)

    exit_code = 0
    error: Optional[BaseException] = None

    try:
        run_user_script(str(cfg["script_path"]), script_args)

    except KeyboardInterrupt as interrupt_error:
        write_user_error_log(
            cfg,
            header="KeyboardInterrupt (Ctrl+C)",
            error=interrupt_error,
        )
        exit_code = INTERRUPTED_EXIT_CODE
        error = None

    except SystemExit as system_exit:
        exit_code = _coerce_exit_code(system_exit.code)

        if exit_code != 0:
            write_user_error_log(
                cfg,
                header=(
                    "User script exited via SystemExit " f"(code={exit_code})"
                ),
                error=system_exit,
            )

        error = None

    except Exception as unhandled_error:
        error = unhandled_error
        exit_code = 1

    finally:
        stop_runtime(runtime, cfg)

    if error is not None:
        report_crash(cfg, error)
        raise SystemExit(1)

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
