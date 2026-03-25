"""
TraceML aggregator process entrypoint.

This module runs the TraceMLAggregator as a standalone process. It reads the
same TRACEML_* environment variables as the executor, starts the TCP server
and UI loop, and shuts down gracefully on SIGTERM/SIGINT.

Expected usage (via CLI)
------------------------
1. Start this process first (aggregator) and wait until it is listening.
2. Then start torchrun workers that run ``traceml/runtime/executor.py``.

Error handling
--------------
- Fatal aggregator errors are logged through the configured logger.
- Fatal aggregator errors are also written to ``aggregator_error.log`` under
  the current session directory.
- A brief error is printed to stderr as a last-resort fallback in case the
  terminal UI has already been torn down or failed.
"""

import json
import os
import platform
import signal
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import psutil
import pynvml

from traceml.aggregator.trace_aggregator import TraceMLAggregator
from traceml.loggers.error_log import get_error_logger, setup_error_logger
from traceml.runtime.settings import TraceMLSettings, TraceMLTCPSettings

AGGREGATOR_ERROR_LOG_NAME = "aggregator_error.log"


def _utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def write_aggregator_error_log(
    session_root: Path,
    header: str,
    error: BaseException,
) -> None:
    """
    Append a fatal aggregator error report to ``aggregator_error.log``.

    This function is best-effort and must never raise. Aggregator failures
    should still surface through stderr and exit codes even if file logging
    fails.
    """
    try:
        session_root.mkdir(parents=True, exist_ok=True)
        path = session_root / AGGREGATOR_ERROR_LOG_NAME

        with open(path, "a", encoding="utf-8", errors="replace") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"{_utc_now_iso()}  {header}\n")
            traceback.print_exception(
                type(error),
                error,
                error.__traceback__,
                file=f,
            )
            f.flush()
    except Exception:
        # Best-effort only: never mask the original failure.
        pass


def read_traceml_env() -> dict[str, Any]:
    """
    Read aggregator configuration from environment variables.

    The aggregator uses the same TRACEML_* variables as the executor/CLI so the
    launcher can configure both components consistently.

    Backward compatibility
    ----------------------
    - ``TRACEML_UI_MODE`` is preferred
    - ``TRACEML_MODE`` is still accepted
    """

    ui_mode = os.environ.get(
        "TRACEML_UI_MODE",
        os.environ.get("TRACEML_MODE", "cli"),
    )

    return {
        "mode": ui_mode,
        "profile": os.environ.get("TRACEML_PROFILE", "run"),
        "interval": float(os.environ.get("TRACEML_INTERVAL", "1.0")),
        "enable_logging": os.environ.get("TRACEML_ENABLE_LOGGING", "0") == "1",
        "logs_dir": os.environ.get("TRACEML_LOGS_DIR", "./logs"),
        "num_display_layers": int(
            os.environ.get("TRACEML_NUM_DISPLAY_LAYERS", "20")
        ),
        "tcp_host": os.environ.get("TRACEML_TCP_HOST", "127.0.0.1"),
        "tcp_port": int(os.environ.get("TRACEML_TCP_PORT", "29765")),
        "remote_max_rows": int(
            os.environ.get("TRACEML_REMOTE_MAX_ROWS", "200")
        ),
        "session_id": os.environ.get("TRACEML_SESSION_ID", ""),
        "history_enabled": os.environ.get("TRACEML_HISTORY_ENABLED", "1")
        == "1",
    }


def _install_signal_handlers(stop_event: threading.Event) -> None:
    """
    Install SIGINT/SIGTERM handlers that request aggregator shutdown.

    The handler is intentionally minimal and only signals the main loop to stop.
    """

    def _handler(signum: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def collect_cpu_info() -> dict:
    return {
        "architecture": platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "cpu_usage_percent": psutil.cpu_percent(interval=0.1),
    }


def save_cpu_info(path: Path) -> None:
    with open(path, "w") as f:
        json.dump(collect_cpu_info(), f, indent=2)


def collect_gpu_info() -> list:
    gpus = []
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name_val = pynvml.nvmlDeviceGetName(h)
            name_str = (
                name_val.decode() if isinstance(name_val, bytes) else name_val
            )
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            gpus.append(
                {
                    "index": i,
                    "name": name_str,
                    "memory_total_mb": mem.total // 1024**2,
                    "memory_used_mb": mem.used // 1024**2,
                    "gpu_util_percent": util.gpu,
                    "mem_util_percent": util.memory,
                    "temperature_c": pynvml.nvmlDeviceGetTemperature(
                        h, pynvml.NVML_TEMPERATURE_GPU
                    ),
                    "power_w": pynvml.nvmlDeviceGetPowerUsage(h) / 1000,
                    "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(
                        h, pynvml.NVML_CLOCK_SM
                    ),
                    "mem_clock_mhz": pynvml.nvmlDeviceGetClockInfo(
                        h, pynvml.NVML_CLOCK_MEM
                    ),
                    "perf_state": pynvml.nvmlDeviceGetPerformanceState(h),
                }
            )
        pynvml.nvmlShutdown()
    except Exception as e:
        return {"error": str(e)}
    return gpus


def save_gpu_info(path: Path) -> None:
    with open(path, "w") as f:
        json.dump(collect_gpu_info(), f, indent=2)


def collect_system_manifest(nproc_per_node: int = 1) -> dict:
    """Collect a unified hardware snapshot used by the heuristics engine."""
    cpu_info = collect_cpu_info()
    ram = psutil.virtual_memory()

    gpus = []
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name_val = pynvml.nvmlDeviceGetName(h)
            name_str = (
                name_val.decode() if isinstance(name_val, bytes) else name_val
            )
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            sm_clock = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
            mem_clock = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
            try:
                cc = pynvml.nvmlDeviceGetCudaComputeCapability(h)
                compute_capability = f"{cc[0]}.{cc[1]}"
            except Exception:
                compute_capability = None
            gpus.append(
                {
                    "index": i,
                    "name": name_str,
                    "memory_total_gb": round(mem.total / 1024**3, 2),
                    "memory_used_gb": round(mem.used / 1024**3, 2),
                    "compute_capability": compute_capability,
                    "sm_clock_mhz": sm_clock,
                    "mem_clock_mhz": mem_clock,
                }
            )
        pynvml.nvmlShutdown()
    except Exception:
        pass

    return {
        "schema_version": 1,
        "generated_at": _utc_now_iso(),
        "cpu": {
            "model": platform.processor(),
            "architecture": cpu_info["architecture"],
            "physical_cores": cpu_info["physical_cores"],
            "logical_cores": cpu_info["logical_cores"],
        },
        "ram": {
            "total_gb": round(ram.total / 1024**3, 2),
            "available_gb": round(ram.available / 1024**3, 2),
        },
        "gpus": gpus,
        "nproc_per_node": nproc_per_node,
    }


def save_system_manifest(path: Path, nproc_per_node: int = 1) -> None:
    with open(path, "w") as f:
        json.dump(collect_system_manifest(nproc_per_node), f, indent=2)


def main() -> None:
    """
    Aggregator process entrypoint.

    Execution flow
    --------------
    1. Initialize logging
    2. Read configuration from environment variables
    3. Create the session directory and settings
    4. Start the aggregator
    5. Wait for a shutdown request
    6. Stop the aggregator and report any fatal errors
    """
    setup_error_logger(is_aggregator=True)
    logger = get_error_logger("TraceMLAggregatorMain")

    cfg = read_traceml_env()

    session_id = str(cfg["session_id"] or "default")
    session_root = Path(str(cfg["logs_dir"])).resolve() / session_id
    session_dir = session_root / "aggregator"
    session_dir.mkdir(parents=True, exist_ok=True)
    db_path = session_dir / "telemetry"
    nproc_per_node = int(os.environ.get("TRACEML_NPROC_PER_NODE", "1"))
    save_cpu_info(session_dir / "cpu_info.json")
    save_gpu_info(session_dir / "gpu_info.json")
    save_system_manifest(session_dir / "system_manifest.json", nproc_per_node)
    stop_event = threading.Event()
    _install_signal_handlers(stop_event)

    agg: Optional[TraceMLAggregator] = None
    err: Optional[BaseException] = None

    try:
        settings = TraceMLSettings(
            mode=str(cfg["mode"]),
            profile=str(cfg["profile"]),
            render_interval_sec=float(cfg["interval"]),
            num_display_layers=int(cfg["num_display_layers"]),
            enable_logging=bool(cfg["enable_logging"]),
            logs_dir=str(cfg["logs_dir"]),
            remote_max_rows=int(cfg["remote_max_rows"]),
            session_id=session_id,
            history_enabled=bool(cfg["history_enabled"]),
            tcp=TraceMLTCPSettings(
                host=str(cfg["tcp_host"]),
                port=int(cfg["tcp_port"]),
            ),
            db_path=str(db_path),
        )

        agg = TraceMLAggregator(
            logger=logger,
            stop_event=stop_event,
            settings=settings,
        )
        logger.info("[TraceML] Starting aggregator")
        agg.start()

        stop_event.wait()

    except BaseException as exc:
        err = exc

    finally:
        if agg is not None:
            try:
                logger.info("[TraceML] Stopping aggregator")
                agg.stop(timeout_sec=5.0)
            except Exception:
                pass

        print(
            f"[TraceML] Logs saved under: {session_root}",
            file=sys.stderr,
            flush=True,
        )

        if err is not None:
            try:
                logger.exception("[TraceML] Aggregator exiting due to error")
            except Exception:
                pass

            write_aggregator_error_log(
                session_root=session_root,
                header="Fatal aggregator error",
                error=err,
            )

            print(
                "\n[TraceML] Aggregator exiting due to error. "
                f"See {session_root / AGGREGATOR_ERROR_LOG_NAME}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exception(
                type(err),
                err,
                err.__traceback__,
                file=sys.stderr,
            )
            sys.stderr.flush()
            raise SystemExit(1)

    print("\n[TraceML] Aggregator stopped.", file=sys.stderr, flush=True)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
