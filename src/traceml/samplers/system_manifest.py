"""
System manifest helpers for TraceML.

This module owns one-time system manifest generation and persistence so the
periodic `SystemSampler` can stay focused on runtime sampling.
"""

from __future__ import annotations

import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import psutil
from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetName,
    nvmlDeviceGetUUID,
)

from traceml.samplers.runtime_context import resolve_runtime_context
from traceml.samplers.utils import write_json_atomic


def _safe_call(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def build_system_manifest(
    *,
    cpu_logical_core_count: int,
    ram_total_memory: float,
    gpu_available: bool,
    gpu_count: int,
    logger: Any,
) -> Dict[str, Any]:
    """
    Build a one-time backend-side system manifest.

    The manifest reflects the actual machine where `SystemSampler` is running.
    Failures are tolerated and represented as best-effort defaults.
    """
    vm = _safe_call(psutil.virtual_memory)

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "created_at_unix": float(time.time()),
        "host": {
            "hostname": _safe_call(socket.gethostname, ""),
            "fqdn": _safe_call(socket.getfqdn, ""),
        },
        "platform": {
            "system": _safe_call(platform.system, ""),
            "release": _safe_call(platform.release, ""),
            "version": _safe_call(platform.version, ""),
            "machine": _safe_call(platform.machine, ""),
            "processor": _safe_call(platform.processor, ""),
            "platform": _safe_call(platform.platform, ""),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": {
                "major": int(sys.version_info.major),
                "minor": int(sys.version_info.minor),
                "micro": int(sys.version_info.micro),
            },
        },
        "process": {
            "pid": int(os.getpid()),
            "ppid": int(os.getppid()),
            "cwd": str(Path.cwd()),
        },
        "cpu": {
            "logical_core_count": int(cpu_logical_core_count or 0),
            "physical_core_count": int(
                _safe_call(lambda: psutil.cpu_count(logical=False), 0) or 0
            ),
            "current_freq_mhz": _safe_call(
                lambda: (
                    float(psutil.cpu_freq().current)
                    if psutil.cpu_freq() is not None
                    else None
                ),
                None,
            ),
            "max_freq_mhz": _safe_call(
                lambda: (
                    float(psutil.cpu_freq().max)
                    if psutil.cpu_freq() is not None
                    else None
                ),
                None,
            ),
        },
        "memory": {
            "ram_total_bytes": float(ram_total_memory or 0.0),
            "ram_available_bytes": (
                float(vm.available) if vm is not None else 0.0
            ),
        },
        "environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "pythonpath": os.environ.get("PYTHONPATH", ""),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
            "world_size": os.environ.get("WORLD_SIZE", ""),
            "rank": os.environ.get("RANK", ""),
            "local_rank": os.environ.get("LOCAL_RANK", ""),
            "master_addr": os.environ.get("MASTER_ADDR", ""),
            "master_port": os.environ.get("MASTER_PORT", ""),
            "traceml_session_id": os.environ.get("TRACEML_SESSION_ID", ""),
            "traceml_logs_dir": os.environ.get("TRACEML_LOGS_DIR", ""),
            "traceml_profile": os.environ.get("TRACEML_PROFILE", ""),
            "traceml_ui_mode": os.environ.get("TRACEML_UI_MODE", ""),
        },
        "gpu": {
            "available": bool(gpu_available),
            "count": int(gpu_count or 0),
            "devices": [],
        },
    }

    if gpu_available and gpu_count > 0:
        devices: List[Dict[str, Any]] = []
        for gpu_id in range(gpu_count):
            try:
                handle = nvmlDeviceGetHandleByIndex(gpu_id)
                mem = nvmlDeviceGetMemoryInfo(handle)

                name = _safe_call(lambda: nvmlDeviceGetName(handle), "")
                uuid = _safe_call(lambda: nvmlDeviceGetUUID(handle), "")

                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                if isinstance(uuid, bytes):
                    uuid = uuid.decode("utf-8", errors="replace")

                devices.append(
                    {
                        "index": int(gpu_id),
                        "name": str(name),
                        "uuid": str(uuid),
                        "memory_total_bytes": float(mem.total),
                    }
                )
            except Exception as e:
                logger.error(
                    f"[TraceML] GPU {gpu_id} manifest probe failed: {e}"
                )
                devices.append(
                    {
                        "index": int(gpu_id),
                        "name": "",
                        "uuid": "",
                        "memory_total_bytes": 0.0,
                    }
                )

        manifest["gpu"]["devices"] = devices

    return manifest


def write_system_manifest_if_missing(
    *,
    cpu_logical_core_count: int,
    ram_total_memory: float,
    gpu_available: bool,
    gpu_count: int,
    logger: Any,
) -> None:
    """
    Write a backend-side system manifest once per session.

    Safe behavior
    -------------
    - never raises
    - does nothing if session context is missing
    - does not overwrite an existing manifest
    """
    try:
        ctx = resolve_runtime_context()
        if not ctx.session_id or not str(ctx.logs_dir):
            logger.error(
                "[TraceML] System manifest skipped: missing TRACEML_LOGS_DIR or TRACEML_SESSION_ID"
            )
            return

        manifest_path = ctx.session_root / "system_manifest.json"
        if manifest_path.exists():
            return

        payload = build_system_manifest(
            cpu_logical_core_count=cpu_logical_core_count,
            ram_total_memory=ram_total_memory,
            gpu_available=gpu_available,
            gpu_count=gpu_count,
            logger=logger,
        )
        write_json_atomic(manifest_path, payload, sort_keys=True)

    except Exception as e:
        logger.error(f"[TraceML] Failed to write system manifest: {e}")
