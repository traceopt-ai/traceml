# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Bounded window aggregation for external telemetry export.

Raw rank telemetry and SQLite history are unchanged. This module reduces only
the optional external stream: step records use semantic-step windows, periodic
records use source-time windows, and runtime context passes through unchanged.
No cross-rank synchronization or diagnosis is performed here.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from traceml_ai.telemetry_export.records import ExportRecord, RecordKind

DEFAULT_STEP_WINDOW = 10
DEFAULT_TIME_WINDOW_SEC = 10.0
STEP_WINDOW_ENV = "TRACEML_OTLP_STEP_WINDOW"
TIME_WINDOW_ENV = "TRACEML_OTLP_TIME_WINDOW_SEC"

_Group = tuple[str, ...]


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _integer(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _boolean(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def _string(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _warn(logger: Optional[Any], message: str, *args: Any) -> None:
    log = getattr(logger, "warning", None)
    if callable(log):
        log(message, *args)


def _env_value(
    name: str,
    default: int | float,
    cast: type[int] | type[float],
    *,
    logger: Optional[Any],
) -> int | float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = cast(raw)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError
        if value <= 0:
            raise ValueError
        return value
    except (TypeError, ValueError):
        _warn(
            logger,
            "[TraceML] %s must be positive; using %s",
            name,
            default,
        )
        return default


@dataclass
class NumericStats:
    """Online count/sum/min/max for finite numeric observations."""

    count: int = 0
    sum: float = 0.0
    min: Optional[float] = None
    max: Optional[float] = None

    def add(self, value: Any) -> None:
        number = _number(value)
        if number is None:
            return
        self.count += 1
        self.sum += number
        self.min = number if self.min is None else min(self.min, number)
        self.max = number if self.max is None else max(self.max, number)

    def to_dict(self) -> Optional[dict[str, Any]]:
        if self.count == 0 or self.min is None or self.max is None:
            return None
        return {
            "count": self.count,
            "sum": float(self.sum),
            "min": float(self.min),
            "max": float(self.max),
        }


@dataclass
class _Window:
    """Generic state shared by all four window record kinds."""

    kind: RecordKind
    resource: dict[str, Any]
    start: int
    end: int
    samples: int = 0
    steps: set[int] = field(default_factory=set)
    start_time_ns: Optional[int] = None
    end_time_ns: Optional[int] = None
    observed_time_ns: int = 0
    stats: dict[_Group, dict[str, NumericStats]] = field(default_factory=dict)
    latest: dict[_Group, dict[str, Any]] = field(default_factory=dict)

    def observe(
        self,
        record: ExportRecord,
        *,
        step: Optional[int] = None,
    ) -> None:
        self.samples += 1
        if step is not None:
            self.steps.add(int(step))
        timestamp = record.timestamp_unix_ns
        if timestamp is not None and int(timestamp) >= 0:
            timestamp = int(timestamp)
            self.start_time_ns = (
                timestamp
                if self.start_time_ns is None
                else min(self.start_time_ns, timestamp)
            )
            self.end_time_ns = (
                timestamp
                if self.end_time_ns is None
                else max(self.end_time_ns, timestamp)
            )
        self.observed_time_ns = max(
            self.observed_time_ns,
            int(record.observed_timestamp_unix_ns),
        )

    def add(self, group: _Group, field_name: str, value: Any) -> None:
        group_stats = self.stats.setdefault(group, {})
        group_stats.setdefault(field_name, NumericStats()).add(value)

    def keep(self, group: _Group, field_name: str, value: Any) -> None:
        if value is not None:
            self.latest.setdefault(group, {})[field_name] = value

    def record(self) -> ExportRecord:
        data = _build_data(self)
        if self.start_time_ns is not None:
            data["start_time_unix_ns"] = self.start_time_ns
        if self.end_time_ns is not None:
            data["end_time_unix_ns"] = self.end_time_ns
        return ExportRecord(
            kind=self.kind,
            timestamp_unix_ns=self.end_time_ns,
            observed_timestamp_unix_ns=self.observed_time_ns,
            resource=dict(self.resource),
            data=data,
        )


def _group_data(window: _Window, group: _Group) -> dict[str, Any]:
    data = dict(window.latest.get(group, {}))
    for name, stats in window.stats.get(group, {}).items():
        value = stats.to_dict()
        if value is not None:
            data[name] = value
    return data


def _step_data(window: _Window) -> dict[str, Any]:
    return {
        "start_step": window.start,
        "end_step": window.end,
        "step_count": len(window.steps),
    }


def _build_timing(window: _Window) -> dict[str, Any]:
    data = _step_data(window)
    groups = set(window.stats) | set(window.latest)
    phases: list[dict[str, Any]] = []
    for group in sorted(groups):
        if len(group) != 3 or group[0] != "phase":
            continue
        values = _group_data(window, group)
        if values:
            phases.append({"phase": group[1], "device": group[2], **values})
    data["phases"] = phases
    return data


def _build_step_memory(window: _Window) -> dict[str, Any]:
    data = _step_data(window)
    devices: list[dict[str, Any]] = []
    for group in sorted(window.stats):
        if len(group) != 2 or group[0] != "device":
            continue
        values = _group_data(window, group)
        if values:
            devices.append({"device": group[1], **values})
    data["devices"] = devices
    return data


def _build_process(window: _Window) -> dict[str, Any]:
    data: dict[str, Any] = {"sample_count": window.samples}
    for section in ("cpu", "memory", "gpu"):
        values = _group_data(window, (section,))
        if values:
            data[section] = values
    return data


def _build_system(window: _Window) -> dict[str, Any]:
    data = _build_process(window)
    devices: list[dict[str, Any]] = []
    groups = set(window.stats) | set(window.latest)
    for group in sorted(groups):
        if len(group) != 2 or group[0] != "gpu_device":
            continue
        values = _group_data(window, group)
        devices.append({"device_index": int(group[1]), **values})
    if devices:
        data.setdefault("gpu", {})["devices"] = devices
    elif "gpu" in data:
        data["gpu"]["devices"] = []
    return data


def _build_data(window: _Window) -> dict[str, Any]:
    if window.kind is RecordKind.STEP_TIMING_WINDOW:
        return _build_timing(window)
    if window.kind is RecordKind.STEP_MEMORY_WINDOW:
        return _build_step_memory(window)
    if window.kind is RecordKind.PROCESS_WINDOW:
        return _build_process(window)
    if window.kind is RecordKind.SYSTEM_WINDOW:
        return _build_system(window)
    return {}


class WindowProcessor:
    """Aggregate normalized records into bounded export windows.

    ``process`` is the only ingestion method. It returns completed windows;
    ``flush`` returns partial windows during clean shutdown. The aggregator
    owns this object, so it needs no thread or internal queue.
    """

    def __init__(
        self,
        *,
        step_window: int = DEFAULT_STEP_WINDOW,
        time_window_sec: float = DEFAULT_TIME_WINDOW_SEC,
        logger: Optional[Any] = None,
    ) -> None:
        if (
            isinstance(step_window, bool)
            or not isinstance(step_window, int)
            or step_window <= 0
        ):
            raise ValueError("step_window must be a positive integer")
        if (
            isinstance(time_window_sec, bool)
            or not math.isfinite(float(time_window_sec))
            or float(time_window_sec) <= 0.0
        ):
            raise ValueError(
                "time_window_sec must be a finite positive number"
            )
        self.step_window = step_window
        self.time_window_sec = float(time_window_sec)
        self._time_window_ns = max(1, int(self.time_window_sec * 1e9))
        self._logger = logger
        self._windows: dict[tuple[RecordKind, str], _Window] = {}

    @classmethod
    def from_env(cls, *, logger: Optional[Any] = None) -> "WindowProcessor":
        """Build a processor from the documented OTLP window settings."""
        return cls(
            step_window=int(
                _env_value(
                    STEP_WINDOW_ENV,
                    DEFAULT_STEP_WINDOW,
                    int,
                    logger=logger,
                )
            ),
            time_window_sec=float(
                _env_value(
                    TIME_WINDOW_ENV,
                    DEFAULT_TIME_WINDOW_SEC,
                    float,
                    logger=logger,
                )
            ),
            logger=logger,
        )

    def process(self, records: Sequence[ExportRecord]) -> list[ExportRecord]:
        """Consume source records and return newly completed windows."""
        completed: list[ExportRecord] = []
        for record in records:
            if record.kind is RecordKind.STEP_TIMING_WINDOW:
                completed.extend(self._process_timing(record))
            elif record.kind is RecordKind.STEP_MEMORY_WINDOW:
                completed.extend(self._process_step_memory(record))
            elif record.kind is RecordKind.PROCESS_WINDOW:
                completed.extend(self._process_process(record))
            elif record.kind is RecordKind.SYSTEM_WINDOW:
                completed.extend(self._process_system(record))
            else:
                completed.append(record)
        return completed

    def flush(self) -> list[ExportRecord]:
        """Return all partial windows and clear processor state."""
        records = [window.record() for window in self._windows.values()]
        self._windows.clear()
        return records

    def _process_timing(self, record: ExportRecord) -> list[ExportRecord]:
        step, window, completed = self._step_state(record)
        if window is None or step is None:
            return completed
        window.observe(record, step=step)
        phases = record.data.get("phases")
        if not isinstance(phases, list):
            return completed
        for phase in phases:
            if not isinstance(phase, Mapping):
                continue
            name = _string(phase.get("phase"))
            device = _string(phase.get("device"))
            if name is None or device is None:
                continue
            group = ("phase", name, device)
            for metric in ("cpu_wall_ms", "gpu_ms", "call_count"):
                window.add(group, metric, phase.get(metric))
            window.keep(group, "gpu_clock", _string(phase.get("gpu_clock")))
        return completed

    def _process_step_memory(self, record: ExportRecord) -> list[ExportRecord]:
        step, window, completed = self._step_state(record)
        if window is None or step is None:
            return completed
        window.observe(record, step=step)
        device = _string(record.data.get("device"))
        if device is None:
            return completed
        group = ("device", device)
        window.add(
            group,
            "peak_allocated_bytes",
            record.data.get("peak_allocated_bytes"),
        )
        window.add(
            group,
            "peak_reserved_bytes",
            record.data.get("peak_reserved_bytes"),
        )
        return completed

    def _process_process(self, record: ExportRecord) -> list[ExportRecord]:
        window, completed = self._time_state(record)
        if window is None:
            return completed
        window.observe(record)
        cpu = _mapping(record.data.get("cpu"))
        memory = _mapping(record.data.get("memory"))
        gpu = _mapping(record.data.get("gpu"))

        window.add(
            ("cpu",),
            "utilization_percent",
            cpu.get("utilization_percent"),
        )
        window.keep(
            ("cpu",),
            "logical_core_count",
            _integer(cpu.get("logical_core_count")),
        )
        window.add(("memory",), "rss_bytes", memory.get("rss_bytes"))
        window.keep(
            ("memory",),
            "host_total_bytes",
            _number(memory.get("host_total_bytes")),
        )
        window.keep(("gpu",), "available", _boolean(gpu.get("available")))
        window.keep(("gpu",), "count", _integer(gpu.get("count")))
        window.keep(
            ("gpu",),
            "device_index",
            _integer(gpu.get("device_index")),
        )
        window.add(
            ("gpu",),
            "allocator_allocated_bytes",
            gpu.get("allocator_allocated_bytes"),
        )
        window.add(
            ("gpu",),
            "allocator_reserved_bytes",
            gpu.get("allocator_reserved_bytes"),
        )
        window.keep(
            ("gpu",),
            "device_total_bytes",
            _number(gpu.get("device_total_bytes")),
        )
        return completed

    def _process_system(self, record: ExportRecord) -> list[ExportRecord]:
        window, completed = self._time_state(record)
        if window is None:
            return completed
        window.observe(record)
        cpu = _mapping(record.data.get("cpu"))
        memory = _mapping(record.data.get("memory"))
        gpu = _mapping(record.data.get("gpu"))

        window.add(
            ("cpu",),
            "utilization_percent",
            cpu.get("utilization_percent"),
        )
        window.add(("memory",), "used_bytes", memory.get("used_bytes"))
        window.keep(
            ("memory",),
            "total_bytes",
            _number(memory.get("total_bytes")),
        )
        window.keep(("gpu",), "available", _boolean(gpu.get("available")))
        window.keep(("gpu",), "count", _integer(gpu.get("count")))

        devices = gpu.get("devices")
        if not isinstance(devices, list):
            return completed
        for device in devices:
            if not isinstance(device, Mapping):
                continue
            index = _integer(device.get("device_index"))
            if index is None:
                continue
            group = ("gpu_device", str(index))
            for metric in (
                "utilization_percent",
                "memory_used_bytes",
                "temperature_celsius",
                "power_usage_watts",
            ):
                window.add(group, metric, device.get(metric))
            window.keep(
                group,
                "memory_total_bytes",
                _number(device.get("memory_total_bytes")),
            )
            window.keep(
                group,
                "power_limit_watts",
                _number(device.get("power_limit_watts")),
            )
        return completed

    def _step_state(
        self, record: ExportRecord
    ) -> tuple[Optional[int], Optional[_Window], list[ExportRecord]]:
        step = _integer(record.data.get("step_number"))
        if step is None or step < 0:
            return None, None, []
        if step == 0:
            start, end = 0, 0
        else:
            start = ((step - 1) // self.step_window) * self.step_window + 1
            end = start + self.step_window - 1
        window, completed = self._window(record, start, end)
        return step, window, completed

    def _time_state(
        self, record: ExportRecord
    ) -> tuple[Optional[_Window], list[ExportRecord]]:
        timestamp = record.timestamp_unix_ns
        if timestamp is None or int(timestamp) < 0:
            timestamp = record.observed_timestamp_unix_ns
        start = (int(timestamp) // self._time_window_ns) * self._time_window_ns
        end = start + self._time_window_ns
        return self._window(record, start, end)

    def _window(
        self,
        record: ExportRecord,
        start: int,
        end: int,
    ) -> tuple[Optional[_Window], list[ExportRecord]]:
        key = (
            record.kind,
            json.dumps(dict(record.resource), sort_keys=True, default=str),
        )
        current = self._windows.get(key)
        if current is not None and start < current.start:
            _warn(self._logger, "[TraceML] dropping late OTLP window record")
            return None, []
        completed = (
            [current.record()]
            if current is not None and start > current.start
            else []
        )
        if current is None or start > current.start:
            current = _Window(
                kind=record.kind,
                resource=dict(record.resource),
                start=start,
                end=end,
            )
            self._windows[key] = current
        return current, completed


__all__ = [
    "DEFAULT_STEP_WINDOW",
    "DEFAULT_TIME_WINDOW_SEC",
    "NumericStats",
    "STEP_WINDOW_ENV",
    "TIME_WINDOW_ENV",
    "WindowProcessor",
]
