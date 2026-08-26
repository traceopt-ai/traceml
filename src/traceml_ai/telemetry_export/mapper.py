# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Pure mapping from existing aggregator envelopes to export records."""

from __future__ import annotations

import math
import os
import time
from typing import Any, Iterable, Mapping, Optional

from traceml_ai.telemetry.envelope import (
    TelemetryEnvelope,
    TelemetryMeta,
    normalize_telemetry_envelope,
)
from traceml_ai.telemetry_export.records import ExportRecord, RecordKind

_SAMPLER_KINDS = {
    "StepTimeSampler": RecordKind.STEP_TIMING,
    "StepMemorySampler": RecordKind.STEP_MEMORY,
    "ProcessSampler": RecordKind.PROCESS_SAMPLE,
    "SystemSampler": RecordKind.SYSTEM_SAMPLE,
    "RuntimeEnvironmentSampler": RecordKind.RUNTIME_CONTEXT,
}

_PHASE_NAMES = {
    "_traceml_internal:step_time": "step_wall",
    "_traceml_internal:dataloader_next": "input_wait",
    "_traceml_internal:forward_time": "forward",
    "_traceml_internal:backward_time": "backward",
    "_traceml_internal:optimizer_step": "optimizer",
    "_traceml_internal:h2d_time": "h2d",
}


def _finite_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _integer(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _string(value: Any) -> Optional[str]:
    return str(value) if isinstance(value, str) else None


def _boolean(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def _put_if_present(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


def _seconds_to_unix_ns(value: Any) -> Optional[int]:
    seconds = _finite_float(value)
    if seconds is None or seconds < 0:
        return None
    return int(seconds * 1_000_000_000)


class ExportRecordMapper:
    """Map live rank payloads without querying SQLite or computing findings."""

    def __init__(
        self,
        *,
        run_name: str,
        service_name: Optional[str] = None,
    ) -> None:
        self._run_name = str(run_name or "default")
        self._service_name = str(
            service_name
            or os.environ.get("OTEL_SERVICE_NAME")
            or "traceml-training"
        )

    def map_payload(
        self,
        payload: Any,
        *,
        observed_timestamp_unix_ns: Optional[int] = None,
    ) -> list[ExportRecord]:
        """Map one envelope or transport batch into independent records."""
        observed_ns = (
            time.time_ns()
            if observed_timestamp_unix_ns is None
            else int(observed_timestamp_unix_ns)
        )
        if isinstance(payload, list):
            records: list[ExportRecord] = []
            for item in payload:
                records.extend(
                    self.map_payload(
                        item,
                        observed_timestamp_unix_ns=observed_ns,
                    )
                )
            return records

        envelope = normalize_telemetry_envelope(payload)
        if envelope is None:
            return []
        return list(self._map_envelope(envelope, observed_ns=observed_ns))

    def _map_envelope(
        self,
        envelope: TelemetryEnvelope,
        *,
        observed_ns: int,
    ) -> Iterable[ExportRecord]:
        kind = _SAMPLER_KINDS.get(str(envelope.meta.sampler or ""))
        if kind is None:
            return

        resource = self._resource(envelope.meta)
        for rows in envelope.tables.values():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                data = self._map_row(kind, row)
                if data is None:
                    continue
                yield ExportRecord(
                    kind=kind,
                    timestamp_unix_ns=self._row_timestamp(kind, row),
                    observed_timestamp_unix_ns=observed_ns,
                    resource=resource,
                    data=data,
                )

    def _resource(self, meta: TelemetryMeta) -> dict[str, Any]:
        resource: dict[str, Any] = {
            "service.name": self._service_name,
            "traceml.run.name": self._run_name,
        }
        _put_if_present(resource, "traceml.global_rank", meta.global_rank)
        _put_if_present(resource, "traceml.local_rank", meta.local_rank)
        _put_if_present(resource, "traceml.node_rank", meta.node_rank)
        _put_if_present(resource, "traceml.world_size", meta.world_size)
        _put_if_present(
            resource,
            "traceml.local_world_size",
            meta.local_world_size,
        )
        _put_if_present(resource, "host.name", meta.hostname)
        _put_if_present(resource, "process.pid", meta.pid)
        return resource

    @staticmethod
    def _row_timestamp(
        kind: RecordKind, row: Mapping[str, Any]
    ) -> Optional[int]:
        key = "timestamp" if kind is RecordKind.STEP_TIMING else "ts"
        return _seconds_to_unix_ns(row.get(key))

    def _map_row(
        self,
        kind: RecordKind,
        row: Mapping[str, Any],
    ) -> Optional[dict[str, Any]]:
        if kind is RecordKind.STEP_TIMING:
            return self._step_timing(row)
        if kind is RecordKind.STEP_MEMORY:
            return self._step_memory(row)
        if kind is RecordKind.PROCESS_SAMPLE:
            return self._process_sample(row)
        if kind is RecordKind.SYSTEM_SAMPLE:
            return self._system_sample(row)
        if kind is RecordKind.RUNTIME_CONTEXT:
            return self._runtime_context(row)
        return None

    @staticmethod
    def _step_timing(row: Mapping[str, Any]) -> Optional[dict[str, Any]]:
        step_number = _integer(row.get("step"))
        events = row.get("events")
        if step_number is None or not isinstance(events, Mapping):
            return None

        phases: list[dict[str, Any]] = []
        for raw_name, by_device in events.items():
            if not isinstance(by_device, Mapping):
                continue
            phase_name = _PHASE_NAMES.get(str(raw_name), str(raw_name))
            for raw_device, raw_stats in by_device.items():
                if not isinstance(raw_stats, Mapping):
                    continue
                phase: dict[str, Any] = {
                    "phase": phase_name,
                    "device": str(raw_device),
                }
                cpu_ms = _finite_float(raw_stats.get("cpu_ms"))
                gpu_ms = _finite_float(raw_stats.get("gpu_ms"))
                call_count = _integer(raw_stats.get("n_calls"))
                _put_if_present(phase, "cpu_wall_ms", cpu_ms)
                _put_if_present(phase, "gpu_ms", gpu_ms)
                if gpu_ms is not None:
                    phase["gpu_clock"] = "cuda_event"
                _put_if_present(phase, "call_count", call_count)
                phases.append(phase)

        return {"step_number": step_number, "phases": phases}

    @staticmethod
    def _step_memory(row: Mapping[str, Any]) -> Optional[dict[str, Any]]:
        step_number = _integer(row.get("step"))
        if step_number is None:
            return None
        data: dict[str, Any] = {"step_number": step_number}
        _put_if_present(data, "device", _string(row.get("device")))
        _put_if_present(
            data,
            "peak_allocated_bytes",
            _finite_float(row.get("peak_alloc")),
        )
        _put_if_present(
            data,
            "peak_reserved_bytes",
            _finite_float(row.get("peak_resv")),
        )
        return data

    @staticmethod
    def _process_sample(row: Mapping[str, Any]) -> dict[str, Any]:
        cpu: dict[str, Any] = {}
        _put_if_present(
            cpu,
            "utilization_percent",
            _finite_float(row.get("cpu")),
        )
        _put_if_present(
            cpu,
            "logical_core_count",
            _integer(row.get("cpu_cores")),
        )

        memory: dict[str, Any] = {}
        _put_if_present(
            memory, "rss_bytes", _finite_float(row.get("ram_used"))
        )
        _put_if_present(
            memory,
            "host_total_bytes",
            _finite_float(row.get("ram_total")),
        )

        gpu: dict[str, Any] = {}
        _put_if_present(gpu, "available", _boolean(row.get("gpu_available")))
        _put_if_present(gpu, "count", _integer(row.get("gpu_count")))
        raw_gpu = row.get("gpu")
        if isinstance(raw_gpu, Mapping):
            _put_if_present(
                gpu,
                "device_index",
                _integer(raw_gpu.get("device")),
            )
            _put_if_present(
                gpu,
                "allocator_allocated_bytes",
                _finite_float(raw_gpu.get("mem_used")),
            )
            _put_if_present(
                gpu,
                "allocator_reserved_bytes",
                _finite_float(raw_gpu.get("mem_reserved")),
            )
            _put_if_present(
                gpu,
                "device_total_bytes",
                _finite_float(raw_gpu.get("mem_total")),
            )

        return {"cpu": cpu, "memory": memory, "gpu": gpu}

    @staticmethod
    def _system_sample(row: Mapping[str, Any]) -> dict[str, Any]:
        cpu: dict[str, Any] = {}
        _put_if_present(
            cpu,
            "utilization_percent",
            _finite_float(row.get("cpu")),
        )
        memory: dict[str, Any] = {}
        _put_if_present(
            memory,
            "used_bytes",
            _finite_float(row.get("ram_used")),
        )
        _put_if_present(
            memory,
            "total_bytes",
            _finite_float(row.get("ram_total")),
        )
        gpu: dict[str, Any] = {}
        _put_if_present(gpu, "available", _boolean(row.get("gpu_available")))
        _put_if_present(gpu, "count", _integer(row.get("gpu_count")))

        devices: list[dict[str, Any]] = []
        raw_gpus = row.get("gpus")
        if isinstance(raw_gpus, list):
            for device_index, raw_gpu in enumerate(raw_gpus):
                if not isinstance(raw_gpu, (list, tuple)) or len(raw_gpu) < 6:
                    continue
                device: dict[str, Any] = {"device_index": device_index}
                fields = (
                    ("utilization_percent", raw_gpu[0]),
                    ("memory_used_bytes", raw_gpu[1]),
                    ("memory_total_bytes", raw_gpu[2]),
                    ("temperature_celsius", raw_gpu[3]),
                    ("power_usage_watts", raw_gpu[4]),
                    ("power_limit_watts", raw_gpu[5]),
                )
                for name, value in fields:
                    _put_if_present(device, name, _finite_float(value))
                devices.append(device)
        gpu["devices"] = devices
        return {"cpu": cpu, "memory": memory, "gpu": gpu}

    @staticmethod
    def _runtime_context(row: Mapping[str, Any]) -> dict[str, Any]:
        data: dict[str, Any] = {}
        fields = (
            ("topology", _string(row.get("topology"))),
            (
                "distributed_initialized",
                _boolean(row.get("distributed_initialized")),
            ),
            ("distributed_backend", _string(row.get("distributed_backend"))),
            ("training_strategy", _string(row.get("training_strategy"))),
            ("strategy_source", _string(row.get("strategy_source"))),
            ("strategy_confidence", _string(row.get("strategy_confidence"))),
        )
        for name, value in fields:
            _put_if_present(data, name, value)
        return data


__all__ = ["ExportRecordMapper"]
