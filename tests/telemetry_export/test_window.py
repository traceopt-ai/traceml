# Copyright 2026 OptAI UG (haftungsbeschraenkt)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from traceml_ai.telemetry_export.records import ExportRecord, RecordKind
from traceml_ai.telemetry_export.window import (
    DEFAULT_STEP_WINDOW,
    DEFAULT_TIME_WINDOW_SEC,
    WindowProcessor,
)


class _Logger:
    def __init__(self) -> None:
        self.warnings: list[tuple[str, tuple]] = []

    def warning(self, message: str, *args) -> None:
        self.warnings.append((message, args))


def _record(
    kind: RecordKind,
    *,
    data: dict,
    timestamp: int,
    rank: int = 0,
) -> ExportRecord:
    return ExportRecord(
        kind=kind,
        timestamp_unix_ns=timestamp,
        observed_timestamp_unix_ns=timestamp + 1,
        resource={
            "service.name": "trainer",
            "traceml.run.name": "run",
            "traceml.global_rank": rank,
        },
        data=data,
    )


def _timing(step: int, value: float, *, rank: int = 0) -> ExportRecord:
    return _record(
        RecordKind.STEP_TIMING_WINDOW,
        timestamp=step * 1_000,
        rank=rank,
        data={
            "step_number": step,
            "phases": [
                {
                    "phase": "forward",
                    "device": "cuda:0",
                    "cpu_wall_ms": value + 1.0,
                    "gpu_ms": value,
                    "gpu_clock": "cuda_event",
                    "call_count": 1,
                }
            ],
        },
    )


def test_step_timing_emits_count_sum_min_max_per_rank_window() -> None:
    processor = WindowProcessor(step_window=2, time_window_sec=10)

    assert processor.process([_timing(1, 4.0)]) == []
    assert processor.process([_timing(2, 6.0)]) == []
    records = processor.process([_timing(3, 10.0)])

    assert len(records) == 1
    record = records[0]
    assert record.kind is RecordKind.STEP_TIMING_WINDOW
    assert record.event_name == "traceml.step_timing_window.v1"
    assert record.timestamp_unix_ns == 2_000
    assert record.data == {
        "start_step": 1,
        "end_step": 2,
        "step_count": 2,
        "start_time_unix_ns": 1_000,
        "end_time_unix_ns": 2_000,
        "phases": [
            {
                "phase": "forward",
                "device": "cuda:0",
                "cpu_wall_ms": {
                    "count": 2,
                    "sum": 12.0,
                    "min": 5.0,
                    "max": 7.0,
                },
                "gpu_ms": {
                    "count": 2,
                    "sum": 10.0,
                    "min": 4.0,
                    "max": 6.0,
                },
                "gpu_clock": "cuda_event",
                "call_count": {
                    "count": 2,
                    "sum": 2.0,
                    "min": 1.0,
                    "max": 1.0,
                },
            }
        ],
    }

    partial = processor.flush()[0]
    assert partial.data["start_step"] == 3
    assert partial.data["end_step"] == 4
    assert partial.data["step_count"] == 1


def test_step_windows_do_not_mix_ranks_or_missing_gpu_values() -> None:
    processor = WindowProcessor(step_window=2, time_window_sec=10)
    rank_zero = _timing(1, 4.0, rank=0)
    rank_one = _timing(1, 8.0, rank=1)
    rank_one.data["phases"][0].pop("gpu_ms")
    rank_one.data["phases"][0].pop("gpu_clock")

    processor.process([rank_zero, rank_one])
    records = processor.flush()

    assert len(records) == 2
    by_rank = {
        int(record.resource["traceml.global_rank"]): record
        for record in records
    }
    assert by_rank[0].data["phases"][0]["gpu_ms"]["count"] == 1
    assert "gpu_ms" not in by_rank[1].data["phases"][0]


def test_step_memory_is_windowed_by_device_and_flushes_partial() -> None:
    processor = WindowProcessor(step_window=10, time_window_sec=10)
    processor.process(
        [
            _record(
                RecordKind.STEP_MEMORY_WINDOW,
                timestamp=10,
                data={
                    "step_number": 1,
                    "device": "cuda:0",
                    "peak_allocated_bytes": 100.0,
                    "peak_reserved_bytes": 200.0,
                },
            ),
            _record(
                RecordKind.STEP_MEMORY_WINDOW,
                timestamp=20,
                data={
                    "step_number": 2,
                    "device": "cuda:0",
                    "peak_allocated_bytes": 300.0,
                    "peak_reserved_bytes": None,
                },
            ),
        ]
    )

    record = processor.flush()[0]
    assert record.data["step_count"] == 2
    device = record.data["devices"][0]
    assert device["peak_allocated_bytes"] == {
        "count": 2,
        "sum": 400.0,
        "min": 100.0,
        "max": 300.0,
    }
    assert device["peak_reserved_bytes"]["count"] == 1


def test_process_samples_use_time_windows_and_keep_latest_capacity() -> None:
    processor = WindowProcessor(step_window=10, time_window_sec=10)

    def process(timestamp: int, cpu: float, rss: float, total: float):
        return _record(
            RecordKind.PROCESS_WINDOW,
            timestamp=timestamp,
            data={
                "cpu": {
                    "utilization_percent": cpu,
                    "logical_core_count": 16,
                },
                "memory": {
                    "rss_bytes": rss,
                    "host_total_bytes": total,
                },
                "gpu": {},
            },
        )

    assert processor.process([process(1_000_000_000, 20.0, 10.0, 100.0)]) == []
    assert processor.process([process(9_000_000_000, 60.0, 30.0, 100.0)]) == []
    records = processor.process([process(11_000_000_000, 40.0, 20.0, 200.0)])

    assert len(records) == 1
    data = records[0].data
    assert data["sample_count"] == 2
    assert data["cpu"]["utilization_percent"] == {
        "count": 2,
        "sum": 80.0,
        "min": 20.0,
        "max": 60.0,
    }
    assert data["memory"]["host_total_bytes"] == 100.0
    assert processor.flush()[0].data["memory"]["host_total_bytes"] == 200.0


def test_system_samples_window_metrics_and_keep_static_values() -> None:
    processor = WindowProcessor(step_window=10, time_window_sec=10)
    for timestamp, utilization in (
        (1_000_000_000, 10.0),
        (2_000_000_000, 90.0),
    ):
        processor.process(
            [
                _record(
                    RecordKind.SYSTEM_WINDOW,
                    timestamp=timestamp,
                    data={
                        "cpu": {"utilization_percent": utilization},
                        "memory": {
                            "used_bytes": 50.0,
                            "total_bytes": 100.0,
                        },
                        "gpu": {
                            "available": True,
                            "count": 1,
                            "devices": [
                                {
                                    "device_index": 0,
                                    "utilization_percent": utilization,
                                    "memory_used_bytes": 20.0,
                                    "memory_total_bytes": 80.0,
                                    "temperature_celsius": 70.0,
                                    "power_usage_watts": 200.0,
                                    "power_limit_watts": 300.0,
                                }
                            ],
                        },
                    },
                )
            ]
        )

    record = processor.flush()[0]
    assert record.event_name == "traceml.system_window.v1"
    assert record.data["cpu"]["utilization_percent"]["max"] == 90.0
    device = record.data["gpu"]["devices"][0]
    assert device["utilization_percent"]["count"] == 2
    assert device["memory_total_bytes"] == 80.0
    assert device["power_limit_watts"] == 300.0


def test_runtime_context_passes_through_without_windowing() -> None:
    processor = WindowProcessor(step_window=10, time_window_sec=10)
    context = _record(
        RecordKind.RUNTIME_CONTEXT,
        timestamp=1,
        data={"topology": "multi_node"},
    )

    assert processor.process([context]) == [context]
    assert processor.flush() == []


def test_window_settings_use_documented_env(monkeypatch) -> None:
    monkeypatch.setenv("TRACEML_OTLP_STEP_WINDOW", "25")
    monkeypatch.setenv("TRACEML_OTLP_TIME_WINDOW_SEC", "30")

    processor = WindowProcessor.from_env()

    assert processor.step_window == 25
    assert processor.time_window_sec == 30.0


def test_invalid_window_settings_warn_and_use_defaults(monkeypatch) -> None:
    logger = _Logger()
    monkeypatch.setenv("TRACEML_OTLP_STEP_WINDOW", "0")
    monkeypatch.setenv("TRACEML_OTLP_TIME_WINDOW_SEC", "not-a-number")

    processor = WindowProcessor.from_env(logger=logger)

    assert processor.step_window == DEFAULT_STEP_WINDOW
    assert processor.time_window_sec == DEFAULT_TIME_WINDOW_SEC
    assert len(logger.warnings) == 2


def test_late_step_record_is_dropped_from_export_window() -> None:
    logger = _Logger()
    processor = WindowProcessor(
        step_window=2,
        time_window_sec=10,
        logger=logger,
    )
    processor.process([_timing(3, 3.0)])

    assert processor.process([_timing(1, 1.0)]) == []
    assert len(logger.warnings) == 1
    record = processor.flush()[0]
    assert record.data["start_step"] == 3
    assert record.data["step_count"] == 1
