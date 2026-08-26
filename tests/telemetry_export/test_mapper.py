# Copyright 2026 OptAI UG (haftungsbeschraenkt)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from traceml_ai.telemetry_export.mapper import ExportRecordMapper
from traceml_ai.telemetry_export.records import RecordKind


def _envelope(sampler: str, row: dict, *, table: str = "table") -> dict:
    return {
        "meta": {
            "global_rank": 5,
            "local_rank": 1,
            "world_size": 8,
            "local_world_size": 4,
            "node_rank": 1,
            "hostname": "worker-1",
            "pid": 4242,
            "sampler": sampler,
            "timestamp": 999.0,
        },
        "body": {"tables": {table: [row]}},
    }


def _mapper() -> ExportRecordMapper:
    return ExportRecordMapper(
        run_name="training-run",
        service_name="trainer",
    )


def test_step_timing_preserves_source_clocks_devices_and_custom_events() -> (
    None
):
    payload = _envelope(
        "StepTimeSampler",
        {
            "seq": 77,
            "timestamp": 123.25,
            "step": 12,
            "events": {
                "_traceml_internal:step_time": {
                    "cuda:1": {
                        "duration_ms": 20.0,
                        "cpu_ms": 20.0,
                        "gpu_ms": 18.0,
                        "is_gpu": False,
                        "n_calls": 1,
                    }
                },
                "user.decode": {
                    "cpu": {
                        "duration_ms": 2.0,
                        "cpu_ms": 2.0,
                        "gpu_ms": None,
                        "is_gpu": False,
                        "n_calls": 2,
                    }
                },
            },
        },
    )

    records = _mapper().map_payload(
        payload,
        observed_timestamp_unix_ns=999,
    )

    assert len(records) == 1
    record = records[0]
    assert record.kind is RecordKind.STEP_TIMING_WINDOW
    assert record.timestamp_unix_ns == 123_250_000_000
    assert record.observed_timestamp_unix_ns == 999
    assert record.resource == {
        "service.name": "trainer",
        "traceml.run.name": "training-run",
        "traceml.global_rank": 5,
        "traceml.local_rank": 1,
        "traceml.node_rank": 1,
        "traceml.world_size": 8,
        "traceml.local_world_size": 4,
        "host.name": "worker-1",
        "process.pid": 4242,
    }
    assert record.data == {
        "step_number": 12,
        "phases": [
            {
                "phase": "step_wall",
                "device": "cuda:1",
                "cpu_wall_ms": 20.0,
                "gpu_ms": 18.0,
                "gpu_clock": "cuda_event",
                "call_count": 1,
            },
            {
                "phase": "user.decode",
                "device": "cpu",
                "cpu_wall_ms": 2.0,
                "call_count": 2,
            },
        ],
    }
    assert "sequence" not in record.to_dict()
    assert "seq" not in record.data


def test_step_memory_uses_source_timestamp_and_does_not_invent_values() -> (
    None
):
    record = _mapper().map_payload(
        _envelope(
            "StepMemorySampler",
            {
                "seq": 2,
                "ts": 10.5,
                "model_id": 12345,
                "step": 9,
                "device": "cpu",
                "peak_alloc": None,
                "peak_resv": None,
            },
        ),
        observed_timestamp_unix_ns=11,
    )[0]

    assert record.kind is RecordKind.STEP_MEMORY_WINDOW
    assert record.timestamp_unix_ns == 10_500_000_000
    assert record.data == {"step_number": 9, "device": "cpu"}


def test_process_and_system_samples_keep_independent_timestamps() -> None:
    process = _envelope(
        "ProcessSampler",
        {
            "seq": 1,
            "ts": 20.0,
            "pid": 4242,
            "cpu": 150.0,
            "cpu_cores": 32,
            "ram_used": 1000.0,
            "ram_total": 8000.0,
            "gpu_available": None,
            "gpu_count": None,
            "gpu": None,
        },
    )
    system = _envelope(
        "SystemSampler",
        {
            "seq": 1,
            "ts": 21.0,
            "cpu": 50.0,
            "ram_used": 4000.0,
            "ram_total": 8000.0,
            "gpu_available": True,
            "gpu_count": 1,
            "gpus": [[90.0, None, 10_000.0, 70.0, 200.0, 300.0]],
        },
    )

    records = _mapper().map_payload(
        [process, system],
        observed_timestamp_unix_ns=30,
    )

    assert [record.kind for record in records] == [
        RecordKind.PROCESS_WINDOW,
        RecordKind.SYSTEM_WINDOW,
    ]
    assert [record.timestamp_unix_ns for record in records] == [
        20_000_000_000,
        21_000_000_000,
    ]
    assert records[0].data["gpu"] == {}
    assert records[1].data["gpu"]["devices"] == [
        {
            "device_index": 0,
            "utilization_percent": 90.0,
            "memory_total_bytes": 10_000.0,
            "temperature_celsius": 70.0,
            "power_usage_watts": 200.0,
            "power_limit_watts": 300.0,
        }
    ]


def test_runtime_context_exports_only_existing_evidence() -> None:
    record = _mapper().map_payload(
        _envelope(
            "RuntimeEnvironmentSampler",
            {
                "seq": 0,
                "ts": 42.0,
                "topology": "multi_node",
                "distributed_initialized": True,
                "distributed_backend": "nccl",
                "training_strategy": "distributed_unknown",
                "strategy_source": "runtime_distributed",
                "strategy_confidence": "low",
            },
        ),
        observed_timestamp_unix_ns=50,
    )[0]

    assert record.kind is RecordKind.RUNTIME_CONTEXT
    assert record.data["training_strategy"] == "distributed_unknown"
    assert record.data["strategy_confidence"] == "low"


def test_malformed_or_unrecognized_payloads_do_not_create_records() -> None:
    assert _mapper().map_payload(None) == []
    assert _mapper().map_payload({"meta": {}, "body": {}}) == []
    assert (
        _mapper().map_payload(_envelope("UnknownSampler", {"ts": 1.0})) == []
    )
