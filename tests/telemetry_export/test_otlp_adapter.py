# Copyright 2026 OptAI UG (haftungsbeschraenkt)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib

import pytest

from traceml_ai.telemetry_export.records import ExportRecord, RecordKind

pytest.importorskip("opentelemetry.sdk")


class _Exporter:
    def __init__(self, success) -> None:
        self.success = success
        self.batches = []
        self.closed = False

    def export(self, batch):
        self.batches.append(batch)
        return self.success

    def force_flush(self):
        return True

    def shutdown(self):
        self.closed = True


def test_otlp_adapter_uses_sdk_batch_processor_and_both_timestamps(
    monkeypatch,
) -> None:
    from opentelemetry.sdk._logs.export import LogRecordExportResult

    exporter = _Exporter(LogRecordExportResult.SUCCESS)
    exporter_module = importlib.import_module(
        "opentelemetry.exporter.otlp.proto.http._log_exporter"
    )
    monkeypatch.setattr(
        exporter_module,
        "OTLPLogExporter",
        lambda: exporter,
    )

    from traceml_ai.telemetry_export.otlp import OtlpLogPipeline

    pipeline = OtlpLogPipeline(
        protocol="http/protobuf",
        shutdown_timeout_sec=1.0,
    )
    pipeline.start()
    pipeline.enqueue(
        [
            ExportRecord(
                kind=RecordKind.STEP_MEMORY,
                timestamp_unix_ns=10,
                observed_timestamp_unix_ns=20,
                resource={
                    "service.name": "trainer",
                    "traceml.run.name": "run",
                    "traceml.global_rank": 1,
                },
                data={
                    "step_number": 3,
                    "peak_allocated_bytes": 100,
                },
            )
        ]
    )
    pipeline.stop()

    assert len(exporter.batches) == 1
    readable = exporter.batches[0][0]
    assert readable.log_record.timestamp == 10
    assert readable.log_record.observed_timestamp == 20
    assert readable.log_record.event_name == "traceml.step_memory.v1"
    assert readable.log_record.body == {
        "step_number": 3,
        "peak_allocated_bytes": 100,
    }
    assert readable.log_record.attributes == {"traceml.schema.version": 1}
    assert readable.resource.attributes["service.name"] == "trainer"
    assert readable.resource.attributes["traceml.global_rank"] == 1
    assert readable.instrumentation_scope.name == "traceml"
    assert exporter.closed


def test_otlp_adapter_closes_exporter_when_stopped_before_start(
    monkeypatch,
) -> None:
    from opentelemetry.sdk._logs.export import LogRecordExportResult

    exporter = _Exporter(LogRecordExportResult.SUCCESS)
    exporter_module = importlib.import_module(
        "opentelemetry.exporter.otlp.proto.http._log_exporter"
    )
    monkeypatch.setattr(
        exporter_module,
        "OTLPLogExporter",
        lambda: exporter,
    )

    from traceml_ai.telemetry_export.otlp import OtlpLogPipeline

    pipeline = OtlpLogPipeline(protocol="http/protobuf")
    pipeline.stop()

    assert exporter.closed
