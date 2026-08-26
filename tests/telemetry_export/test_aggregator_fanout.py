# Copyright 2026 OptAI UG (haftungsbeschraenkt)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from traceml_ai.aggregator.trace_aggregator import TraceMLAggregator


class _Pipeline:
    def __init__(self) -> None:
        self.records = []

    def enqueue(self, records) -> None:
        self.records.extend(records)


class _Mapper:
    def __init__(self) -> None:
        self.payload = None
        self.observed = None

    def map_payload(self, payload, *, observed_timestamp_unix_ns):
        self.payload = payload
        self.observed = observed_timestamp_unix_ns
        return ["record"]


def test_live_fanout_maps_payload_without_reading_sqlite() -> None:
    aggregator = TraceMLAggregator.__new__(TraceMLAggregator)
    aggregator._export_pipeline = _Pipeline()
    aggregator._export_mapper = _Mapper()

    payload = {"meta": {"sampler": "SystemSampler"}}
    aggregator._enqueue_external_export(payload)

    assert aggregator._export_mapper.payload is payload
    assert isinstance(aggregator._export_mapper.observed, int)
    assert aggregator._export_pipeline.records == ["record"]
