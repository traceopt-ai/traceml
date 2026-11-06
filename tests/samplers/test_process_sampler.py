# Copyright 2026 OptAI UG (haftungsbeschraenkt)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sqlite3

import traceml_ai.samplers.process_sampler as process_sampler_module
from traceml_ai.aggregator.sqlite_writers import process as projection
from traceml_ai.samplers.process_sampler import ProcessSampler
from traceml_ai.samplers.schema.process import ProcessSample
from traceml_ai.telemetry.envelope import normalize_telemetry_envelope


class _Logger:
    def error(self, *args, **kwargs) -> None:
        return None


class _FailingProcess:
    def cpu_percent(self, interval=None):
        raise RuntimeError("cpu unavailable")

    def memory_info(self):
        raise RuntimeError("memory unavailable")


def test_collection_failure_is_unavailable_not_zero() -> None:
    sampler = ProcessSampler.__new__(ProcessSampler)
    sampler.process = _FailingProcess()
    sampler.logger = _Logger()

    assert sampler._sample_cpu() is None
    assert sampler._sample_ram() is None


def test_missing_process_handle_is_unavailable_not_zero() -> None:
    sampler = ProcessSampler.__new__(ProcessSampler)
    sampler.process = None

    assert sampler._sample_cpu() is None
    assert sampler._sample_ram() is None


def test_source_timestamp_survives_wire_and_projection(monkeypatch) -> None:
    timestamp_unix_ns = 1_700_000_000_000_000_001
    sampler = ProcessSampler.__new__(ProcessSampler)
    sampler.sample_idx = 0
    sampler.pid = 7
    sampler.gpu_available = False
    sampler.gpu_count = 0
    sampler.cpu_count = 8
    sampler.ram_total = 16.0
    sampler._sample_gpu = lambda: None
    sampler._sample_cpu = lambda: 25.0
    sampler._sample_ram = lambda: 4.0
    sent = []
    sampler._add_record = sent.append
    sampler.logger = _Logger()
    monkeypatch.setattr(
        process_sampler_module.time, "time_ns", lambda: timestamp_unix_ns
    )

    sampler.sample()
    (wire_row,) = sent
    assert "ts" not in wire_row
    assert (
        ProcessSample.from_wire(wire_row).timestamp_unix_ns
        == timestamp_unix_ns
    )

    envelope = normalize_telemetry_envelope(
        {
            "meta": {"sampler": "ProcessSampler"},
            "body": {
                "tables": {
                    "ProcessTable": [
                        wire_row,
                        {"ts": 1.25},
                        {"timestamp_unix_ns": None},
                        {"timestamp_unix_ns": 2**63},
                        {},
                    ]
                }
            },
        }
    )
    with sqlite3.connect(":memory:") as conn:
        projection.init_schema(conn)
        projection.insert_rows(
            conn,
            projection.build_rows(envelope, observed_timestamp_unix_ns=99),
        )
        timestamps = conn.execute(
            "SELECT timestamp_unix_ns FROM process_samples ORDER BY id;"
        ).fetchall()

    assert timestamps == [
        (timestamp_unix_ns,),
        (None,),
        (None,),
        (None,),
        (None,),
    ]
