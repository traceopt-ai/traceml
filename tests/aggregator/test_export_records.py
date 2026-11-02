# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Typed export records mirror the SQLite projection rows."""

from __future__ import annotations

import dataclasses
import json
import sqlite3

import pytest

from traceml_ai.aggregator import sqlite_writer
from traceml_ai.aggregator.export import records as export_records
from traceml_ai.aggregator.export.records import (
    RECORD_TYPES,
    StepMemoryRecord,
    StepTimeRecord,
    records_from_payload,
)
from traceml_ai.aggregator.sqlite_writers import (
    step_memory,
    step_time,
)
from traceml_ai.runtime.sender import SenderIdentity
from traceml_ai.telemetry.envelope import (
    build_telemetry_envelope,
    normalize_telemetry_envelope,
)

OBSERVED = 1_800_000_000_000_000_000
TS = 1_700_000_000_123_456_789

EVENTS = {
    "_traceml_internal:forward_time": {
        "cuda:0": {
            "is_gpu": True,
            "duration_ms": 9.0,
            "cpu_ms": 3.0,
            "gpu_ms": 9.0,
            "n_calls": 2,
        }
    }
}

# (record, writer, sampler, table, row, expected record count)
CASES = [
    (
        StepTimeRecord,
        step_time,
        "StepTimeSampler",
        "StepTimeTable",
        {"seq": 3, "timestamp_unix_ns": TS, "step": 7, "events": EVENTS},
        1,
    ),
    (
        StepMemoryRecord,
        step_memory,
        "StepMemorySampler",
        "step_memory",
        {
            "seq": 3,
            "timestamp_unix_ns": TS,
            "device": "cuda:0",
            "step": 7,
            "peak_alloc": 1024.0,
            "peak_resv": 2048.0,
        },
        1,
    ),
]


def _payload(sampler: str, table: str, row: dict) -> dict:
    return build_telemetry_envelope(
        identity=SenderIdentity(
            global_rank=1,
            local_rank=1,
            world_size=2,
            local_world_size=2,
            hostname="host-a",
            pid=4242,
        ),
        sampler_name=sampler,
        tables={table: [row]},
    )


def _ddl_columns(cls) -> list[str]:
    with sqlite3.connect(":memory:") as conn:
        writer = {c[0]: c[1] for c in CASES}[cls]
        writer.init_schema(conn)
        info = conn.execute(f"PRAGMA table_info({cls.TABLE});").fetchall()
    return [str(col[1]) for col in info if col[1] != "id"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c[0].STREAM)
def test_record_equals_writer_row(case):
    cls, writer, sampler, table, row, count = case
    payload = _payload(sampler, table, row)

    batch = records_from_payload(payload, observed_timestamp_unix_ns=OBSERVED)
    rows = writer.build_rows(
        envelope=normalize_telemetry_envelope(payload),
        observed_timestamp_unix_ns=OBSERVED,
    )[cls.TABLE]

    got = [r for r in batch.records if type(r) is cls]
    assert batch.skipped == 0
    assert len(got) == len(rows) == count
    for record, built in zip(got, rows):
        assert len(cls.COLUMNS) == len(built)
        expected = dict(zip(cls.COLUMNS, built))
        if cls is StepTimeRecord:
            expected["events"] = json.loads(built[-1])
            assert record.events == EVENTS
        assert record.to_dict() == {
            "stream": cls.STREAM,
            "schema_version": 1,
            **expected,
        }
        assert expected.get("global_rank", expected.get("rank")) == 1
        assert expected["observed_timestamp_unix_ns"] == OBSERVED
        assert expected["timestamp_unix_ns"] == TS


def test_step_time_without_integer_timestamp_is_none():
    payload = _payload(
        "StepTimeSampler",
        "StepTimeTable",
        {"seq": 1, "timestamp": 1.5, "step": 1, "events": {}},
    )

    (record,) = records_from_payload(payload).records

    assert record.timestamp_unix_ns is None
    assert record.to_dict()["timestamp_unix_ns"] is None
    assert record.events == {}


def test_list_batch_yields_records_for_every_envelope():
    batch = records_from_payload(
        [_payload(*case[2:5]) for case in CASES] + ["not an envelope"]
    )

    streams = {r.STREAM for r in batch.records}
    assert streams == {cls.STREAM for cls in RECORD_TYPES}
    assert batch.skipped == 1


def test_unknown_sampler_yields_nothing():
    batch = records_from_payload(
        _payload("OtherSampler", "StepTimeTable", CASES[0][4])
    )

    assert batch.records == []
    assert batch.skipped == 0


def test_writers_are_sqlite_projection_writers():
    assert set(export_records._WRITERS) <= set(
        sqlite_writer._PROJECTION_WRITERS
    )
    tables = set()
    with sqlite3.connect(":memory:") as conn:
        for writer in export_records._WRITERS:
            writer.init_schema(conn)
        for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ):
            tables.add(name)
    tables.discard("sqlite_sequence")
    assert tables == {cls.TABLE for cls in RECORD_TYPES}


@pytest.mark.parametrize("cls", RECORD_TYPES, ids=lambda c: c.STREAM)
def test_columns_match_ddl(cls):
    fields = tuple(f.name for f in dataclasses.fields(cls))
    assert cls.COLUMNS == fields
    columns = [
        "events_json" if name == "events" else name for name in cls.COLUMNS
    ]
    assert columns == _ddl_columns(cls)
