# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Typed export records mirror the SQLite projection rows."""

from __future__ import annotations

import dataclasses
import json
import re
import sqlite3
from pathlib import Path

import pytest

from traceml_ai.aggregator import sqlite_writer
from traceml_ai.aggregator.export import records as export_records
from traceml_ai.aggregator.export.records import (
    RECORD_TYPES,
    StepTimeRecord,
    records_from_payload,
)
from traceml_ai.aggregator.sqlite_writers import step_time
from traceml_ai.runtime.sender import SenderIdentity
from traceml_ai.telemetry.envelope import (
    build_telemetry_envelope,
    normalize_telemetry_envelope,
)

OBSERVED = 1_800_000_000_000_000_000
TS = 1_700_000_000_123_456_789
DOC = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "developer_guide"
    / "export-records.md"
)

FORWARD = "_traceml_internal:forward_time"
EVENTS = {
    FORWARD: {
        "cuda:0": {
            "is_gpu": True,
            "duration_ms": 9.0,
            "cpu_ms": 3.0,
            "gpu_ms": 9.0,
            "n_calls": 2,
        }
    }
}

IDENTITY = {
    "rank": 3,
    "global_rank": 3,
    "local_rank": 1,
    "world_size": 8,
    "local_world_size": 4,
    "node_rank": 1,
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
]


def _payload(sampler: str, table: str, row: dict) -> dict:
    return build_telemetry_envelope(
        identity=SenderIdentity(
            global_rank=3,
            local_rank=1,
            world_size=8,
            local_world_size=4,
            node_rank=1,
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
        for name, value in IDENTITY.items():
            if name in expected:
                assert getattr(record, name) == value, name
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


def _step_time_row(events_json):
    payload = _payload(*CASES[0][2:5])
    (row,) = step_time.build_rows(
        envelope=normalize_telemetry_envelope(payload),
        observed_timestamp_unix_ns=OBSERVED,
    )[StepTimeRecord.TABLE]
    return payload, row[:-1] + (events_json,)


def _step_payload(step: int) -> dict:
    return _payload(*CASES[0][2:4], dict(CASES[0][4], step=step))


def _serve_rows(monkeypatch, row):
    monkeypatch.setattr(
        step_time,
        "build_rows",
        lambda **_: {StepTimeRecord.TABLE: [row]},
    )


@pytest.mark.parametrize("text", [None, ""])
def test_step_time_absent_events_text_is_empty_map(text, monkeypatch):
    payload, row = _step_time_row(text)

    assert StepTimeRecord.from_row(row).events == {}
    assert StepTimeRecord.from_row(list(row)).events == {}
    _serve_rows(monkeypatch, row)
    batch = records_from_payload(payload)
    assert [r.events for r in batch.records] == [{}]
    assert batch.skipped == 0


@pytest.mark.parametrize("text", ["null", "[]"])
def test_step_time_non_object_events_text_is_skipped(text, monkeypatch):
    payload, row = _step_time_row(text)

    with pytest.raises(ValueError):
        StepTimeRecord.from_row(row)
    _serve_rows(monkeypatch, row)
    batch = records_from_payload(payload)
    assert batch.records == []
    assert batch.skipped == 1


def test_adversarial_items_are_skipped_not_raised():
    batch = records_from_payload(
        [None, 42, ["nested"], {"meta": {"sampler": None}}]
    )

    assert isinstance(batch, export_records.RecordBatch)
    assert batch.records == []
    assert batch.skipped == 4


def test_raising_writer_counts_once_and_loses_its_envelope(monkeypatch):
    build_rows = step_time.build_rows
    failing = _step_payload(7)
    failing["body"]["tables"]["StepTimeTable"].append(
        dict(CASES[0][4], step=8)
    )

    def boom_on_two_rows(**kwargs):
        if len(kwargs["envelope"].tables["StepTimeTable"]) == 2:
            raise RuntimeError("projection failed")
        return build_rows(**kwargs)

    monkeypatch.setattr(step_time, "build_rows", boom_on_two_rows)
    batch = records_from_payload([failing, _step_payload(9)])

    assert [r.step for r in batch.records] == [9]
    assert batch.skipped == 1


@pytest.mark.parametrize(
    "gpu, gpu_ms",
    [({"gpu_ms": 0.0}, 0.0), ({}, None)],
    ids=["measured_zero", "missing"],
)
def test_step_time_gpu_ms_keeps_zero_and_missing(gpu, gpu_ms):
    stats = {"is_gpu": True, "duration_ms": 0.0, "cpu_ms": 3.0, **gpu}
    payload = _payload(
        "StepTimeSampler",
        "StepTimeTable",
        dict(CASES[0][4], events={FORWARD: {"cuda:0": stats}}),
    )

    (record,) = records_from_payload(payload).records

    assert record.events[FORWARD]["cuda:0"]["gpu_ms"] == gpu_ms


def test_list_batch_yields_records_for_every_envelope():
    batch = records_from_payload(
        [_step_payload(7), _step_payload(8), "not an envelope"]
    )

    assert [(r.STREAM, r.step) for r in batch.records] == [
        ("step_time", 7),
        ("step_time", 8),
    ]
    assert batch.skipped == 1


def test_unknown_sampler_yields_nothing():
    batch = records_from_payload(
        _payload("OtherSampler", "StepTimeTable", CASES[0][4])
    )

    assert batch.records == []
    assert batch.skipped == 0


def test_export_writers_are_the_step_time_writer_only():
    # Deliberately a subset of the SQLite projection writers: the other
    # streams get records in a follow-up.
    assert export_records._WRITERS == (step_time,)
    assert set(export_records._WRITERS) < set(
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


@pytest.mark.parametrize("cls", RECORD_TYPES, ids=lambda c: c.STREAM)
def test_doc_table_lists_record_fields(cls):
    text = DOC.read_text(encoding="utf-8")
    section = text.split(f"### {cls.__name__}\n", 1)[1].split("\n#", 1)[0]
    documented = re.findall(r"^\| `(\w+)` \|", section, flags=re.M)

    assert documented == [f.name for f in dataclasses.fields(cls)]
