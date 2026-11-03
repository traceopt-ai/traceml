# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""The JSON exporter writes one JSON line per record, one file per stream."""

from __future__ import annotations

import dataclasses
import json

import pytest

from traceml_ai.aggregator.export.json_exporter import JsonExporter
from traceml_ai.aggregator.export.records import StepTimeRecord

EVENTS = {
    "_traceml_internal:forward_time": {
        "cuda:0": {"is_gpu": True, "duration_ms": 9.0, "n_calls": 2}
    }
}


def _record(step: int = 7, **overrides) -> StepTimeRecord:
    fields = {
        "observed_timestamp_unix_ns": 1_800_000_000_000_000_000,
        "rank": 3,
        "global_rank": 3,
        "local_rank": 1,
        "world_size": 8,
        "local_world_size": 4,
        "node_rank": 1,
        "hostname": "host-a",
        "timestamp_unix_ns": 1_700_000_000_123_456_789,
        "seq": step,
        "step": step,
        "events": EVENTS,
    }
    fields.update(overrides)
    return StepTimeRecord(**fields)


def _lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_one_file_per_stream(tmp_path) -> None:
    exporter = JsonExporter(tmp_path)
    exporter.open()
    exporter.write([_record(1), _record(2)])
    exporter.flush()
    exporter.close()

    assert sorted(p.name for p in tmp_path.iterdir()) == ["step_time.jsonl"]


def test_existing_file_refuses_to_overwrite(tmp_path) -> None:
    path = tmp_path / "step_time.jsonl"
    path.write_text("from an earlier run\n")
    exporter = JsonExporter(tmp_path)
    exporter.open()

    with pytest.raises(FileExistsError) as info:
        exporter.write([_record()])

    assert str(path) in str(info.value)
    assert path.read_text() == "from an earlier run\n"


def test_data_lines_carry_the_record(tmp_path) -> None:
    exporter = JsonExporter(tmp_path)
    exporter.open()
    exporter.write([_record(1), _record(2)])
    exporter.close()

    lines = _lines(tmp_path / "step_time.jsonl")
    for line, step in zip(lines[:2], (1, 2)):
        expected = _record(step).to_dict()
        del expected["hostname"]
        assert line == expected
        assert line["stream"] == "step_time"
        assert line["schema_version"] == 1


def test_lines_are_compact_and_parse_alone(tmp_path) -> None:
    exporter = JsonExporter(tmp_path)
    exporter.open()
    exporter.write([_record(1)])
    exporter.close()

    text = (tmp_path / "step_time.jsonl").read_text()
    assert text.endswith("\n")
    for raw in text.splitlines():
        assert ", " not in raw and '": ' not in raw
        assert isinstance(json.loads(raw), dict)


def test_hostname_absent_by_default(tmp_path) -> None:
    exporter = JsonExporter(tmp_path)
    exporter.open()
    exporter.write([_record()])
    exporter.close()

    assert "hostname" not in _lines(tmp_path / "step_time.jsonl")[0]


def test_hostname_present_when_asked(tmp_path) -> None:
    exporter = JsonExporter(tmp_path, include_hostname=True)
    exporter.open()
    exporter.write([_record()])
    exporter.close()

    assert _lines(tmp_path / "step_time.jsonl")[0]["hostname"] == "host-a"


def test_nan_raises_value_error(tmp_path) -> None:
    exporter = JsonExporter(tmp_path)
    exporter.open()
    bad = dataclasses.replace(
        _record(),
        events={"x": {"cuda:0": {"duration_ms": float("nan")}}},
    )

    with pytest.raises(ValueError):
        exporter.write([bad])


def test_closing_line_is_last_and_carries_counts(tmp_path) -> None:
    exporter = JsonExporter(tmp_path)
    exporter.open()
    exporter.write([_record(1), _record(2)])
    exporter.write([_record(3)])
    exporter.close(dropped_batches=4)

    lines = _lines(tmp_path / "step_time.jsonl")
    assert len(lines) == 4
    assert lines[-1] == {
        "stream": "step_time",
        "closing": True,
        "schema_version": 1,
        "written_records": 3,
        "dropped_batches": 4,
    }
    assert all("closing" not in line for line in lines[:-1])


def test_second_close_adds_nothing(tmp_path) -> None:
    exporter = JsonExporter(tmp_path)
    exporter.open()
    exporter.write([_record()])
    exporter.close(dropped_batches=1)
    before = (tmp_path / "step_time.jsonl").read_text()
    exporter.close(dropped_batches=9)

    assert (tmp_path / "step_time.jsonl").read_text() == before


def test_open_creates_a_missing_nested_directory(tmp_path) -> None:
    directory = tmp_path / "a" / "b" / "export"
    exporter = JsonExporter(directory)
    exporter.open()

    assert directory.is_dir()
    exporter.write([_record()])
    exporter.close()
    assert (directory / "step_time.jsonl").exists()
