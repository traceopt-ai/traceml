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

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "process.jsonl",
        "runtime_environment.jsonl",
        "stdout_stderr.jsonl",
        "step_memory.jsonl",
        "step_time.jsonl",
        "system.jsonl",
        "system_gpu.jsonl",
    ]


def test_open_creates_each_stream_file_empty(tmp_path) -> None:
    exporter = JsonExporter(tmp_path)
    exporter.open()

    path = tmp_path / "step_time.jsonl"
    assert path.exists()
    assert path.read_text() == ""
    exporter.close()


def test_existing_file_is_refused_at_open(tmp_path) -> None:
    path = tmp_path / "step_time.jsonl"
    path.write_text("from an earlier run\n")
    exporter = JsonExporter(tmp_path)

    with pytest.raises(FileExistsError) as info:
        exporter.open()

    assert str(path) in str(info.value)
    assert path.read_text() == "from an earlier run\n"


def test_zero_records_leaves_only_the_closing_line(tmp_path) -> None:
    exporter = JsonExporter(tmp_path)
    exporter.open()
    exporter.close()

    assert _lines(tmp_path / "step_time.jsonl") == [
        {
            "stream": "step_time",
            "closing": True,
            "schema_version": 1,
            "written_records": 0,
            "dropped_batches": 0,
        }
    ]


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


def test_nan_in_a_batch_writes_no_line_of_it(tmp_path) -> None:
    exporter = JsonExporter(tmp_path)
    exporter.open()
    bad = dataclasses.replace(
        _record(2),
        events={"x": {"cuda:0": {"duration_ms": float("nan")}}},
    )

    with pytest.raises(ValueError):
        exporter.write([_record(1), bad])
    exporter.close()

    lines = _lines(tmp_path / "step_time.jsonl")
    assert len(lines) == 1
    assert lines[0]["closing"] is True
    assert lines[0]["written_records"] == 0


def test_write_after_close_raises(tmp_path) -> None:
    exporter = JsonExporter(tmp_path)
    exporter.open()
    exporter.close()

    with pytest.raises(RuntimeError, match="JsonExporter is closed"):
        exporter.write([_record()])


def test_open_after_close_raises(tmp_path) -> None:
    exporter = JsonExporter(tmp_path)
    exporter.open()
    exporter.close()

    with pytest.raises(RuntimeError, match="JsonExporter is closed"):
        exporter.open()


def test_write_before_open_raises_value_error(tmp_path) -> None:
    exporter = JsonExporter(tmp_path)

    with pytest.raises(ValueError):
        exporter.write([_record()])


def test_failed_os_write_gets_no_closing_line(tmp_path, monkeypatch) -> None:
    exporter = JsonExporter(tmp_path)
    exporter.open()
    exporter.write([_record(1)])
    exporter.flush()
    handle = exporter._files["step_time"]

    def broken_write(text):
        raise OSError("no space left on device")

    monkeypatch.setattr(handle, "write", broken_write)
    with pytest.raises(OSError):
        exporter.write([_record(2)])
    exporter.close(dropped_batches=1)

    assert handle.closed
    lines = _lines(tmp_path / "step_time.jsonl")
    assert [line["step"] for line in lines] == [1]
    assert all("closing" not in line for line in lines)


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
    assert (directory / "step_time.jsonl").exists()
    exporter.close()
