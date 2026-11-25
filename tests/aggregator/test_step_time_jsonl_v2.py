# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""``step_time.jsonl`` schema 2: one object per event, ``traceml:`` names (#680)."""

from __future__ import annotations

import json

import pytest

from traceml_ai.aggregator.export.json_exporter import JsonExporter
from traceml_ai.aggregator.export.records import StepTimeRecord

IDENTITY = {
    "observed_timestamp_unix_ns": 1_800_000_000_000_000_000,
    "rank": 5,
    "global_rank": 5,
    "local_rank": 1,
    "world_size": 8,
    "local_world_size": 4,
    "node_rank": 1,
    "hostname": "host-a",
    "timestamp_unix_ns": 1_700_000_000_123_456_789,
    "seq": 12,
    "step": 11,
}


def _stats(cpu=None, gpu=None, is_gpu=None, n_calls=None, duration=None):
    return {
        "is_gpu": is_gpu,
        "duration_ms": duration,
        "cpu_ms": cpu,
        "gpu_ms": gpu,
        "n_calls": n_calls,
    }


def _export(tmp_path, events, *, stored=False) -> list[dict]:
    """Write one step-time record and return all lines of step_time.jsonl.

    ``stored`` builds the record the way the aggregator does, from a row whose
    ``events_json`` was serialised with sorted keys.
    """
    if stored:
        row = tuple(IDENTITY.values()) + (json.dumps(events, sort_keys=True),)
        record = StepTimeRecord.from_row(row)
    else:
        record = StepTimeRecord(**IDENTITY, events=events)
    exporter = JsonExporter(tmp_path, streams=["step_time"])
    exporter.open()
    exporter.write([record])
    exporter.close()
    return [
        json.loads(line)
        for line in (tmp_path / "step_time.jsonl").read_text().splitlines()
    ]


def test_cpu_only_event(tmp_path) -> None:
    events = {
        "_traceml_internal:forward_time": {
            "cpu": _stats(8.2, None, False, 1, 8.2)
        }
    }
    line = _export(tmp_path, events)[0]

    assert line["schema_version"] == 2
    assert line["events"] == {
        "traceml:forward_time": {
            "device": "cpu",
            "cpu_ms": 8.2,
            "gpu_ms": None,
            "is_gpu": False,
            "n_calls": 1,
        }
    }


def test_gpu_timing_and_repeated_calls_are_copied_without_recalculation(
    tmp_path,
) -> None:
    events = {
        "_traceml_internal:backward_time": {
            "cuda:0": _stats(2.1, 18.4, True, 3, 18.4)
        }
    }
    value = _export(tmp_path, events)[0]["events"]["traceml:backward_time"]

    assert value == {
        "device": "cuda:0",
        "cpu_ms": 2.1,
        "gpu_ms": 18.4,
        "is_gpu": True,
        "n_calls": 3,
    }
    assert "duration_ms" not in value


def test_missing_stays_null_and_measured_zero_stays_zero(tmp_path) -> None:
    events = {
        "_traceml_internal:h2d_time": {
            "cuda:0": _stats(None, 0.0, True, 0, 0.0)
        }
    }
    value = _export(tmp_path, events)[0]["events"]["traceml:h2d_time"]

    assert value["cpu_ms"] is None
    assert value["gpu_ms"] == 0
    assert value["n_calls"] == 0


def test_each_event_keeps_its_own_device(tmp_path) -> None:
    events = {
        "_traceml_internal:dataloader_next": {
            "cpu": _stats(3.7, None, False, 1)
        },
        "_traceml_internal:forward_time": {
            "cuda:1": _stats(1.0, 9.0, True, 1)
        },
    }
    exported = _export(tmp_path, events)[0]["events"]

    assert exported["traceml:dataloader_next"]["device"] == "cpu"
    assert exported["traceml:forward_time"]["device"] == "cuda:1"


def test_an_event_on_several_devices_exports_the_first_stored_device_only(
    tmp_path,
) -> None:
    events = {
        "_traceml_internal:optimizer_step": {
            "cuda:1": _stats(4.0, 40.0, True, 1),
            "cuda:0": _stats(1.0, 10.0, True, 1),
        }
    }
    value = _export(tmp_path, events, stored=True)[0]["events"][
        "traceml:optimizer_step"
    ]

    # Storage sorts device keys, so the first stored device is cuda:0.
    assert value == {
        "device": "cuda:0",
        "cpu_ms": 1.0,
        "gpu_ms": 10.0,
        "is_gpu": True,
        "n_calls": 1,
    }


def test_an_event_with_no_device_entry_exports_nulls(tmp_path) -> None:
    value = _export(tmp_path, {"_traceml_internal:step_time": {}})[0][
        "events"
    ]["traceml:step_time"]

    assert value == {
        "device": None,
        "cpu_ms": None,
        "gpu_ms": None,
        "is_gpu": None,
        "n_calls": None,
    }


def test_identity_metadata_and_custom_names_are_unchanged(tmp_path) -> None:
    events = {
        "my_custom_phase": {"cpu": _stats(5.0, None, False, 2)},
        "_traceml_internal:step_time": {"cpu": _stats(27.0, None, False, 1)},
    }
    line = _export(tmp_path, events)[0]

    for key, value in IDENTITY.items():
        if key != "hostname":
            assert line[key] == value
    assert sorted(line["events"]) == ["my_custom_phase", "traceml:step_time"]


def test_a_duplicate_exported_name_raises_and_writes_no_line(tmp_path) -> None:
    record = StepTimeRecord(
        **IDENTITY,
        events={
            "_traceml_internal:forward_time": {"cpu": _stats(1.0)},
            "traceml:forward_time": {"cpu": _stats(2.0)},
        },
    )
    exporter = JsonExporter(tmp_path, streams=["step_time"])
    exporter.open()

    with pytest.raises(
        ValueError, match="Duplicate exported event name: traceml:forward_time"
    ):
        exporter.write([record])
    exporter.close()

    lines = [
        json.loads(line)
        for line in (tmp_path / "step_time.jsonl").read_text().splitlines()
    ]
    assert lines == [
        {
            "stream": "step_time",
            "closing": True,
            "schema_version": 2,
            "written_records": 0,
            "dropped_batches": 0,
        }
    ]


def test_other_streams_keep_schema_version_1(tmp_path) -> None:
    exporter = JsonExporter(tmp_path, streams=["step_time", "step_memory"])
    exporter.open()
    exporter.close()

    closing = json.loads((tmp_path / "step_memory.jsonl").read_text())
    assert closing["schema_version"] == 1
