# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""The ``export_streams`` setting selects which records are exported."""

from __future__ import annotations

import json

import pytest

from tests.aggregator.test_export_records import CASES, _payload
from tests.aggregator.test_export_worker import (
    _FakeLogger,
    _RecordingExporter,
)
from traceml_ai.aggregator.export import (
    DEFAULT_EXPORT_STREAMS,
    RECORD_TYPES,
    ExportWorker,
    JsonExporter,
    StepTimeRecord,
    parse_export_streams,
)


def test_default_selects_step_time():
    assert parse_export_streams(DEFAULT_EXPORT_STREAMS) == ("step_time",)


def test_subset_and_order():
    assert parse_export_streams("step_memory, step_time") == (
        "step_time",
        "step_memory",
    )


def test_system_includes_gpu():
    assert parse_export_streams("system") == ("system", "system_gpu")


def test_all_selects_every_stream():
    streams = parse_export_streams("all")

    assert streams == tuple(cls.STREAM for cls in RECORD_TYPES)
    assert len(streams) == 7
    assert "runtime_environment" in streams
    assert "stdout_stderr" in streams


@pytest.mark.parametrize(
    "value, named",
    [
        ("bogus", "'bogus'"),
        ("", "at least one"),
        (" , ", "at least one"),
        ("step_time,bogus", "'bogus'"),
    ],
)
def test_unknown_and_empty_are_rejected(value, named):
    with pytest.raises(ValueError) as exc:
        parse_export_streams(value)

    assert named in str(exc.value)
    assert "'all'" in str(exc.value)


def test_worker_hands_only_selected_records_to_the_exporter():
    exporter = _RecordingExporter()
    worker = ExportWorker(
        exporter,
        interval_sec=60.0,
        logger=_FakeLogger(),
        streams=["step_time"],
    )
    # One payload whose envelopes build step time, step memory and system.
    built = {"step_time", "step_memory", "system"}
    payload = [_payload(*c[2:5]) for c in CASES if c[0].STREAM in built]

    worker.start()
    worker.submit(payload)
    stats = worker.stop()

    records = exporter.snapshot()
    assert len(records) == 1
    assert isinstance(records[0], StepTimeRecord)
    assert stats.written_records == 1
    assert stats.skipped_items == 0
    assert stats.failed_records == 0


def test_json_exporter_creates_only_selected_files(tmp_path):
    exporter = JsonExporter(tmp_path, streams=("system", "system_gpu"))
    exporter.open()
    exporter.close()

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "system.jsonl",
        "system_gpu.jsonl",
    ]
    for stream in ("system", "system_gpu"):
        text = (tmp_path / f"{stream}.jsonl").read_text()
        (closing,) = [json.loads(line) for line in text.splitlines()]
        assert closing["stream"] == stream
        assert closing["closing"] is True
        assert closing["written_records"] == 0
