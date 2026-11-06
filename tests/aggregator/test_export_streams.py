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
    EXPORT_STREAM_NAMES,
    RECORD_TYPES,
    ExportWorker,
    JsonExporter,
    StepTimeRecord,
    parse_export_streams,
    records_from_payload,
)
from traceml_ai.aggregator.sqlite_writers import system


def test_default_selects_step_time():
    assert parse_export_streams(DEFAULT_EXPORT_STREAMS) == ("step_time",)


def test_every_setting_name_has_a_stream_mapping():
    from traceml_ai.aggregator.export import _STREAMS_BY_NAME

    assert set(_STREAMS_BY_NAME) == set(EXPORT_STREAM_NAMES)


@pytest.mark.parametrize(
    "streams, message",
    [
        ("all", "not a string"),
        ((), "at least one stream"),
        (("step_time", "gpu"), "Unknown export streams: ['gpu']"),
    ],
)
def test_a_bad_streams_argument_raises_instead_of_exporting_nothing(
    tmp_path, streams, message
):
    calls = [
        lambda: records_from_payload(
            _payload(*CASES[0][2:5]), streams=streams
        ),
        lambda: ExportWorker(_RecordingExporter(), streams=streams),
        lambda: JsonExporter(tmp_path, streams=streams),
    ]
    for call in calls:
        with pytest.raises(ValueError) as exc:
            call()
        assert message in str(exc.value)


def test_every_layer_carries_the_same_default():
    from traceml_ai.config.yaml_loader import BUILT_IN_DEFAULTS
    from traceml_ai.integrations.ray import TraceMLRayConfig
    from traceml_ai.runtime.settings import TraceMLSettings

    assert BUILT_IN_DEFAULTS["export_streams"] == DEFAULT_EXPORT_STREAMS
    assert TraceMLSettings().export_streams == DEFAULT_EXPORT_STREAMS
    assert TraceMLRayConfig().export_streams == DEFAULT_EXPORT_STREAMS


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


def test_unselected_writer_is_not_run_or_counted(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("projection failed")

    monkeypatch.setattr(system, "build_rows", boom)
    payload = _payload(*next(c for c in CASES if c[1] is system)[2:5])

    excluded = records_from_payload(payload, streams=("step_time",))
    kept = records_from_payload(payload)

    assert excluded.records == []
    assert excluded.skipped == 0
    assert kept.skipped == 1


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
