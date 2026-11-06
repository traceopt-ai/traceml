# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""The aggregator builds, feeds and closes the configured exporter."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from traceml_ai.aggregator.export import ExportWorker, JsonExporter
from traceml_ai.aggregator.trace_aggregator import (
    TraceMLAggregator,
    _resolve_exporter,
)
from traceml_ai.runtime.sender import SenderIdentity
from traceml_ai.runtime.settings import (
    AggregatorTransportSettings,
    TraceMLSettings,
)
from traceml_ai.telemetry.control import build_rank_finished_payload
from traceml_ai.telemetry.envelope import build_telemetry_envelope


class _Logger:
    def info(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


class _TCP:
    def __init__(self, messages):
        self._messages = list(messages)

    def poll(self):
        while self._messages:
            yield self._messages.pop(0)

    def wait_for_data(self, timeout):
        return False

    def open_connections(self):
        return 0

    def stop(self):
        return None


def _settings(tmp_path: Path, exporter: str, **overrides) -> TraceMLSettings:
    values = {
        "mode": "summary",
        "logs_dir": str(tmp_path / "logs"),
        "session_id": "run",
        "history_enabled": False,
        "db_path": str(tmp_path / "telemetry"),
        "exporter": exporter,
        "aggregator": AggregatorTransportSettings(port=0),
    }
    values.update(overrides)
    return TraceMLSettings(**values)


def _step_time_payload() -> dict:
    return build_telemetry_envelope(
        identity=SenderIdentity(
            global_rank=0, local_rank=0, hostname="host-a", pid=4242
        ),
        sampler_name="StepTimeSampler",
        tables={
            "StepTimeTable": [
                {
                    "seq": 1,
                    "timestamp_unix_ns": 1_700_000_000_000_000_000,
                    "step": 1,
                    "events": {},
                }
            ]
        },
    )


def _rank_finished() -> dict:
    return build_rank_finished_payload(
        global_rank=0, world_size=1, node_rank=0, hostname="host-a"
    )


def test_resolve_exporter_is_off_for_an_empty_name(tmp_path):
    assert _resolve_exporter(_settings(tmp_path, ""), tmp_path) is None


def test_resolve_exporter_builds_jsonl_under_the_session(tmp_path):
    sink, _ = _resolve_exporter(_settings(tmp_path, "jsonl"), tmp_path / "run")

    assert isinstance(sink, JsonExporter)
    assert sink._directory == tmp_path / "run" / "aggregator" / "export"
    assert sink._include_hostname is False


def test_resolve_exporter_strips_the_name(tmp_path):
    sink, _ = _resolve_exporter(_settings(tmp_path, " jsonl "), tmp_path)

    assert isinstance(sink, JsonExporter)


def test_resolve_exporter_passes_the_hostname_opt_in(tmp_path):
    settings = _settings(tmp_path, "jsonl", exporter_hostname=True)

    sink, _ = _resolve_exporter(settings, tmp_path)

    assert sink._include_hostname is True


def test_resolve_exporter_rejects_an_unknown_name(tmp_path):
    with pytest.raises(ValueError, match="'bogus'.*jsonl"):
        _resolve_exporter(_settings(tmp_path, "bogus"), tmp_path)


def test_aggregator_rejects_an_unknown_exporter_at_construction(tmp_path):
    with pytest.raises(ValueError, match="'bogus'"):
        TraceMLAggregator(
            logger=_Logger(),
            stop_event=threading.Event(),
            settings=_settings(tmp_path, "bogus"),
        )


def test_aggregator_rejects_an_unknown_export_stream_at_construction(
    tmp_path,
):
    with pytest.raises(ValueError, match="'bogus'"):
        TraceMLAggregator(
            logger=_Logger(),
            stop_event=threading.Event(),
            settings=_settings(tmp_path, "jsonl", export_streams="bogus"),
        )


@pytest.mark.parametrize(
    ("world_size", "capacity"),
    [(1, 4_096), (496, 4_960)],
)
def test_aggregator_sizes_export_queue_for_world_size(
    tmp_path, world_size, capacity
):
    agg = TraceMLAggregator(
        logger=_Logger(),
        stop_event=threading.Event(),
        settings=_settings(tmp_path, "jsonl", expected_world_size=world_size),
    )

    assert agg._export_worker._q.maxsize == capacity


def test_start_starts_the_export_worker_after_sqlite(tmp_path):
    calls = []

    class _Part:
        def __init__(self, name):
            self._name = name

        def start(self):
            calls.append(self._name)

    agg = TraceMLAggregator(
        logger=_Logger(),
        stop_event=threading.Event(),
        settings=_settings(tmp_path, ""),
    )
    agg._tcp_server = _Part("tcp")
    agg._sqlite_writer = _Part("sqlite")
    agg._export_worker = _Part("export")
    agg._display_driver = _Part("display")
    agg._thread = _Part("loop")

    agg.start()

    assert calls == ["tcp", "sqlite", "export", "display", "loop"]


def test_jsonl_exporter_writes_admitted_step_time(tmp_path):
    agg = TraceMLAggregator(
        logger=_Logger(),
        stop_event=threading.Event(),
        settings=_settings(tmp_path, "jsonl"),
    )
    assert isinstance(agg._export_worker, ExportWorker)
    agg._tcp_server = _TCP([[_step_time_payload(), _rank_finished()]])
    agg._export_worker.start()

    agg._drain_tcp()
    agg.stop(timeout_sec=5.0)

    path = tmp_path / "logs" / "run" / "aggregator" / "export"
    lines = [
        json.loads(line)
        for line in (path / "step_time.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(lines) == 2
    assert lines[0]["stream"] == "step_time"
    assert lines[0]["step"] == 1
    assert "hostname" not in lines[0]
    assert lines[1]["closing"] is True
    assert lines[1]["written_records"] == 1
    assert lines[1]["dropped_batches"] == 0
    assert not (
        tmp_path / "logs" / "run" / "aggregator" / "finalization_warning.json"
    ).exists()


def test_jsonl_exporter_keeps_the_hostname_when_asked(tmp_path):
    agg = TraceMLAggregator(
        logger=_Logger(),
        stop_event=threading.Event(),
        settings=_settings(tmp_path, "jsonl", exporter_hostname=True),
    )
    agg._tcp_server = _TCP([[_step_time_payload(), _rank_finished()]])
    agg._export_worker.start()

    agg._drain_tcp()
    agg.stop(timeout_sec=5.0)

    path = tmp_path / "logs" / "run" / "aggregator" / "export"
    data = json.loads(
        (path / "step_time.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert data["stream"] == "step_time"
    assert data["hostname"] == "host-a"


def test_no_exporter_leaves_no_export_directory(tmp_path):
    agg = TraceMLAggregator(
        logger=_Logger(),
        stop_event=threading.Event(),
        settings=_settings(tmp_path, ""),
    )
    assert agg._export_worker is None
    agg._tcp_server = _TCP([[_step_time_payload(), _rank_finished()]])

    agg._drain_tcp()
    agg.stop(timeout_sec=5.0)

    assert not (tmp_path / "logs" / "run" / "aggregator" / "export").exists()
