# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""A ``module:factory`` exporter value is imported and called once."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from traceml_ai.aggregator.export import (
    ExporterContext,
    ExportWorker,
    StepTimeRecord,
)
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
    def __init__(self):
        self.errors = []

    def info(self, *args, **kwargs):
        return None

    def error(self, msg, *args, **kwargs):
        self.errors.append(str(msg))

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


class _RecordingExporter:
    def __init__(self, context):
        self.context = context
        self.calls = []

    def open(self):
        self.calls.append(("open",))

    def write(self, records):
        self.calls.append(("write", list(records)))

    def flush(self):
        self.calls.append(("flush",))

    def close(self, *, dropped_batches=0):
        self.calls.append(("close", dropped_batches))


_MADE: list = []


def make_exporter(context):
    exporter = _RecordingExporter(context)
    _MADE.append(exporter)
    return exporter


def raising_factory(context):
    raise RuntimeError("factory broke")


def none_factory(context):
    return None


@pytest.fixture(autouse=True)
def _clear_made():
    _MADE.clear()
    yield
    _MADE.clear()


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


def _run(
    tmp_path: Path, exporter: str, logger: _Logger, *, fail_finalize: bool
) -> Path:
    """
    Run one aggregator over one step and return its aggregator dir.

    ``fail_finalize`` makes the SQLite finalize raise, so ``stop`` writes
    ``finalization_error.json``, which always carries the export stats.
    """
    agg = TraceMLAggregator(
        logger=logger,
        stop_event=threading.Event(),
        settings=_settings(tmp_path, exporter, mode="cli"),
    )
    agg._tcp_server = _TCP([[_step_time_payload(), _rank_finished()]])
    if agg._export_worker is not None:
        agg._export_worker.start()
    if fail_finalize:

        def finalize(timeout_sec):
            raise RuntimeError("finalize broke")

        agg._sqlite_writer.finalize = finalize

    agg._drain_tcp()
    agg.stop(timeout_sec=5.0)
    return tmp_path / "logs" / "run" / "aggregator"


def test_factory_receives_the_context(tmp_path):
    settings = _settings(
        tmp_path, f"{__name__}:make_exporter", exporter_hostname=True
    )
    logger = _Logger()

    sink = _resolve_exporter(settings, tmp_path / "run", logger)

    assert sink is _MADE[0]
    assert sink.context == ExporterContext(
        session_root=tmp_path / "run",
        export_dir=tmp_path / "run" / "aggregator" / "export",
        include_hostname=True,
    )
    assert logger.errors == []
    assert not (tmp_path / "run").exists()


def test_factory_value_is_stripped(tmp_path):
    settings = _settings(tmp_path, f" {__name__} : make_exporter ")

    sink = _resolve_exporter(settings, tmp_path, _Logger())

    assert sink is _MADE[0]
    assert sink.context.include_hostname is False


def test_factory_exporter_receives_the_worker_records(tmp_path):
    logger = _Logger()
    agg = TraceMLAggregator(
        logger=logger,
        stop_event=threading.Event(),
        settings=_settings(tmp_path, f"{__name__}:make_exporter"),
    )
    assert isinstance(agg._export_worker, ExportWorker)
    agg._tcp_server = _TCP([[_step_time_payload(), _rank_finished()]])
    agg._export_worker.start()

    agg._drain_tcp()
    agg.stop(timeout_sec=5.0)

    (exporter,) = _MADE
    names = [call[0] for call in exporter.calls]
    assert names == ["open", "write", "flush", "close"]
    (record,) = exporter.calls[1][1]
    assert isinstance(record, StepTimeRecord)
    assert record.step == 1
    assert exporter.calls[3] == ("close", 0)
    assert logger.errors == []
    assert not (tmp_path / "logs" / "run" / "aggregator" / "export").exists()


@pytest.mark.parametrize(
    "value, failure",
    [
        ("no_such_module_xyz:make", "ModuleNotFoundError"),
        (f"{__name__}:no_such_attr", "AttributeError"),
        (f"{__name__}:raising_factory", "factory broke"),
        (f"{__name__}:none_factory", "returned None"),
        ("mod:", "module:factory"),
    ],
)
def test_failed_factory_is_logged_once_and_disables_export(
    tmp_path, value, failure
):
    logger = _Logger()

    sink = _resolve_exporter(_settings(tmp_path, value), tmp_path, logger)

    assert sink is None
    (message,) = logger.errors
    assert message.startswith("[TraceML] ")
    assert repr(value) in message
    assert failure in message
    assert "export is disabled" in message


@pytest.mark.parametrize(
    "value",
    [
        "no_such_module_xyz:make",
        f"{__name__}:no_such_attr",
        f"{__name__}:raising_factory",
        f"{__name__}:none_factory",
    ],
)
@pytest.mark.parametrize("fail_finalize", [False, True])
def test_failed_factory_run_matches_a_run_without_export(
    tmp_path, value, fail_finalize
):
    logger = _Logger()

    aggregator_dir = _run(
        tmp_path / "factory", value, logger, fail_finalize=fail_finalize
    )
    baseline_dir = _run(
        tmp_path / "off", "", _Logger(), fail_finalize=fail_finalize
    )

    (message,) = logger.errors
    assert message.startswith("[TraceML] ")
    assert not (aggregator_dir / "export").exists()
    artifacts = sorted(p.name for p in aggregator_dir.glob("finalization_*"))
    baseline = sorted(p.name for p in baseline_dir.glob("finalization_*"))
    assert artifacts == baseline
    assert artifacts == (["finalization_error.json"] if fail_finalize else [])
    for name in artifacts:
        data = json.loads((aggregator_dir / name).read_text())
        assert data["export"] is None
