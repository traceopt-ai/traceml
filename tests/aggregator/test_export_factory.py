# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""A ``module:factory`` exporter value is imported and called once."""

from __future__ import annotations

import json
import sys
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
    _FailedExporter,
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

    def start(self):
        return None

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


def exiting_factory(context):
    sys.exit(4)


def int_factory(context):
    return 42


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


def _open_failure(sink) -> str:
    """Return the message ``open`` raises; the other calls are inert."""
    assert isinstance(sink, _FailedExporter)
    assert sink.write([object()]) is None
    assert sink.flush() is None
    assert sink.close(dropped_batches=3) is None
    with pytest.raises(RuntimeError) as exc:
        sink.open()
    return str(exc.value)


def test_factory_receives_the_context(tmp_path):
    settings = _settings(
        tmp_path, f"{__name__}:make_exporter", exporter_hostname=True
    )

    sink, streams = _resolve_exporter(settings, tmp_path / "run")

    assert sink is _MADE[0]
    assert streams == ("step_time",)
    assert sink.context == ExporterContext(
        session_root=tmp_path / "run",
        export_dir=tmp_path / "run" / "aggregator" / "export",
        include_hostname=True,
        streams=("step_time",),
    )
    assert not (tmp_path / "run").exists()


def test_factory_value_is_stripped(tmp_path):
    settings = _settings(tmp_path, f" {__name__} : make_exporter ")

    sink, _ = _resolve_exporter(settings, tmp_path)

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
    assert exporter.context.streams == ("step_time",)
    export_dir = tmp_path / "logs" / "run" / "aggregator" / "export"
    assert exporter.context.export_dir == export_dir.resolve()
    assert not export_dir.exists()


def test_factory_exporter_receives_only_the_selected_streams(tmp_path):
    step_memory = build_telemetry_envelope(
        identity=SenderIdentity(
            global_rank=0, local_rank=0, hostname="host-a", pid=4242
        ),
        sampler_name="StepMemorySampler",
        tables={
            "step_memory": [
                {
                    "seq": 1,
                    "timestamp_unix_ns": 1_700_000_000_000_000_000,
                    "device": "cuda:0",
                    "step": 1,
                    "peak_alloc": 1024.0,
                    "peak_resv": 2048.0,
                }
            ]
        },
    )
    process = build_telemetry_envelope(
        identity=SenderIdentity(
            global_rank=0, local_rank=0, hostname="host-a", pid=4242
        ),
        sampler_name="ProcessSampler",
        tables={
            "ProcessTable": [
                {"seq": 1, "timestamp_unix_ns": 1_700_000_000_000_000_000}
            ]
        },
    )
    agg = TraceMLAggregator(
        logger=_Logger(),
        stop_event=threading.Event(),
        settings=_settings(
            tmp_path,
            f"{__name__}:make_exporter",
            export_streams="step_time,step_memory",
        ),
    )
    agg._tcp_server = _TCP(
        [[_step_time_payload(), step_memory, process, _rank_finished()]]
    )
    agg._export_worker.start()

    agg._drain_tcp()
    agg.stop(timeout_sec=5.0)

    (exporter,) = _MADE
    assert exporter.context.streams == ("step_time", "step_memory")
    writes = [call[1] for call in exporter.calls if call[0] == "write"]
    assert [[r.STREAM for r in batch] for batch in writes] == [
        ["step_time", "step_memory"]
    ]
    assert agg._export_worker.stats().skipped_items == 0


@pytest.mark.parametrize(
    "value, failure",
    [
        ("no_such_module_xyz:make", "ModuleNotFoundError"),
        (f"{__name__}:no_such_attr", "AttributeError"),
        (f"{__name__}:raising_factory", "factory broke"),
        (f"{__name__}:none_factory", "factory returned None"),
        (f"{__name__}:exiting_factory", "SystemExit(4)"),
        ("mod:", "is not of the form 'module:factory'"),
        ("a:b:c", "is not of the form 'module:factory'"),
    ],
)
def test_failed_factory_gives_an_exporter_whose_open_raises(
    tmp_path, value, failure
):
    sink, _ = _resolve_exporter(_settings(tmp_path, value), tmp_path)

    message = _open_failure(sink)
    assert message.startswith(f"[TraceML] Exporter {value!r} ")
    assert failure in message


def test_module_that_exits_on_import_does_not_stop_the_aggregator(
    tmp_path, monkeypatch
):
    (tmp_path / "traceml_exiting_exporter.py").write_text(
        "import sys\nsys.exit(3)\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    value = "traceml_exiting_exporter:make"

    sink, _ = _resolve_exporter(_settings(tmp_path, value), tmp_path)

    message = _open_failure(sink)
    assert message.startswith(f"[TraceML] Exporter {value!r} could not be")
    assert "SystemExit(3)" in message
    assert "traceml_exiting_exporter" not in sys.modules


def _run_summary(tmp_path: Path, exporter: str, logger: _Logger) -> Path:
    """Run a started summary-mode aggregator over one step; return its dir."""
    agg = TraceMLAggregator(
        logger=logger,
        stop_event=threading.Event(),
        settings=_settings(
            tmp_path,
            exporter,
            history_enabled=True,
            db_path=str(tmp_path / "telemetry.db"),
        ),
    )
    agg._tcp_server = _TCP([[_step_time_payload(), _rank_finished()]])
    agg.start()
    agg.stop(timeout_sec=10.0)
    return tmp_path / "logs" / "run"


@pytest.mark.parametrize(
    "value, logged",
    [
        ("no_such_module_xyz:make", "no_such_module_xyz:make"),
        (f"{__name__}:no_such_attr", f"{__name__}:no_such_attr"),
        (f"{__name__}:raising_factory", f"{__name__}:raising_factory"),
        (f"{__name__}:none_factory", f"{__name__}:none_factory"),
        # Not an exporter: the worker's first ``open`` call fails.
        (f"{__name__}:int_factory", "int.open failed"),
    ],
)
def test_failed_factory_run_records_the_loss(tmp_path, value, logged):
    logger = _Logger()

    session_root = _run_summary(tmp_path, value, logger)

    (message,) = logger.errors
    assert message.startswith("[TraceML] ")
    assert "export disabled" in message
    assert logged in message
    aggregator_dir = session_root / "aggregator"
    assert not (aggregator_dir / "export").exists()
    summary = json.loads((session_root / "final_summary.json").read_text())
    assert "export" not in summary
    warning = json.loads(
        (aggregator_dir / "finalization_warning.json").read_text()
    )
    assert warning["reason"] == "export_incomplete"
    assert warning["export"] == {
        "written_records": 0,
        "dropped_batches": 0,
        "failed_records": 1,
        "skipped_items": 0,
        "exporter_failed": True,
    }


@pytest.mark.parametrize(
    "value",
    ["no_such_module_xyz:make", f"{__name__}:raising_factory"],
)
def test_failed_factory_is_in_the_error_artifact(tmp_path, value):
    logger = _Logger()
    agg = TraceMLAggregator(
        logger=logger,
        stop_event=threading.Event(),
        settings=_settings(tmp_path, value, mode="cli"),
    )
    agg._tcp_server = _TCP([[_step_time_payload(), _rank_finished()]])
    agg._export_worker.start()

    def finalize(timeout_sec):
        raise RuntimeError("finalize broke")

    agg._sqlite_writer.finalize = finalize

    agg._drain_tcp()
    agg.stop(timeout_sec=5.0)

    (message,) = logger.errors
    assert value in message
    aggregator_dir = tmp_path / "logs" / "run" / "aggregator"
    assert not (aggregator_dir / "export").exists()
    assert sorted(p.name for p in aggregator_dir.glob("finalization_*")) == [
        "finalization_error.json"
    ]
    error = json.loads(
        (aggregator_dir / "finalization_error.json").read_text()
    )
    assert error["export"]["exporter_failed"] is True
    assert error["export"]["failed_records"] == 1
    assert error["export"]["written_records"] == 0
