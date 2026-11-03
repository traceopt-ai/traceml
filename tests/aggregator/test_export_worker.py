# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""The export worker moves admitted payloads to an exporter off-thread."""

from __future__ import annotations

import json
import threading
import time

from traceml_ai.aggregator.export import worker as worker_module
from traceml_ai.aggregator.export.json_exporter import JsonExporter
from traceml_ai.aggregator.export.worker import ExportStats, ExportWorker
from traceml_ai.runtime.sender import SenderIdentity
from traceml_ai.telemetry.envelope import build_telemetry_envelope

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


NAN_EVENTS = {"x": {"cuda:0": {"duration_ms": float("nan")}}}


def _payload(step: int = 7, events: dict = EVENTS) -> dict:
    """One step-time envelope that builds exactly one record."""
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
        sampler_name="StepTimeSampler",
        tables={
            "StepTimeTable": [
                {
                    "seq": step,
                    "timestamp_unix_ns": TS,
                    "step": step,
                    "events": events,
                }
            ]
        },
    )


class _FakeLogger:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def exception(self, message: str, *args: object) -> None:
        self.errors.append(message % args if args else message)

    def error(self, message: str, *args: object) -> None:
        self.errors.append(message % args if args else message)


class _RecordingExporter:
    """Records every call; subclasses sleep or raise."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.records: list = []
        self.open_calls = 0
        self.write_calls = 0
        self.flush_calls = 0
        self.close_calls = 0
        self.close_dropped: list[int] = []
        self.in_write = False
        self.closed_during_write = False

    def open(self) -> None:
        with self._lock:
            self.open_calls += 1

    def write(self, records) -> None:
        with self._lock:
            self.write_calls += 1
            self.in_write = True
        try:
            self._write(list(records))
        finally:
            with self._lock:
                self.in_write = False

    def _write(self, records: list) -> None:
        with self._lock:
            self.records.extend(records)

    def flush(self) -> None:
        with self._lock:
            self.flush_calls += 1

    def close(self, *, dropped_batches: int = 0) -> None:
        with self._lock:
            self.close_calls += 1
            self.close_dropped.append(dropped_batches)
            if self.in_write:
                self.closed_during_write = True

    def snapshot(self) -> list:
        with self._lock:
            return list(self.records)


class _SlowExporter(_RecordingExporter):
    def __init__(self, delay: float) -> None:
        super().__init__()
        self._delay = delay

    def _write(self, records: list) -> None:
        time.sleep(self._delay)
        super()._write(records)


class _FailingExporter(_RecordingExporter):
    def _write(self, records: list) -> None:
        raise RuntimeError("disk full")


class _FailingOpenExporter(_RecordingExporter):
    def open(self) -> None:
        super().open()
        raise OSError("read-only file system")


class _FailingFlushExporter(_RecordingExporter):
    def flush(self) -> None:
        super().flush()
        raise OSError("flush failed")


class _FailingCloseExporter(_RecordingExporter):
    def close(self, *, dropped_batches: int = 0) -> None:
        super().close(dropped_batches=dropped_batches)
        raise OSError("close failed")


def _wait_until(pred, timeout: float = 2.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


def test_records_reach_the_exporter_within_one_tick() -> None:
    exporter = _RecordingExporter()
    worker = ExportWorker(exporter, interval_sec=0.05, logger=_FakeLogger())
    worker.start()
    try:
        worker.submit(_payload(1))
        worker.submit(_payload(2))
        assert _wait_until(lambda: len(exporter.snapshot()) == 2, 0.5)
    finally:
        stats = worker.stop()

    assert [r.step for r in exporter.snapshot()] == [1, 2]
    assert exporter.open_calls == 1
    assert exporter.flush_calls >= 1
    assert exporter.close_calls == 1
    assert stats == ExportStats(
        written_records=2,
        dropped_batches=0,
        failed_records=0,
        skipped_items=0,
        exporter_failed=False,
    )


def test_full_queue_counts_drops_and_submit_returns_at_once() -> None:
    exporter = _RecordingExporter()
    worker = ExportWorker(exporter, max_queue=2, logger=_FakeLogger())

    # Thread not started, so nothing drains and the queue fills.
    start = time.monotonic()
    for step in (1, 2, 3):
        worker.submit(_payload(step))
    elapsed = time.monotonic() - start

    assert elapsed < 0.1
    assert worker.stats().dropped_batches == 1

    worker.start()
    stats = worker.stop(timeout_sec=1.0)
    assert [r.step for r in exporter.snapshot()] == [1, 2]
    assert stats.written_records == 2
    assert stats.dropped_batches == 1


def test_failing_exporter_is_closed_once_and_later_records_fail() -> None:
    logger = _FakeLogger()
    exporter = _FailingExporter()
    worker = ExportWorker(exporter, interval_sec=0.05, logger=logger)
    worker.start()
    try:
        worker.submit(_payload(1))
        assert _wait_until(lambda: worker.stats().exporter_failed)
        assert exporter.close_calls == 1

        worker.submit(_payload(2))
        worker.submit(_payload(3))
        assert _wait_until(lambda: worker.stats().failed_records == 3)
    finally:
        stats = worker.stop()

    assert exporter.write_calls == 1
    assert exporter.close_calls == 1
    assert stats.failed_records == 3
    assert stats.written_records == 0
    assert stats.dropped_batches == 0
    assert stats.exporter_failed is True
    assert len(logger.errors) == 1
    assert logger.errors[0].startswith("[TraceML]")
    assert "_FailingExporter" in logger.errors[0]
    assert "disk full" in logger.errors[0]


def test_stop_is_bounded_with_a_slow_exporter() -> None:
    delay = 0.3
    logger = _FakeLogger()
    exporter = _SlowExporter(delay)
    worker = ExportWorker(exporter, interval_sec=0.01, logger=logger)
    worker.start()
    worker.submit(_payload(0))
    assert _wait_until(lambda: exporter.write_calls == 1)
    for step in range(1, 6):
        worker.submit(_payload(step))

    start = time.monotonic()
    stats = worker.stop(timeout_sec=0.5)
    elapsed = time.monotonic() - start

    # Write 0 ends before the deadline, write 1 is in flight at it, and
    # the four payloads behind it are dropped.
    assert elapsed < 0.5 + 0.15
    assert stats.written_records == 1
    assert stats.dropped_batches == 4
    assert sum("[TraceML]" in e and "dropped" in e for e in logger.errors)

    assert _wait_until(lambda: exporter.close_calls == 1)
    final = worker.stats()
    assert final.written_records == 2
    assert final.dropped_batches == 4
    assert len(exporter.snapshot()) == 2
    assert exporter.close_dropped == [4]


def test_stop_returns_at_the_join_while_a_write_runs() -> None:
    exporter = _SlowExporter(1.0)
    worker = ExportWorker(exporter, interval_sec=0.01, logger=_FakeLogger())
    worker.start()
    worker.submit(_payload(1))
    assert _wait_until(lambda: exporter.write_calls == 1)
    worker.submit(_payload(2))

    start = time.monotonic()
    stats = worker.stop(timeout_sec=0.2)
    elapsed = time.monotonic() - start

    assert elapsed < 0.35
    assert stats.written_records == 0
    assert stats.dropped_batches == 1
    assert exporter.close_calls == 0

    assert _wait_until(lambda: exporter.close_calls == 1)
    assert worker.stats().written_records == 1
    assert exporter.close_dropped == [worker.stats().dropped_batches]
    assert exporter.close_dropped == [1]
    assert exporter.closed_during_write is False
    assert len(exporter.snapshot()) == 1


def test_stop_drains_what_was_queued_before_any_tick() -> None:
    exporter = _RecordingExporter()
    worker = ExportWorker(exporter, interval_sec=60.0, logger=_FakeLogger())
    worker.start()
    worker.submit(_payload(1))
    worker.submit(_payload(2))

    start = time.monotonic()
    stats = worker.stop(timeout_sec=1.0)

    assert time.monotonic() - start < 1.0
    assert [r.step for r in exporter.snapshot()] == [1, 2]
    assert stats.written_records == 2
    assert stats.dropped_batches == 0


def test_skipped_items_counts_a_junk_item() -> None:
    exporter = _RecordingExporter()
    worker = ExportWorker(exporter, interval_sec=60.0, logger=_FakeLogger())
    worker.start()
    worker.submit([_payload(1), {"junk": True}])
    stats = worker.stop()

    assert stats.skipped_items == 1
    assert stats.written_records == 1


def test_close_receives_the_final_dropped_batches() -> None:
    exporter = _RecordingExporter()
    worker = ExportWorker(
        exporter, interval_sec=60.0, max_queue=1, logger=_FakeLogger()
    )
    worker.start()
    for step in (1, 2, 3):
        worker.submit(_payload(step))
    stats = worker.stop()

    assert stats.dropped_batches == 2
    assert stats.written_records == 1
    assert exporter.close_dropped == [2]


def test_submit_after_stop_counts_a_drop() -> None:
    exporter = _RecordingExporter()
    worker = ExportWorker(exporter, interval_sec=60.0, logger=_FakeLogger())
    worker.start()
    worker.stop()
    worker.submit(_payload(1))

    assert worker.stats().dropped_batches == 1
    assert exporter.write_calls == 0


def test_second_stop_reports_a_drop_after_the_first() -> None:
    exporter = _RecordingExporter()
    worker = ExportWorker(exporter, interval_sec=60.0, logger=_FakeLogger())
    worker.start()
    first = worker.stop()
    worker.submit(_payload(1))
    second = worker.stop()

    assert first.dropped_batches == 0
    assert second.dropped_batches == 1
    assert exporter.close_calls == 1


def test_stop_twice_is_safe() -> None:
    exporter = _RecordingExporter()
    worker = ExportWorker(exporter, interval_sec=60.0, logger=_FakeLogger())
    worker.start()
    worker.submit(_payload(1))
    first = worker.stop()
    second = worker.stop()

    assert first == second
    assert exporter.close_calls == 1
    assert exporter.write_calls == 1


def test_raising_open_is_logged_once_and_marks_failed() -> None:
    logger = _FakeLogger()
    exporter = _FailingOpenExporter()
    worker = ExportWorker(exporter, interval_sec=0.05, logger=logger)
    worker.start()
    try:
        assert worker.stats().exporter_failed is True
        worker.submit(_payload(1))
        assert _wait_until(lambda: worker.stats().failed_records == 1)
    finally:
        stats = worker.stop()

    assert stats.exporter_failed is True
    assert stats.failed_records == 1
    assert exporter.write_calls == 0
    assert exporter.close_calls == 0
    assert len(logger.errors) == 1
    assert logger.errors[0].startswith("[TraceML]")
    assert "_FailingOpenExporter" in logger.errors[0]


def test_stop_without_start_drops_the_queue_and_calls_nothing() -> None:
    exporter = _RecordingExporter()
    worker = ExportWorker(exporter, logger=_FakeLogger())
    worker.submit(_payload(1))
    worker.submit(_payload(2))
    stats = worker.stop()

    assert stats.dropped_batches == 2
    assert exporter.open_calls == 0
    assert exporter.write_calls == 0
    assert exporter.close_calls == 0


def test_interval_is_floored_and_non_finite_falls_back() -> None:
    exporter = _RecordingExporter()
    assert ExportWorker(exporter, interval_sec=0)._interval_sec == 0.01
    assert ExportWorker(exporter, interval_sec=-5)._interval_sec == 0.01
    nan = ExportWorker(exporter, interval_sec=float("nan"))
    assert nan._interval_sec == 1.0
    inf = ExportWorker(exporter, interval_sec=float("inf"))
    assert inf._interval_sec == 1.0


def test_raising_flush_keeps_records_written() -> None:
    logger = _FakeLogger()
    exporter = _FailingFlushExporter()
    worker = ExportWorker(exporter, interval_sec=0.01, logger=logger)
    # Queued before the thread starts, so the first tick takes all three.
    for step in (1, 2, 3):
        worker.submit(_payload(step))
    worker.start()
    assert _wait_until(lambda: worker.stats().exporter_failed)
    stats = worker.stop()

    assert exporter.write_calls == 1
    assert stats.written_records == 3
    assert stats.failed_records == 0
    assert stats.exporter_failed is True
    assert exporter.close_calls == 1
    assert len(logger.errors) == 1
    assert logger.errors[0].startswith("[TraceML]")
    assert "flush" in logger.errors[0]


def test_raising_close_is_logged_and_stop_returns() -> None:
    logger = _FakeLogger()
    exporter = _FailingCloseExporter()
    worker = ExportWorker(exporter, interval_sec=60.0, logger=logger)
    worker.start()
    worker.submit(_payload(1))
    stats = worker.stop(timeout_sec=1.0)

    assert stats.written_records == 1
    assert exporter.close_calls == 1
    assert len(logger.errors) == 1
    assert logger.errors[0].startswith("[TraceML]")
    assert "close failed" in logger.errors[0]


def test_raising_conversion_skips_the_whole_list(monkeypatch) -> None:
    def boom(payload):
        raise RuntimeError("bad payload")

    monkeypatch.setattr(worker_module, "records_from_payload", boom)
    exporter = _RecordingExporter()
    worker = ExportWorker(exporter, interval_sec=60.0, logger=_FakeLogger())
    worker.start()
    worker.submit([_payload(1), _payload(2), _payload(3)])
    worker.submit(_payload(4))
    stats = worker.stop()

    assert stats.skipped_items == 4
    assert stats.written_records == 0
    assert exporter.write_calls == 0


def _jsonl(path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_json_exporter_nan_mid_batch_fails_the_whole_batch(tmp_path) -> None:
    logger = _FakeLogger()
    worker = ExportWorker(
        JsonExporter(tmp_path), interval_sec=0.01, logger=logger
    )
    # Queued before the thread starts, so the first tick takes both.
    worker.submit(_payload(1))
    worker.submit(_payload(2, events=NAN_EVENTS))
    worker.start()
    assert _wait_until(lambda: worker.stats().exporter_failed)
    stats = worker.stop()

    assert stats.failed_records == 2
    assert stats.written_records == 0
    assert stats.exporter_failed is True
    lines = _jsonl(tmp_path / "step_time.jsonl")
    data = [line for line in lines if "closing" not in line]
    # ``write`` refused the batch before any OS write, so the closing
    # line is there and matches the data lines on disk.
    assert data == []
    assert lines[-1]["closing"] is True
    assert lines[-1]["written_records"] == len(data)
    assert len(logger.errors) == 1
    assert logger.errors[0].startswith("[TraceML]")


def test_json_exporter_hostname_through_the_worker(tmp_path) -> None:
    exporter = JsonExporter(tmp_path, include_hostname=True)
    worker = ExportWorker(exporter, interval_sec=60.0, logger=_FakeLogger())
    worker.start()
    worker.submit(_payload(1))
    stats = worker.stop()

    lines = _jsonl(tmp_path / "step_time.jsonl")
    assert stats.written_records == 1
    assert lines[0]["hostname"] == "host-a"
    assert lines[-1]["closing"] is True
    assert lines[-1]["written_records"] == 1
