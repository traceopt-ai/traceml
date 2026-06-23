import json
import threading
import time
from pathlib import Path

import pytest

from traceml_ai.aggregator import aggregator_main, trace_aggregator
from traceml_ai.aggregator.sqlite_writer import SQLiteFinalizeResult
from traceml_ai.aggregator.trace_aggregator import (
    TraceMLAggregator,
    TraceMLFinalizationError,
)
from traceml_ai.runtime.settings import TraceMLSettings
from traceml_ai.sdk.protocol import get_final_summary_json_path
from traceml_ai.telemetry.control import build_rank_finished_payload


class _Logger:
    def error(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


class _StoppedThread:
    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


class _Display:
    def stop(self):
        return None


class _TCP:
    def __init__(self, messages=None):
        self._messages = list(messages or [])
        self.stopped = False

    def poll(self):
        while self._messages:
            yield self._messages.pop(0)

    def wait_for_data(self, timeout):
        return False

    def stop(self):
        self.stopped = True


class _Writer:
    def __init__(self, result):
        self.result = result
        self.ingested = []
        self.finalize_timeouts = []

    def ingest(self, payload):
        self.ingested.append(payload)

    def finalize(self, timeout_sec):
        self.finalize_timeouts.append(timeout_sec)
        return self.result

    def stats(self):
        return {"queue_size": 0, "last_error": None}


def _ok_result():
    return SQLiteFinalizeResult(
        ok=True,
        elapsed_sec=0.0,
        enqueued=1,
        written=1,
        dropped=0,
        queue_size=0,
        checkpoint_ok=True,
        error=None,
    )


def _make_aggregator(tmp_path: Path, *, writer: _Writer, tcp: _TCP):
    agg = TraceMLAggregator.__new__(TraceMLAggregator)
    agg._logger = _Logger()
    agg._stop_event = threading.Event()
    agg._settings = TraceMLSettings(
        mode="summary",
        logs_dir=str(tmp_path),
        session_id="run",
        history_enabled=True,
        db_path=str(tmp_path / "run" / "aggregator" / "telemetry"),
        expected_world_size=2,
    )
    agg._started = True
    agg._expected_world_size = 2
    agg._finished_ranks = {}
    agg._drain_lock = threading.Lock()
    agg._thread = _StoppedThread()
    agg._display_driver = _Display()
    agg._tcp_server = tcp
    agg._sqlite_writer = writer
    return agg


def _write_fake_summary(tmp_path: Path):
    path = get_final_summary_json_path(tmp_path / "run")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": "test"}),
        encoding="utf-8",
    )


def test_split_telemetry_payloads_consumes_rank_finished_control(tmp_path):
    writer = _Writer(_ok_result())
    control = build_rank_finished_payload(
        global_rank=1,
        world_size=2,
        node_rank=0,
        hostname="worker",
    )
    agg = _make_aggregator(
        tmp_path,
        writer=writer,
        tcp=_TCP(),
    )

    payloads = agg._split_telemetry_payloads(
        [{"sampler": "SystemSampler"}, control]
    )

    assert payloads == [[{"sampler": "SystemSampler"}]]
    assert sorted(agg._finished_ranks) == [1]


def test_stop_writes_warning_when_rank_finished_markers_are_missing(
    monkeypatch,
    tmp_path,
):
    writer = _Writer(_ok_result())
    control = build_rank_finished_payload(
        global_rank=0,
        world_size=2,
        node_rank=0,
        hostname="worker",
    )
    agg = _make_aggregator(
        tmp_path,
        writer=writer,
        tcp=_TCP(messages=[control]),
    )

    monkeypatch.setattr(
        trace_aggregator,
        "generate_summary",
        lambda *args, **kwargs: _write_fake_summary(tmp_path),
    )

    agg.stop(timeout_sec=0.01)

    warning_path = (
        tmp_path / "run" / "aggregator" / "finalization_warning.json"
    )
    warning = json.loads(warning_path.read_text(encoding="utf-8"))
    assert warning["status"] == "warning"
    assert warning["finished_ranks"] == [0]
    assert warning["missing_ranks"] == [1]


def test_stop_writes_error_and_raises_when_sqlite_finalize_fails(tmp_path):
    writer = _Writer(
        SQLiteFinalizeResult(
            ok=False,
            elapsed_sec=0.0,
            enqueued=1,
            written=0,
            dropped=0,
            queue_size=1,
            checkpoint_ok=False,
            error="boom",
        )
    )
    control_messages = [
        build_rank_finished_payload(
            global_rank=rank,
            world_size=2,
            node_rank=0,
            hostname="worker",
        )
        for rank in (0, 1)
    ]
    agg = _make_aggregator(
        tmp_path,
        writer=writer,
        tcp=_TCP(messages=control_messages),
    )

    with pytest.raises(TraceMLFinalizationError):
        agg.stop(timeout_sec=0.1)

    error_path = tmp_path / "run" / "aggregator" / "finalization_error.json"
    error = json.loads(error_path.read_text(encoding="utf-8"))
    assert error["status"] == "error"
    assert "SQLite history did not finalize" in error["error"]


def test_stop_reserves_positive_sqlite_finalize_budget_when_ranks_missing(
    monkeypatch,
    tmp_path,
):
    class _SlowTCP(_TCP):
        def wait_for_data(self, timeout):
            time.sleep(float(timeout))
            return False

    writer = _Writer(_ok_result())
    agg = _make_aggregator(
        tmp_path,
        writer=writer,
        tcp=_SlowTCP(
            messages=[
                build_rank_finished_payload(
                    global_rank=0,
                    world_size=2,
                    node_rank=0,
                    hostname="worker",
                )
            ]
        ),
    )
    monkeypatch.setattr(
        trace_aggregator,
        "generate_summary",
        lambda *args, **kwargs: _write_fake_summary(tmp_path),
    )

    agg.stop(timeout_sec=0.02)

    assert writer.finalize_timeouts
    assert writer.finalize_timeouts[-1] > 0.0


def test_drain_tcp_is_serialized(tmp_path):
    class _ConcurrentDetectTCP:
        def __init__(self):
            self._lock = threading.Lock()
            self._active = 0
            self.max_active = 0

        def poll(self):
            with self._lock:
                self._active += 1
                self.max_active = max(self.max_active, self._active)
            try:
                time.sleep(0.02)
                yield {"sampler": "UnknownSampler"}
            finally:
                with self._lock:
                    self._active -= 1

    writer = _Writer(_ok_result())
    tcp = _ConcurrentDetectTCP()
    agg = _make_aggregator(tmp_path, writer=writer, tcp=tcp)

    threads = [
        threading.Thread(target=agg._drain_tcp),
        threading.Thread(target=agg._drain_tcp),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert tcp.max_active == 1


def test_aggregator_main_does_not_swallow_keyboard_interrupt_during_stop(
    monkeypatch,
    tmp_path,
):
    class _Handle:
        def stop(self, timeout_sec):
            raise KeyboardInterrupt()

    def fake_start_aggregator(settings, logger, stop_event):
        stop_event.set()
        return _Handle()

    monkeypatch.setenv("TRACEML_LOGS_DIR", str(tmp_path))
    monkeypatch.setenv("TRACEML_SESSION_ID", "run")
    monkeypatch.setenv("TRACEML_UI_MODE", "summary")
    monkeypatch.setenv("TRACEML_FINALIZE_TIMEOUT_SEC", "1")
    monkeypatch.setattr(
        aggregator_main, "setup_error_logger", lambda **kwargs: None
    )
    monkeypatch.setattr(
        aggregator_main, "_install_signal_handlers", lambda event: None
    )
    monkeypatch.setattr(
        aggregator_main,
        "start_aggregator",
        fake_start_aggregator,
    )

    with pytest.raises(KeyboardInterrupt):
        aggregator_main.main()
