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


def _sys_payload(global_rank: int = 5) -> dict:
    return {
        "global_rank": global_rank,
        "local_rank": 0,
        "world_size": 2,
        "local_world_size": 1,
        "node_rank": global_rank,
        "hostname": f"worker-{global_rank}",
        "sampler": "SystemSampler",
        "timestamp": 123.0,
        "tables": {
            "SystemTable": [
                {
                    "seq": 1,
                    "ts": 123.0,
                    "cpu": 42.0,
                    "ram_used": 4_000.0,
                    "ram_total": 16_000.0,
                    "gpu_available": False,
                    "gpu_count": 0,
                    "gpus": [],
                }
            ]
        },
    }


def test_sqlite_finalize_budget_clamp_regimes() -> None:
    budget = TraceMLAggregator._sqlite_finalize_budget
    assert budget(0.0) == 0.0
    assert budget(-5.0) == 0.0
    # total < MIN (5s): a fraction of total, floored at the tiny floor.
    assert budget(4.0) == pytest.approx(1.0)
    assert budget(2.0) == pytest.approx(0.5)
    # total >= MIN: 25% clamped into [5, 60].
    assert budget(16.0) == pytest.approx(5.0)  # 4.0 -> clamped up to MIN
    assert budget(40.0) == pytest.approx(10.0)  # 25% lands in-range
    assert budget(400.0) == pytest.approx(60.0)  # clamped down to MAX


def test_split_telemetry_payloads_duplicate_rank_is_idempotent(tmp_path):
    agg = _make_aggregator(tmp_path, writer=_Writer(_ok_result()), tcp=_TCP())
    marker = build_rank_finished_payload(
        global_rank=0, world_size=2, node_rank=0, hostname="worker"
    )
    agg._split_telemetry_payloads([marker, marker])
    agg._split_telemetry_payloads(marker)
    assert agg._finished_ranks_snapshot() == [0]


def test_stop_writes_no_warning_when_all_ranks_finished(monkeypatch, tmp_path):
    control_messages = [
        build_rank_finished_payload(
            global_rank=rank, world_size=2, node_rank=0, hostname="worker"
        )
        for rank in (0, 1)
    ]
    agg = _make_aggregator(
        tmp_path,
        writer=_Writer(_ok_result()),
        tcp=_TCP(messages=control_messages),
    )
    monkeypatch.setattr(
        trace_aggregator,
        "generate_summary",
        lambda *args, **kwargs: _write_fake_summary(tmp_path),
    )

    agg.stop(timeout_sec=0.5)

    agg_dir = tmp_path / "run" / "aggregator"
    assert not (agg_dir / "finalization_warning.json").exists()
    assert not (agg_dir / "finalization_error.json").exists()
    assert get_final_summary_json_path(tmp_path / "run").is_file()


def test_stop_downgrades_post_write_summary_error_to_warning(
    monkeypatch,
    tmp_path,
):
    control_messages = [
        build_rank_finished_payload(
            global_rank=rank, world_size=2, node_rank=0, hostname="worker"
        )
        for rank in (0, 1)
    ]
    agg = _make_aggregator(
        tmp_path,
        writer=_Writer(_ok_result()),
        tcp=_TCP(messages=control_messages),
    )

    def _summary_then_raise(*args, **kwargs):
        _write_fake_summary(tmp_path)
        raise RuntimeError("post-write render boom")

    monkeypatch.setattr(
        trace_aggregator, "generate_summary", _summary_then_raise
    )

    # The artifact is written before the raise, so stop() must NOT fail.
    agg.stop(timeout_sec=0.5)

    agg_dir = tmp_path / "run" / "aggregator"
    assert get_final_summary_json_path(tmp_path / "run").is_file()
    assert not (agg_dir / "finalization_error.json").exists()
    warning = json.loads(
        (agg_dir / "finalization_warning.json").read_text(encoding="utf-8")
    )
    assert "post-write render boom" in warning["generate_summary_error"]


def test_stop_generates_summary_through_real_writer(tmp_path):
    from traceml_ai.aggregator.sqlite_writer import (
        SQLiteWriterConfig,
        SQLiteWriterSimple,
    )

    db_path = tmp_path / "run" / "aggregator" / "telemetry"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    writer = SQLiteWriterSimple(
        SQLiteWriterConfig(path=str(db_path), flush_interval_sec=0.01)
    )
    writer.start()
    for _ in range(3):
        writer.ingest(_sys_payload())
    control_messages = [
        build_rank_finished_payload(
            global_rank=rank, world_size=2, node_rank=0, hostname="worker"
        )
        for rank in (0, 1)
    ]
    agg = _make_aggregator(
        tmp_path, writer=writer, tcp=_TCP(messages=control_messages)
    )

    # Real finalize + real generate_summary (NOT monkeypatched): the literal
    # inverse of #164 -- finalize cleanly, then the summary must appear.
    agg.stop(timeout_sec=10.0)

    assert get_final_summary_json_path(tmp_path / "run").is_file()
    assert not Path(str(db_path) + "-wal").exists()
    assert not Path(str(db_path) + "-shm").exists()
    assert not (
        tmp_path / "run" / "aggregator" / "finalization_error.json"
    ).exists()
