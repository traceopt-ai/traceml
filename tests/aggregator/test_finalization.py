import json
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from traceml_ai.aggregator import (
    aggregator_main,
    summary_service,
    trace_aggregator,
)
from traceml_ai.aggregator.sqlite_writer import SQLiteFinalizeResult
from traceml_ai.aggregator.summary_service import FinalSummaryService
from traceml_ai.aggregator.trace_aggregator import (
    TraceMLAggregator,
    TraceMLFinalizationError,
)
from traceml_ai.runtime.settings import AggregatorEndpoint, TraceMLSettings
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
    # One live rank connection by default, so a missing rank_finished marker
    # keeps settle waiting for the full budget as it would with a real peer.
    def __init__(self, messages=None, open_connections=1):
        self._messages = list(messages or [])
        self._open_connections = int(open_connections)
        self.stopped = False

    def poll(self):
        while self._messages:
            yield self._messages.pop(0)

    def wait_for_data(self, timeout):
        return False

    def open_connections(self):
        return self._open_connections

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
    agg._foreign_senders = {}
    agg._drain_lock = threading.Lock()
    agg._summary_lock = threading.Lock()
    agg._thread = _StoppedThread()
    agg._display_driver = _Display()
    agg._tcp_server = tcp
    agg._sqlite_writer = writer
    agg._export_worker = None
    return agg


def _write_fake_summary(tmp_path: Path):
    path = get_final_summary_json_path(tmp_path / "run")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "test",
                "generated_at": str(time.time_ns()),
            }
        ),
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


def test_stop_waits_for_a_requested_summary_before_generating(
    monkeypatch,
    tmp_path,
):
    # A final_summary() request is still generating on the loop thread when
    # stop() runs; stop() joins that thread for at most 5 s, so the two
    # paths must not generate at the same time.
    agg = _make_aggregator(
        tmp_path,
        writer=_Writer(_ok_result()),
        tcp=_TCP(open_connections=0),
    )
    service = FinalSummaryService(
        logger=_Logger(),
        session_root=tmp_path / "run",
        db_path="db",
        flush_history=lambda timeout: True,
        settle_telemetry=lambda timeout: True,
        summary_lock=agg._summary_lock,
    )
    state = {"active": 0, "peak": 0, "calls": []}
    state_lock = threading.Lock()
    entered = threading.Event()

    def fake_generate(*args, **kwargs):
        with state_lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            state["calls"].append(kwargs["print_to_stdout"])
        entered.set()
        time.sleep(0.2)
        _write_fake_summary(tmp_path)
        with state_lock:
            state["active"] -= 1

    monkeypatch.setattr(summary_service, "generate_summary", fake_generate)
    monkeypatch.setattr(trace_aggregator, "generate_summary", fake_generate)

    request = threading.Thread(
        target=service._handle_request,
        kwargs={"request_id": "request-1"},
    )
    request.start()
    assert entered.wait(timeout=2.0)
    agg.stop(timeout_sec=5.0)
    request.join(timeout=2.0)

    assert state["peak"] == 1
    # The request path ran first; stop() still refreshed the summary.
    assert state["calls"] == [False, True]


def test_stop_fails_when_a_requested_summary_never_finishes(
    monkeypatch,
    tmp_path,
):
    agg = _make_aggregator(
        tmp_path,
        writer=_Writer(_ok_result()),
        tcp=_TCP(open_connections=0),
    )
    generated = []
    monkeypatch.setattr(
        trace_aggregator,
        "generate_summary",
        lambda *args, **kwargs: generated.append(True),
    )

    agg._summary_lock.acquire()
    try:
        with pytest.raises(TraceMLFinalizationError, match="still"):
            agg.stop(timeout_sec=0.01)
    finally:
        agg._summary_lock.release()

    assert generated == []
    error_path = tmp_path / "run" / "aggregator" / "finalization_error.json"
    assert "still" in json.loads(error_path.read_text())["error"]


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


def _read_finalization_warning(tmp_path: Path) -> dict:
    path = tmp_path / "run" / "aggregator" / "finalization_warning.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _rank_finished(rank: int) -> dict:
    return build_rank_finished_payload(
        global_rank=rank,
        world_size=2,
        node_rank=0,
        hostname="worker",
    )


def test_settle_stops_waiting_once_no_rank_connection_is_open(
    monkeypatch,
    tmp_path,
):
    # A killed rank never sends rank_finished. Before this, settle waited out
    # its whole budget for it (240 s at the default finalize timeout) with the
    # aggregator port still bound, so a SIGTERM looked like a hang.
    writer = _Writer(_ok_result())
    agg = _make_aggregator(
        tmp_path,
        writer=writer,
        tcp=_TCP(messages=[_rank_finished(0)], open_connections=0),
    )
    monkeypatch.setattr(
        trace_aggregator,
        "generate_summary",
        lambda *args, **kwargs: _write_fake_summary(tmp_path),
    )

    started = time.monotonic()
    agg.stop(timeout_sec=30.0)
    elapsed = time.monotonic() - started

    assert elapsed < 5.0
    warning = _read_finalization_warning(tmp_path)
    assert warning["reason"] == "ranks_disconnected"
    assert warning["finished_ranks"] == [0]
    assert warning["missing_ranks"] == [1]


def test_settle_keeps_the_full_budget_while_a_rank_is_connected(
    monkeypatch,
    tmp_path,
):
    # A rank that is still connected may still send late telemetry, which is
    # what settle exists for on multi-node jobs, so it keeps the whole budget.
    class _SlowTCP(_TCP):
        def wait_for_data(self, timeout):
            time.sleep(float(timeout))
            return False

    writer = _Writer(_ok_result())
    agg = _make_aggregator(
        tmp_path,
        writer=writer,
        tcp=_SlowTCP(messages=[_rank_finished(0)], open_connections=1),
    )
    monkeypatch.setattr(
        trace_aggregator,
        "generate_summary",
        lambda *args, **kwargs: _write_fake_summary(tmp_path),
    )
    budget = 0.6
    settle_budget = budget - TraceMLAggregator._sqlite_finalize_budget(budget)

    started = time.monotonic()
    agg.stop(timeout_sec=budget)
    elapsed = time.monotonic() - started

    assert elapsed >= 0.8 * settle_budget
    warning = _read_finalization_warning(tmp_path)
    assert warning["reason"] == "timeout"
    assert warning["missing_ranks"] == [1]


def test_settle_drains_frames_that_arrive_as_the_last_rank_disconnects(
    monkeypatch,
    tmp_path,
):
    # Frames a dying rank sent just before its socket closed are still in
    # flight when the connection count reaches zero. Settle must take them
    # before it finalizes, or the early exit would lose telemetry.
    late_frame = {"sampler": "SystemSampler", "late": True}

    class _LateFrameTCP(_TCP):
        def __init__(self):
            super().__init__(open_connections=0)
            self._late_pending = True

        def wait_for_data(self, timeout):
            if self._late_pending:
                self._late_pending = False
                self._messages.append(late_frame)
                return True
            return False

    writer = _Writer(_ok_result())
    agg = _make_aggregator(tmp_path, writer=writer, tcp=_LateFrameTCP())
    monkeypatch.setattr(
        trace_aggregator,
        "generate_summary",
        lambda *args, **kwargs: _write_fake_summary(tmp_path),
    )

    agg.stop(timeout_sec=30.0)

    assert late_frame in writer.ingested
    assert _read_finalization_warning(tmp_path)["reason"] == (
        "ranks_disconnected"
    )


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
        "tables": {
            "SystemTable": [
                {
                    "seq": 1,
                    "timestamp_unix_ns": 123_000_000_000,
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


def test_stop_rejects_stale_summary_when_refresh_fails(
    monkeypatch,
    tmp_path,
):
    agg = _make_aggregator(
        tmp_path,
        writer=_Writer(_ok_result()),
        tcp=_TCP(),
    )
    _write_fake_summary(tmp_path)
    stale_payload = json.loads(
        get_final_summary_json_path(tmp_path / "run").read_text(
            encoding="utf-8"
        )
    )

    def _raise_before_write(*args, **kwargs):
        raise RuntimeError("generation boom")

    monkeypatch.setattr(
        trace_aggregator,
        "generate_summary",
        _raise_before_write,
    )

    with pytest.raises(
        TraceMLFinalizationError,
        match="not refreshed",
    ):
        agg.stop(timeout_sec=0.5)

    current_payload = json.loads(
        get_final_summary_json_path(tmp_path / "run").read_text(
            encoding="utf-8"
        )
    )
    assert current_payload == stale_payload


def test_fatal_summary_failure_is_logged_once_at_process_boundary(
    monkeypatch,
    tmp_path,
):
    aggregator = _make_aggregator(
        tmp_path,
        writer=_Writer(_ok_result()),
        tcp=_TCP(),
    )
    logger = Mock()
    aggregator._logger = logger
    monkeypatch.setattr(
        trace_aggregator,
        "generate_summary",
        Mock(side_effect=RuntimeError("generation failed")),
    )
    monkeypatch.setattr(
        aggregator_main,
        "_install_signal_handlers",
        lambda _event: None,
    )

    handle = Mock()
    handle.endpoint = AggregatorEndpoint("127.0.0.1", 1234, "run")
    handle.stop.side_effect = aggregator.stop

    def _start(_settings, *, logger, stop_event):
        stop_event.set()
        return handle

    monkeypatch.setattr(aggregator_main, "start_aggregator", _start)

    rc = aggregator_main.run_aggregator(
        TraceMLSettings(
            mode="summary",
            logs_dir=str(tmp_path),
            session_id="run",
            finalize_timeout_sec=0.0,
        ),
        logger=logger,
    )

    assert rc == 1
    logger.error.assert_called_once()
    assert logger.error.call_count + logger.exception.call_count == 1
    assert logger.error.call_args.args[0] == (
        "[TraceML] Aggregator exiting due to error"
    )


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


# Export worker shutdown inside stop().


def _export_stats(**overrides):
    from traceml_ai.aggregator.export import ExportStats

    values = {
        "written_records": 4,
        "dropped_batches": 0,
        "failed_records": 0,
        "skipped_items": 0,
        "exporter_failed": False,
    }
    values.update(overrides)
    return ExportStats(**values)


class _LoggedTCP(_TCP):
    def __init__(self, calls):
        super().__init__()
        self._calls = calls

    def stop(self):
        self._calls.append("tcp_stop")
        super().stop()


class _LoggedWriter(_Writer):
    def __init__(self, calls):
        super().__init__(_ok_result())
        self._calls = calls

    def finalize(self, timeout_sec):
        self._calls.append("sqlite_finalize")
        return super().finalize(timeout_sec)


class _FakeExportWorker:
    def __init__(self, calls, stats):
        self._calls = calls
        self._stats = stats
        self.stop_timeouts = []

    def submit(self, payload):
        return None

    def stop(self, timeout_sec):
        self._calls.append("export_stop")
        self.stop_timeouts.append(timeout_sec)
        return self._stats

    def stats(self):
        return self._stats


class _LateFailureExportWorker(_FakeExportWorker):
    """``stop`` returns clean counts; a call still running fails later."""

    def stop(self, timeout_sec):
        clean = super().stop(timeout_sec)
        self._stats = _export_stats(failed_records=1, exporter_failed=True)
        return clean


def _export_aggregator(monkeypatch, tmp_path, *, stats=None, worker=True):
    """An aggregator whose settle, TCP stop and finalize log their order."""
    calls = []
    agg = _make_aggregator(
        tmp_path, writer=_LoggedWriter(calls), tcp=_LoggedTCP(calls)
    )
    if worker:
        agg._export_worker = _FakeExportWorker(
            calls, stats if stats is not None else _export_stats()
        )
    settle_budgets = []

    def _settle(timeout_sec):
        calls.append("settle")
        settle_budgets.append(timeout_sec)
        return None

    agg._settle_end_of_run_telemetry = _settle
    monkeypatch.setattr(
        trace_aggregator,
        "generate_summary",
        lambda *args, **kwargs: _write_fake_summary(tmp_path),
    )
    return agg, calls, settle_budgets


def _warning_path(tmp_path):
    return tmp_path / "run" / "aggregator" / "finalization_warning.json"


def test_export_worker_stops_after_tcp_and_before_sqlite_finalize(
    monkeypatch, tmp_path
):
    agg, calls, _ = _export_aggregator(monkeypatch, tmp_path)

    agg.stop(timeout_sec=10.0)

    assert calls == ["settle", "tcp_stop", "export_stop", "sqlite_finalize"]
    assert agg._export_worker.stop_timeouts == [pytest.approx(1.0, abs=0.01)]


@pytest.mark.parametrize(
    ("left", "expected"),
    [(0.0, 0.0), (0.1, 0.1), (0.5, 0.5), (10.0, 1.0), (40.0, 2.0)],
)
def test_export_stop_budget_is_a_slice_of_the_time_left(left, expected):
    assert TraceMLAggregator._export_stop_budget(left) == pytest.approx(
        expected
    )


def test_zero_timeout_gives_export_no_time(monkeypatch, tmp_path):
    agg, _, _ = _export_aggregator(monkeypatch, tmp_path)

    agg.stop(timeout_sec=0.0)

    assert agg._export_worker.stop_timeouts == [0.0]


def test_small_timeout_never_gives_export_more_than_it(monkeypatch, tmp_path):
    agg, _, _ = _export_aggregator(monkeypatch, tmp_path)

    agg.stop(timeout_sec=0.1)

    (budget,) = agg._export_worker.stop_timeouts
    assert 0.0 <= budget <= 0.1


def test_export_budget_leaves_the_sqlite_budget_unchanged(
    monkeypatch, tmp_path
):
    # A frozen clock makes every remaining() equal the full timeout, so the
    # budgets are compared exactly.
    monkeypatch.setattr(
        trace_aggregator,
        "time",
        type("_Clock", (), {"monotonic": staticmethod(lambda: 100.0)}),
    )
    real_budget = TraceMLAggregator._sqlite_finalize_budget
    budget_calls = []

    def _spy_budget(timeout_sec):
        result = real_budget(timeout_sec)
        budget_calls.append((timeout_sec, result))
        return result

    monkeypatch.setattr(
        TraceMLAggregator, "_sqlite_finalize_budget", staticmethod(_spy_budget)
    )
    runs = {}
    for worker in (False, True):
        root = tmp_path / str(worker)
        agg, _, settle = _export_aggregator(monkeypatch, root, worker=worker)
        agg.stop(timeout_sec=40.0)
        runs[worker] = (settle, agg._sqlite_writer.finalize_timeouts)

    assert budget_calls == [(40.0, 10.0), (40.0, 10.0)]
    assert runs[False][1] == runs[True][1] == [40.0]
    assert runs[False][0] == [30.0]
    assert runs[True][0] == [28.0]


def test_export_losses_write_a_warning_without_failing_summary_mode(
    monkeypatch, tmp_path
):
    stats = _export_stats(failed_records=3, exporter_failed=True)
    agg, _, _ = _export_aggregator(monkeypatch, tmp_path, stats=stats)
    assert agg._settings.mode == "summary"

    agg.stop(timeout_sec=10.0)

    warning = json.loads(_warning_path(tmp_path).read_text(encoding="utf-8"))
    assert warning["status"] == "warning"
    assert warning["export"] == {
        "written_records": 4,
        "dropped_batches": 0,
        "failed_records": 3,
        "skipped_items": 0,
        "exporter_failed": True,
    }
    assert not (
        tmp_path / "run" / "aggregator" / "finalization_error.json"
    ).exists()


def test_dropped_export_batches_join_an_existing_warning(
    monkeypatch, tmp_path
):
    stats = _export_stats(dropped_batches=2)
    agg, _, _ = _export_aggregator(monkeypatch, tmp_path, stats=stats)
    agg._settle_end_of_run_telemetry = lambda timeout_sec: {
        "status": "warning",
        "completed_at": "now",
        "missing_ranks": [1],
    }

    agg.stop(timeout_sec=10.0)

    warning = json.loads(_warning_path(tmp_path).read_text(encoding="utf-8"))
    assert warning["missing_ranks"] == [1]
    assert warning["export"]["dropped_batches"] == 2


def test_skipped_export_items_alone_write_a_warning(monkeypatch, tmp_path):
    stats = _export_stats(written_records=0, skipped_items=2)
    agg, _, _ = _export_aggregator(monkeypatch, tmp_path, stats=stats)

    agg.stop(timeout_sec=10.0)

    warning = json.loads(_warning_path(tmp_path).read_text(encoding="utf-8"))
    assert warning["reason"] == "export_incomplete"
    assert warning["message"] == (
        "Telemetry export did not write every admitted record."
    )
    assert "SQLite" not in warning["message"]
    assert warning["export"] == {
        "written_records": 0,
        "dropped_batches": 0,
        "failed_records": 0,
        "skipped_items": 2,
        "exporter_failed": False,
    }


def test_export_stats_are_read_again_after_sqlite_finalize(
    monkeypatch, tmp_path
):
    agg, calls, _ = _export_aggregator(monkeypatch, tmp_path)
    agg._export_worker = _LateFailureExportWorker(calls, _export_stats())

    agg.stop(timeout_sec=10.0)

    warning = json.loads(_warning_path(tmp_path).read_text(encoding="utf-8"))
    assert warning["reason"] == "export_incomplete"
    assert warning["export"]["failed_records"] == 1
    assert warning["export"]["exporter_failed"] is True


def test_error_artifact_reads_the_export_stats_again(monkeypatch, tmp_path):
    agg, calls, _ = _export_aggregator(monkeypatch, tmp_path)
    agg._export_worker = _LateFailureExportWorker(calls, _export_stats())
    agg._sqlite_writer.result = SQLiteFinalizeResult(
        ok=False,
        elapsed_sec=0.0,
        enqueued=1,
        written=0,
        dropped=0,
        queue_size=1,
        checkpoint_ok=False,
        error="boom",
    )

    with pytest.raises(TraceMLFinalizationError):
        agg.stop(timeout_sec=10.0)

    error = json.loads(
        (
            tmp_path / "run" / "aggregator" / "finalization_error.json"
        ).read_text(encoding="utf-8")
    )
    assert error["export"]["exporter_failed"] is True


def test_clean_export_writes_no_export_warning(monkeypatch, tmp_path):
    agg, _, _ = _export_aggregator(monkeypatch, tmp_path)

    agg.stop(timeout_sec=10.0)

    assert not _warning_path(tmp_path).exists()


def test_export_stats_reach_the_error_artifact(monkeypatch, tmp_path):
    agg, _, _ = _export_aggregator(monkeypatch, tmp_path)
    agg._sqlite_writer.result = SQLiteFinalizeResult(
        ok=False,
        elapsed_sec=0.0,
        enqueued=1,
        written=0,
        dropped=0,
        queue_size=1,
        checkpoint_ok=False,
        error="boom",
    )

    with pytest.raises(TraceMLFinalizationError):
        agg.stop(timeout_sec=10.0)

    error = json.loads(
        (
            tmp_path / "run" / "aggregator" / "finalization_error.json"
        ).read_text(encoding="utf-8")
    )
    assert error["export"]["written_records"] == 4


def test_export_worker_stops_even_when_settle_raises(monkeypatch, tmp_path):
    agg, calls, _ = _export_aggregator(monkeypatch, tmp_path)

    def _settle(timeout_sec):
        raise RuntimeError("settle failed")

    agg._settle_end_of_run_telemetry = _settle

    with pytest.raises(RuntimeError, match="settle failed"):
        agg.stop(timeout_sec=10.0)

    assert calls == ["export_stop"]


def test_settle_runs_for_export_when_history_is_disabled(
    monkeypatch, tmp_path
):
    from dataclasses import replace

    agg, calls, _ = _export_aggregator(monkeypatch, tmp_path)
    agg._settings = replace(agg._settings, mode="cli", history_enabled=False)

    agg.stop(timeout_sec=10.0)

    assert calls == ["settle", "tcp_stop", "export_stop", "sqlite_finalize"]


def test_settle_is_skipped_without_history_or_export(monkeypatch, tmp_path):
    from dataclasses import replace

    agg, calls, _ = _export_aggregator(monkeypatch, tmp_path, worker=False)
    agg._settings = replace(agg._settings, mode="cli", history_enabled=False)

    agg.stop(timeout_sec=10.0)

    assert calls == ["tcp_stop", "sqlite_finalize"]
