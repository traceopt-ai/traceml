from __future__ import annotations

import socket
import sys
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock

import pytest

from traceml_ai.runtime import lifecycle
from traceml_ai.runtime.runtime import TraceMLRuntime
from traceml_ai.runtime.settings import AggregatorEndpoint, TraceMLSettings
from traceml_ai.telemetry.control import (
    build_hello_ack_payload,
    build_hello_payload,
    is_hello,
)
from traceml_ai.transport.tcp_transport import TCPConfig, TCPServer


@dataclass
class _FakeAggregator:
    logger: Any
    stop_event: Any
    settings: TraceMLSettings
    starts: int = 0
    stops: int = 0

    @property
    def endpoint(self) -> AggregatorEndpoint:
        return AggregatorEndpoint(
            host=self.settings.aggregator.connect_host,
            port=43210,
            session_id=self.settings.session_id or "default",
        )

    def start(self) -> None:
        self.starts += 1

    def stop(self, timeout_sec: float) -> None:
        self.stops += 1


def test_start_aggregator_returns_idempotent_handle(
    monkeypatch,
    tmp_path,
) -> None:
    created: list[_FakeAggregator] = []

    def _factory(*, logger, stop_event, settings):
        agg = _FakeAggregator(
            logger=logger,
            stop_event=stop_event,
            settings=settings,
        )
        created.append(agg)
        return agg

    monkeypatch.setattr(lifecycle, "_build_aggregator", _factory)

    handle = lifecycle.start_aggregator(
        TraceMLSettings(
            mode="summary",
            logs_dir=str(tmp_path),
            session_id="ray-test",
        )
    )

    assert handle.endpoint.host == "127.0.0.1"
    assert handle.endpoint.port == 43210
    assert handle.endpoint.session_id == "ray-test"
    assert handle.session_root == tmp_path / "ray-test"
    assert handle.db_path == tmp_path / "ray-test" / "aggregator" / "telemetry"
    assert created[0].starts == 1

    handle.stop()
    handle.stop()

    assert handle.stop_event.is_set()
    assert created[0].stops == 1


def test_start_aggregator_logs_endpoint_failure_once(monkeypatch, tmp_path):
    class _EndpointFailureAggregator(_FakeAggregator):
        @property
        def endpoint(self):
            raise RuntimeError("endpoint unavailable")

    created = []

    def _factory(*, logger, stop_event, settings):
        aggregator = _EndpointFailureAggregator(
            logger=logger,
            stop_event=stop_event,
            settings=settings,
        )
        created.append(aggregator)
        return aggregator

    logger = Mock()
    monkeypatch.setattr(lifecycle, "_build_aggregator", _factory)

    with pytest.raises(RuntimeError, match="endpoint unavailable"):
        lifecycle.start_aggregator(
            TraceMLSettings(logs_dir=str(tmp_path), session_id="run"),
            logger=logger,
        )

    logger.error.assert_called_once()
    assert created[0].stop_event.is_set()
    assert created[0].stops == 1


class _FakeRuntime:
    """Minimal TraceMLRuntime stand-in for start_runtime() tests."""

    def __init__(self, settings: TraceMLSettings | None = None) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


def test_runtime_start_does_not_replace_python_streams() -> None:
    stdout = sys.stdout
    stderr = sys.stderr
    runtime = TraceMLRuntime.__new__(TraceMLRuntime)
    runtime._exporter = Mock()
    runtime._sampler_thread = Mock()

    runtime.start()

    assert sys.stdout is stdout
    assert sys.stderr is stderr


def test_wait_for_aggregator_true_when_listening() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        assert lifecycle.wait_for_aggregator(
            "127.0.0.1", port, timeout_sec=2.0, poll_interval_sec=0.05
        )
    finally:
        server.close()


def test_wait_for_aggregator_false_when_closed() -> None:
    # Bind then close to obtain a port that is (very likely) not listening.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    assert not lifecycle.wait_for_aggregator(
        "127.0.0.1", port, timeout_sec=0.3, poll_interval_sec=0.05
    )


def test_start_runtime_registers_active_handle(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(lifecycle, "TraceMLRuntime", _FakeRuntime)
    lifecycle._ACTIVE_RUNTIME_HANDLE = None
    try:
        handle = lifecycle.start_runtime(
            TraceMLSettings(logs_dir=str(tmp_path), session_id="reg-test"),
            fail_open=False,
        )
        assert lifecycle.get_active_runtime_handle() is handle
        assert handle.runtime.started is True
    finally:
        lifecycle._ACTIVE_RUNTIME_HANDLE = None


class _BoomRuntime:
    """TraceMLRuntime stand-in whose start() fails, to test fail_open."""

    def __init__(self, settings: TraceMLSettings | None = None) -> None:
        pass

    def start(self) -> None:
        raise RuntimeError("boom")

    def stop(self) -> None:
        pass


def test_start_runtime_fail_open_registers_noop_handle(
    monkeypatch, tmp_path, capsys
) -> None:
    # A failed start with fail_open=True must still register a no-op active
    # handle so a later traceml.init() does not try to start the runtime again,
    # and it must detach LOUDLY (one stderr warning) so a silently dark
    # integration is never mistaken for a healthy run.
    monkeypatch.setattr(lifecycle, "TraceMLRuntime", _BoomRuntime)
    lifecycle._ACTIVE_RUNTIME_HANDLE = None
    try:
        handle = lifecycle.start_runtime(
            TraceMLSettings(logs_dir=str(tmp_path), session_id="fail-open"),
            fail_open=True,
        )
        assert lifecycle.get_active_runtime_handle() is handle
        assert isinstance(handle.runtime, lifecycle.NoOpRuntime)
        assert "[TraceML]" in capsys.readouterr().err  # detached loudly
    finally:
        lifecycle._ACTIVE_RUNTIME_HANDLE = None


@pytest.mark.parametrize("session_id", [".", "..", "a/../..", "/outside"])
def test_start_aggregator_rejects_session_id_outside_logs_dir(
    monkeypatch, tmp_path, session_id: str
) -> None:
    build = Mock()
    monkeypatch.setattr(lifecycle, "_build_aggregator", build)

    with pytest.raises(ValueError, match="session_id"):
        lifecycle.start_aggregator(
            TraceMLSettings(
                mode="summary",
                logs_dir=str(tmp_path / "logs"),
                session_id=session_id,
            )
        )

    build.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("session_id", [".", "..", "a/../.."])
def test_start_runtime_rejects_session_id_outside_logs_dir(
    monkeypatch, tmp_path, session_id: str
) -> None:
    monkeypatch.setattr(lifecycle, "TraceMLRuntime", _FakeRuntime)
    settings = TraceMLSettings(
        logs_dir=str(tmp_path / "logs"), session_id=session_id
    )
    lifecycle._ACTIVE_RUNTIME_HANDLE = None
    try:
        with pytest.raises(ValueError, match="session_id"):
            lifecycle.start_runtime(settings, fail_open=False)

        lifecycle._ACTIVE_RUNTIME_HANDLE = None
        handle = lifecycle.start_runtime(settings, fail_open=True)
        assert isinstance(handle.runtime, lifecycle.NoOpRuntime)
    finally:
        lifecycle._ACTIVE_RUNTIME_HANDLE = None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("session_id", ["exp 1", "team/exp1"])
def test_start_aggregator_keeps_session_ids_the_cli_would_refuse(
    monkeypatch, tmp_path, session_id: str
) -> None:
    # Only ids whose folder leaves the logs dir are refused here; the CLI's
    # single-segment rule is not imposed on existing Python API callers.
    monkeypatch.setattr(lifecycle, "_build_aggregator", _FakeAggregator)

    handle = lifecycle.start_aggregator(
        TraceMLSettings(
            mode="summary", logs_dir=str(tmp_path), session_id=session_id
        )
    )

    assert handle.session_root == (tmp_path / session_id).resolve()
    handle.stop()


class _FlushableRuntime:
    """Runtime stand-in that tracks stop state and flush calls."""

    def __init__(self) -> None:
        self.stopped = False
        self.flushes: list[float] = []

    def flush_pending_telemetry(self, timeout_sec: float) -> bool:
        self.flushes.append(timeout_sec)
        return True

    def stop(self) -> None:
        self.stopped = True


def test_runtime_handle_flushes_only_a_live_runtime() -> None:
    runtime = _FlushableRuntime()
    handle = lifecycle.RuntimeHandle(runtime)  # type: ignore[arg-type]

    assert handle.is_live
    assert handle.flush_pending_telemetry(1.5) is True
    assert runtime.flushes == [1.5]

    # The CLI executor stops the runtime itself, not through the handle.
    runtime.stop()
    assert not handle.is_live
    assert handle.flush_pending_telemetry(1.5) is False
    assert runtime.flushes == [1.5]

    noop = lifecycle.RuntimeHandle(lifecycle.NoOpRuntime())
    assert not noop.is_live
    assert noop.flush_pending_telemetry(1.5) is False


# probe_aggregator: the init() preflight with the TraceML hello.

_HELLO = build_hello_payload(
    session_id="run-b", run_nonce=None, session_source="explicit"
)


def _answering_server(admitted: bool) -> TCPServer:
    def reply(message):
        if not is_hello(message):
            return None
        return build_hello_ack_payload(
            session_id="run-b",
            traceml_version="test",
            mismatch=None if admitted else "session_id",
        )

    server = TCPServer(
        TCPConfig(host="127.0.0.1", port=0), first_frame_reply=reply
    )
    server.start()
    return server


@pytest.mark.parametrize(
    ("admitted", "status"),
    [
        (True, lifecycle.PROBE_ADMITTED),
        (False, lifecycle.PROBE_REFUSED),
    ],
)
def test_probe_reports_the_aggregator_ack(admitted, status) -> None:
    server = _answering_server(admitted)
    try:
        probe = lifecycle.probe_aggregator(
            "127.0.0.1", server.port, _HELLO, timeout_sec=2.0
        )
    finally:
        server.stop()

    assert probe.status == status
    assert probe.ack is not None
    assert probe.ack.session_id == "run-b"


def _one_shot_listener(handle) -> socket.socket:
    import threading

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def run():
        conn, _ = server.accept()
        with conn:
            handle(conn)

    threading.Thread(target=run, daemon=True).start()
    return server


def _hang_up(conn: socket.socket) -> None:
    # What an older aggregator does: read to EOF, then close silently.
    conn.settimeout(2.0)
    while conn.recv(1024):
        pass


def _http_error(conn: socket.socket) -> None:
    _hang_up(conn)
    conn.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")


@pytest.mark.parametrize(
    ("handle", "status"),
    [
        (lambda conn: time.sleep(1.0), lifecycle.PROBE_NO_REPLY),
        (_hang_up, lifecycle.PROBE_NO_REPLY),
        (_http_error, lifecycle.PROBE_NOT_TRACEML),
    ],
    ids=["silent", "hangs-up", "http"],
)
def test_probe_reports_a_listener_without_an_ack(handle, status) -> None:
    server = _one_shot_listener(handle)
    try:
        probe = lifecycle.probe_aggregator(
            "127.0.0.1",
            server.getsockname()[1],
            _HELLO,
            timeout_sec=2.0,
            reply_timeout_sec=0.3,
        )
    finally:
        server.close()

    assert probe == lifecycle.AggregatorProbe(status)


def test_probe_reports_an_unreachable_port() -> None:
    probe_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe_sock.bind(("127.0.0.1", 0))
    port = probe_sock.getsockname()[1]
    probe_sock.close()

    probe = lifecycle.probe_aggregator(
        "127.0.0.1", port, _HELLO, timeout_sec=0.3, poll_interval_sec=0.05
    )

    assert probe == lifecycle.AggregatorProbe(lifecycle.PROBE_UNREACHABLE)
