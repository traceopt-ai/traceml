import errno
import socket

import pytest

from traceml_ai.transport.tcp_transport import (
    TCPClient,
    TCPConfig,
    TCPServer,
    bind_exclusive_listener,
)


class _FakeSocket:
    def __init__(self):
        self.bound_to = None
        self.closed = False
        self.options = []

    def setsockopt(self, *args):
        self.options.append(args)

    def bind(self, address):
        self.bound_to = address

    def getsockname(self):
        host, _port = self.bound_to
        return (host, 54321)

    def listen(self, _backlog):
        return None

    def accept(self):
        raise OSError("closed")

    def close(self):
        self.closed = True


class _RecordingLogger:
    def __init__(self):
        self.errors = []

    def error(self, message, *args):
        self.errors.append(message % args)


class _FailingSendSocket:
    def __init__(self):
        self.closed = False

    def sendall(self, _data):
        raise ConnectionResetError("connection reset by peer")

    def close(self):
        self.closed = True


@pytest.mark.parametrize("exclusive", [None, 42])
def test_listener_uses_platform_appropriate_socket_option(
    monkeypatch, exclusive
) -> None:
    fake = _FakeSocket()
    monkeypatch.setattr(
        "traceml_ai.transport.tcp_transport.socket.socket",
        lambda *_args, **_kwargs: fake,
    )
    if exclusive is None:
        monkeypatch.delattr(socket, "SO_EXCLUSIVEADDRUSE", raising=False)
        expected_option = socket.SO_REUSEADDR
    else:
        monkeypatch.setattr(
            socket, "SO_EXCLUSIVEADDRUSE", exclusive, raising=False
        )
        expected_option = exclusive

    result = bind_exclusive_listener("127.0.0.1", 43170, backlog=1)

    assert result is fake
    assert fake.options == [(socket.SOL_SOCKET, expected_option, 1)]


def test_tcp_server_exposes_actual_port_for_dynamic_bind(monkeypatch) -> None:
    fake = _FakeSocket()
    monkeypatch.setattr(
        "traceml_ai.transport.tcp_transport.socket.socket",
        lambda *_args, **_kwargs: fake,
    )

    server = TCPServer(TCPConfig(host="127.0.0.1", port=0))
    server.start()
    try:
        assert fake.bound_to == ("127.0.0.1", 0)
        assert server.port == 54321
    finally:
        server.stop()
    assert fake.closed


def _addr_in_use(exc: OSError) -> bool:
    return exc.errno in (
        errno.EADDRINUSE,
        getattr(errno, "WSAEADDRINUSE", errno.EADDRINUSE),
    )


def test_second_tcp_server_cannot_share_a_live_port() -> None:
    # An aggregator orphaned by a killed `traceml run` keeps listening on
    # the default port. If the next run's aggregator could bind beside it,
    # the kernel would hand some rank connections to the orphan and the new
    # session would silently record nothing.
    first = TCPServer(TCPConfig(host="127.0.0.1", port=0))
    first.start()
    try:
        second = TCPServer(TCPConfig(host="127.0.0.1", port=first.port))
        with pytest.raises(OSError) as excinfo:
            second.start()
        assert _addr_in_use(excinfo.value)
        assert second._sock is None
    finally:
        first.stop()


def test_tcp_server_stop_releases_its_port() -> None:
    first = TCPServer(TCPConfig(host="127.0.0.1", port=0))
    first.start()
    port = first.port
    thread = first._thread
    first.stop()
    assert thread is not None
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert first._sock is None

    second = TCPServer(TCPConfig(host="127.0.0.1", port=port))
    second.start()
    second.stop()


def test_tcp_server_logs_listener_close_failure() -> None:
    class _FailingCloseSocket(_FakeSocket):
        def shutdown(self, _how):
            return None

        def close(self):
            raise OSError("close failed")

    server = TCPServer(TCPConfig())
    server._sock = _FailingCloseSocket()
    logger = _RecordingLogger()
    server.logger = logger

    server.stop()

    assert server._sock is None
    assert logger.errors == [
        "[TraceML] TCP listener close failed: OSError: close failed"
    ]


def test_tcp_server_rebinds_port_left_in_time_wait() -> None:
    # A restart must still bind at once while the previous session's
    # server-side connections sit in TIME_WAIT. Linux skips a TIME_WAIT
    # conflict only when both the old and new sockets set SO_REUSEADDR, so
    # this listener stands in for the previous aggregator.
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    client = socket.create_connection(("127.0.0.1", port))
    conn, _ = listener.accept()
    conn.close()  # server closes first, so its side enters TIME_WAIT
    client.close()
    listener.close()

    server = TCPServer(TCPConfig(host="127.0.0.1", port=port))
    server.start()
    try:
        assert server.port == port
    finally:
        server.stop()


def test_tcp_client_logs_send_failure_and_remains_best_effort() -> None:
    client = TCPClient(TCPConfig(host="10.0.0.8", port=29765))
    sock = _FailingSendSocket()
    logger = _RecordingLogger()
    client._sock = sock
    client._connected = True
    client.logger = logger

    client.send_batch([{"sample": 1}])

    assert sock.closed
    assert client._connected is False
    assert client._sock is None
    assert logger.errors == [
        "[TraceML] TCP telemetry send_batch failed for 10.0.0.8:29765: "
        "ConnectionResetError: connection reset by peer"
    ]
