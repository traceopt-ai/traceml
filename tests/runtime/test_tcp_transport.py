import socket

from traceml_ai.transport.tcp_transport import TCPClient, TCPConfig, TCPServer


class _FakeSocket:
    def __init__(self):
        self.bound_to = None
        self.closed = False

    def setsockopt(self, *_args):
        return None

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


def test_tcp_server_starts_without_so_reuseport(monkeypatch) -> None:
    # Uses a real socket: _FakeSocket swallows setsockopt, so only a
    # genuine bind proves the Windows path (no SO_REUSEPORT) works.
    monkeypatch.delattr(
        "traceml_ai.transport.tcp_transport.socket.SO_REUSEPORT",
        raising=False,
    )

    server = TCPServer(TCPConfig(host="127.0.0.1", port=0))
    server.start()
    try:
        assert server.port > 0
    finally:
        server.stop()


def test_tcp_server_starts_when_so_reuseport_is_rejected(
    monkeypatch,
) -> None:
    # A kernel older than the headers Python was built against exposes the
    # constant but rejects the option, so presence alone is not enough.
    monkeypatch.setattr(
        "traceml_ai.transport.tcp_transport.socket.SO_REUSEPORT",
        15,
        raising=False,
    )

    class _RejectingSocket(_FakeSocket):
        def setsockopt(self, _level, option, _value):
            if option == socket.SO_REUSEPORT:
                raise OSError(92, "Protocol not available")
            return None

    fake = _RejectingSocket()
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
