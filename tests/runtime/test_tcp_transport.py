from traceml_ai.transport.tcp_transport import TCPConfig, TCPServer


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
