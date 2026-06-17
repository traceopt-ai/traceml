from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Any

from traceml_ai.runtime import lifecycle
from traceml_ai.runtime.settings import AggregatorEndpoint, TraceMLSettings


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


class _FakeRuntime:
    """Minimal TraceMLRuntime stand-in for start_runtime() tests."""

    def __init__(self, settings: TraceMLSettings | None = None) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


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
