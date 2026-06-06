from __future__ import annotations

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
