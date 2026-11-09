import pytest

from traceml_ai.runtime.environment import RuntimeEnvironmentInfo
from traceml_ai.runtime import environment_state


@pytest.fixture(autouse=True)
def _reset_runtime_environment_state():
    environment_state.reset_runtime_environment_state()
    yield
    environment_state.reset_runtime_environment_state()


def _info(strategy: str = "single_process") -> RuntimeEnvironmentInfo:
    return RuntimeEnvironmentInfo(
        topology="single_process",
        distributed_initialized=False,
        distributed_backend=None,
        training_strategy=strategy,
        strategy_source="topology",
        strategy_confidence="high",
    )


def test_publish_once_drain_once_and_reset(monkeypatch) -> None:
    timestamp_unix_ns = 123_500_000_001
    monkeypatch.setattr(
        environment_state.time, "time_ns", lambda: timestamp_unix_ns
    )

    assert environment_state.has_runtime_environment_info() is False
    assert environment_state.publish_runtime_environment_once(_info()) is True
    assert environment_state.has_runtime_environment_info() is True

    monkeypatch.setattr(
        environment_state.time,
        "time_ns",
        lambda: timestamp_unix_ns + 10_000_000_000,
    )
    row = environment_state.pop_runtime_environment_record()
    assert row == {
        "seq": 0,
        "timestamp_unix_ns": timestamp_unix_ns,
        "topology": "single_process",
        "distributed_initialized": False,
        "distributed_backend": None,
        "training_strategy": "single_process",
        "strategy_source": "topology",
        "strategy_confidence": "high",
    }
    assert environment_state.pop_runtime_environment_record() is None

    environment_state.reset_runtime_environment_state()

    assert environment_state.has_runtime_environment_info() is False
    assert environment_state.pop_runtime_environment_record() is None


def test_second_publish_is_ignored() -> None:
    assert environment_state.publish_runtime_environment_once(_info("ddp"))
    assert not environment_state.publish_runtime_environment_once(
        _info("fsdp")
    )

    row = environment_state.pop_runtime_environment_record()

    assert row is not None
    assert row["training_strategy"] == "ddp"
