import pytest

from traceml_ai.runtime.environment import RuntimeEnvironmentInfo
from traceml_ai.runtime.environment_state import (
    publish_runtime_environment_once,
    reset_runtime_environment_state,
)
from traceml_ai.runtime.sender import SenderIdentity
from traceml_ai.samplers.runtime_environment_sampler import (
    RuntimeEnvironmentSampler,
)


@pytest.fixture(autouse=True)
def _reset_runtime_environment_state():
    reset_runtime_environment_state()
    yield
    reset_runtime_environment_state()


def _info(strategy: str = "ddp") -> RuntimeEnvironmentInfo:
    return RuntimeEnvironmentInfo(
        topology="single_node_multi_process",
        distributed_initialized=True,
        distributed_backend="nccl",
        training_strategy=strategy,
        strategy_source="runtime_model",
        strategy_confidence="high",
    )


def test_sampler_emits_no_row_before_publish() -> None:
    sampler = RuntimeEnvironmentSampler()

    sampler.sample()

    assert sampler.db.get_table("RuntimeEnvironmentTable") is None
    assert sampler.sender.collect_payload() is None


def test_sampler_drains_one_published_environment_row() -> None:
    publish_runtime_environment_once(_info())
    sampler = RuntimeEnvironmentSampler()

    sampler.sample()
    sampler.sample()

    rows = list(sampler.db.get_table("RuntimeEnvironmentTable") or [])
    assert len(rows) == 1
    assert rows[0]["seq"] == 0
    assert rows[0]["training_strategy"] == "ddp"
    assert rows[0]["distributed_backend"] == "nccl"


def test_sender_envelope_body_contains_runtime_environment_table() -> None:
    publish_runtime_environment_once(_info("fsdp"))
    sampler = RuntimeEnvironmentSampler()
    sampler.sender.identity = SenderIdentity(
        global_rank=3,
        local_rank=1,
        world_size=8,
        local_world_size=4,
        node_rank=0,
        hostname="worker-0",
        pid=1234,
    )

    sampler.sample()
    payload = sampler.sender.collect_payload()

    assert payload is not None
    assert payload["meta"]["sampler"] == "RuntimeEnvironmentSampler"
    assert payload["meta"]["global_rank"] == 3
    rows = payload["body"]["tables"]["RuntimeEnvironmentTable"]
    assert len(rows) == 1
    assert rows[0]["training_strategy"] == "fsdp"
    assert rows[0]["strategy_source"] == "runtime_model"
