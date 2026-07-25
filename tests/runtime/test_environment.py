# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from traceml_ai.runtime.environment import detect_runtime_environment


class _FakeDistributed:
    def __init__(
        self,
        *,
        available=True,
        initialized=True,
        backend="nccl",
        rank=0,
        world_size=1,
    ):
        self._available = available
        self._initialized = initialized
        self._backend = backend
        self._rank = rank
        self._world_size = world_size

    def is_available(self):
        return self._available

    def is_initialized(self):
        return self._initialized

    def get_backend(self):
        return self._backend

    def get_rank(self):
        return self._rank

    def get_world_size(self):
        return self._world_size


class _BrokenDistributed:
    def is_available(self):
        raise RuntimeError("distributed unavailable")

    def is_initialized(self):
        raise RuntimeError("distributed unavailable")

    def get_rank(self):
        raise RuntimeError("rank unavailable")

    def get_world_size(self):
        raise RuntimeError("world unavailable")


class _FakeDDP:
    pass


class _FakeFSDP:
    pass


class _PlainModel:
    pass


def _fake_torch(*, distributed=None, ddp_cls=None, fsdp_cls=None):
    dist = distributed or _FakeDistributed()
    dist.fsdp = SimpleNamespace(
        FullyShardedDataParallel=fsdp_cls,
        fully_sharded_data_parallel=SimpleNamespace(
            FullyShardedDataParallel=fsdp_cls
        ),
    )
    return SimpleNamespace(
        distributed=dist,
        nn=SimpleNamespace(
            parallel=SimpleNamespace(DistributedDataParallel=ddp_cls)
        ),
    )


def test_runtime_environment_single_process_without_torch():
    info = detect_runtime_environment(env={}, torch_loader=lambda: None)

    assert info.topology == "single_process"
    assert not info.distributed_initialized
    assert info.distributed_backend is None
    assert info.training_strategy == "single_process"
    assert info.strategy_source == "topology"
    assert info.strategy_confidence == "high"
    assert info.to_record() == {
        "topology": "single_process",
        "distributed_initialized": False,
        "distributed_backend": None,
        "training_strategy": "single_process",
        "strategy_source": "topology",
        "strategy_confidence": "high",
    }


def test_runtime_environment_single_node_multi_process_from_env():
    info = detect_runtime_environment(
        env={
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "4",
            "LOCAL_WORLD_SIZE": "4",
            "GROUP_RANK": "0",
        },
        torch_loader=lambda: None,
    )

    assert info.topology == "single_node_multi_process"
    assert info.training_strategy == "distributed_unknown"
    assert info.strategy_source == "topology"
    assert info.strategy_confidence == "low"


def test_runtime_environment_multi_node_from_env():
    info = detect_runtime_environment(
        env={
            "RANK": "5",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "8",
            "LOCAL_WORLD_SIZE": "4",
            "GROUP_RANK": "1",
        },
        torch_loader=lambda: None,
    )

    assert info.topology == "multi_node"
    assert info.training_strategy == "distributed_unknown"


def test_runtime_environment_detects_ddp_model():
    torch = _fake_torch(
        distributed=_FakeDistributed(world_size=4),
        ddp_cls=_FakeDDP,
    )

    info = detect_runtime_environment(
        model=_FakeDDP(),
        env={"RANK": "0", "WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "4"},
        torch_loader=lambda: torch,
    )

    assert info.training_strategy == "ddp"
    assert info.strategy_source == "runtime_model"
    assert info.strategy_confidence == "high"


def test_runtime_environment_detects_fsdp_model():
    torch = _fake_torch(
        distributed=_FakeDistributed(world_size=4),
        fsdp_cls=_FakeFSDP,
    )

    info = detect_runtime_environment(
        model=_FakeFSDP(),
        env={"RANK": "0", "WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "4"},
        torch_loader=lambda: torch,
    )

    assert info.training_strategy == "fsdp"
    assert info.strategy_source == "runtime_model"
    assert info.strategy_confidence == "high"


def test_runtime_environment_captures_distributed_backend():
    torch = _fake_torch(
        distributed=_FakeDistributed(
            initialized=True,
            backend="gloo",
            rank=1,
            world_size=2,
        )
    )

    info = detect_runtime_environment(env={}, torch_loader=lambda: torch)

    assert info.topology == "multi_node"
    assert info.distributed_initialized
    assert info.distributed_backend == "gloo"
    assert info.training_strategy == "distributed_unknown"
    assert info.strategy_source == "runtime_distributed"


def test_runtime_environment_degrades_when_distributed_raises():
    torch = _fake_torch(distributed=_BrokenDistributed())

    info = detect_runtime_environment(
        model=_PlainModel(),
        env={"RANK": "0", "WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2"},
        torch_loader=lambda: torch,
    )

    assert info.topology == "single_node_multi_process"
    assert not info.distributed_initialized
    assert info.distributed_backend is None
    assert info.training_strategy == "distributed_unknown"
    assert info.strategy_confidence == "low"
