# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

import os

from traceml.runtime.identity import RuntimeIdentity, resolve_runtime_identity


class _FakeDistributed:
    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def get_rank(self):
        return 3

    def get_world_size(self):
        return 4


class _FakeCuda:
    def device_count(self):
        raise AssertionError("identity resolution must not touch CUDA")


class _FakeTorch:
    distributed = _FakeDistributed()
    cuda = _FakeCuda()


def test_runtime_identity_single_process_defaults():
    identity = resolve_runtime_identity(env={})

    assert identity.global_rank == 0
    assert identity.local_rank == 0
    assert identity.rank == 0
    assert identity.world_size == 1
    assert identity.local_world_size == 1
    assert identity.node_rank == 0
    assert identity.hostname
    assert identity.pid == os.getpid()
    assert not identity.is_distributed
    assert not identity.is_multinode


def test_runtime_identity_resolves_single_node_torchrun_env():
    identity = resolve_runtime_identity(
        env={
            "RANK": "2",
            "LOCAL_RANK": "2",
            "WORLD_SIZE": "4",
            "LOCAL_WORLD_SIZE": "4",
            "GROUP_RANK": "0",
        }
    )

    assert identity.global_rank == 2
    assert identity.local_rank == 2
    assert identity.rank == 2
    assert identity.world_size == 4
    assert identity.local_world_size == 4
    assert identity.node_rank == 0
    assert identity.is_distributed
    assert not identity.is_multinode


def test_runtime_identity_resolves_multinode_torchrun_env():
    identity = resolve_runtime_identity(
        env={
            "RANK": "5",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "8",
            "LOCAL_WORLD_SIZE": "4",
            "GROUP_RANK": "1",
        }
    )

    assert identity.global_rank == 5
    assert identity.local_rank == 1
    assert identity.rank == 1
    assert identity.world_size == 8
    assert identity.local_world_size == 4
    assert identity.node_rank == 1
    assert identity.is_distributed
    assert identity.is_multinode


def test_runtime_identity_infers_missing_local_rank():
    identity = resolve_runtime_identity(
        env={
            "RANK": "5",
            "WORLD_SIZE": "8",
            "LOCAL_WORLD_SIZE": "4",
        }
    )

    assert identity.global_rank == 5
    assert identity.local_rank == 1
    assert identity.node_rank == 1


def test_runtime_identity_can_fall_back_to_torch_distributed_state():
    identity = resolve_runtime_identity(
        env={"LOCAL_WORLD_SIZE": "2"},
        torch_loader=lambda: _FakeTorch(),
    )

    assert identity.global_rank == 3
    assert identity.local_rank == 1
    assert identity.world_size == 4
    assert identity.local_world_size == 2
    assert identity.node_rank == 1
    assert identity.is_distributed
    assert identity.is_multinode


def test_runtime_identity_does_not_probe_cuda_for_local_world_size():
    identity = resolve_runtime_identity(
        env={"RANK": "3", "WORLD_SIZE": "4"},
        torch_loader=lambda: _FakeTorch(),
    )

    assert identity.global_rank == 3
    assert identity.local_world_size == 1
    assert identity.local_rank == 0
    assert identity.node_rank == 3


def test_runtime_identity_meta_shape():
    identity = RuntimeIdentity(
        global_rank=5,
        local_rank=1,
        world_size=8,
        local_world_size=4,
        node_rank=1,
        hostname="worker-1",
        pid=123,
    )

    assert identity.to_meta() == {
        "rank": 1,
        "global_rank": 5,
        "local_rank": 1,
        "world_size": 8,
        "local_world_size": 4,
        "node_rank": 1,
        "hostname": "worker-1",
        "pid": 123,
    }
