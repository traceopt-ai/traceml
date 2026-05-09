# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

from traceml.runtime.runtime import TraceMLRuntime
from traceml.runtime.sender import SenderIdentity
from traceml.runtime.settings import TraceMLSettings


class _FakeLogger:
    def error(self, *_args, **_kwargs) -> None:
        return None

    def exception(self, *_args, **_kwargs) -> None:
        return None


class _FakeTCPClient:
    def __init__(self, _config) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakePublisher:
    instances: list["_FakePublisher"] = []

    def __init__(
        self,
        *,
        tcp_client,
        identity,
        logger,
    ) -> None:
        self.tcp_client = tcp_client
        self.identity = identity
        self.logger = logger
        self.attached_samplers = None
        _FakePublisher.instances.append(self)

    def attach_senders(self, samplers) -> None:
        self.attached_samplers = list(samplers)

    def publish(self, samplers) -> None:
        return None

    def close(self) -> None:
        return None


def test_runtime_uses_local_rank_for_samplers_and_global_rank_for_publisher(
    monkeypatch,
):
    build_calls: list[dict] = []

    def _fake_build_samplers(**kwargs):
        build_calls.append(kwargs)
        return []

    _FakePublisher.instances.clear()
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("GROUP_RANK", "1")
    monkeypatch.setattr(
        "traceml.runtime.runtime.setup_error_logger",
        lambda: None,
    )
    monkeypatch.setattr(
        "traceml.runtime.runtime.get_error_logger",
        lambda _name: _FakeLogger(),
    )
    monkeypatch.setattr(
        "traceml.runtime.runtime.build_samplers",
        _fake_build_samplers,
    )
    monkeypatch.setattr("traceml.runtime.runtime.TCPClient", _FakeTCPClient)
    monkeypatch.setattr(
        "traceml.runtime.runtime.TelemetryPublisher",
        _FakePublisher,
    )

    runtime = TraceMLRuntime(settings=TraceMLSettings(mode="summary"))

    assert runtime.global_rank == 5
    assert runtime.local_rank == 1
    assert runtime.world_size == 8
    assert build_calls[0]["local_rank"] == 1
    assert _FakePublisher.instances[0].identity == SenderIdentity(
        global_rank=5,
        local_rank=1,
        world_size=8,
        local_world_size=4,
        node_rank=1,
        hostname=runtime.identity.hostname,
        pid=runtime.identity.pid,
    )
