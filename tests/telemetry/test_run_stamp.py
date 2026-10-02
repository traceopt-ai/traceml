# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Rank-side run stamp on telemetry and control payloads.

Every payload names the run it belongs to (``session_id``) and the launch
that started it (``run_nonce``) so the aggregator can drop telemetry from a
rank left over from another run. Both keys are optional and additive.
"""

from __future__ import annotations

from unittest.mock import Mock

from traceml_ai.runtime.runtime import TraceMLRuntime
from traceml_ai.runtime.sender import SenderIdentity
from traceml_ai.runtime.settings import TraceMLSettings
from traceml_ai.telemetry.control import build_rank_finished_payload
from traceml_ai.telemetry.envelope import (
    TelemetryMeta,
    build_telemetry_envelope,
)


def test_identity_emits_stamp_fields():
    identity = SenderIdentity(
        global_rank=1, local_rank=1, session_id="run-a", run_nonce="n1"
    )

    fields = identity.to_payload_fields()

    assert fields["session_id"] == "run-a"
    assert fields["run_nonce"] == "n1"


def test_identity_without_stamp_emits_none():
    fields = SenderIdentity(global_rank=0, local_rank=0).to_payload_fields()

    assert fields["session_id"] is None
    assert fields["run_nonce"] is None


def test_envelope_meta_carries_stamp():
    payload = build_telemetry_envelope(
        identity=SenderIdentity(
            global_rank=0, local_rank=0, session_id="run-a", run_nonce="n1"
        ),
        sampler_name="SystemSampler",
        tables={"SystemTable": [{"seq": 1}]},
        timestamp=1.0,
    )

    assert payload["meta"]["session_id"] == "run-a"
    assert payload["meta"]["run_nonce"] == "n1"
    meta = TelemetryMeta.from_mapping(payload["meta"])
    assert (meta.session_id, meta.run_nonce) == ("run-a", "n1")


def test_meta_from_old_rank_has_no_stamp():
    meta = TelemetryMeta.from_mapping({"global_rank": 0, "sampler": "X"})

    assert meta.session_id is None
    assert meta.run_nonce is None


def test_rank_finished_payload_carries_stamp():
    stamped = build_rank_finished_payload(
        global_rank=0,
        world_size=1,
        node_rank=0,
        hostname="host-a",
        session_id="run-a",
        run_nonce="n1",
    )
    unstamped = build_rank_finished_payload(
        global_rank=0, world_size=1, node_rank=0, hostname="host-a"
    )

    assert stamped["session_id"] == "run-a"
    assert stamped["run_nonce"] == "n1"
    assert unstamped["session_id"] is None
    assert unstamped["run_nonce"] is None


class _Logger:
    def error(self, *_args, **_kwargs) -> None:
        return None

    def exception(self, *_args, **_kwargs) -> None:
        return None


class _TCPClient:
    def __init__(self, _config) -> None:
        return None

    def close(self) -> None:
        return None


class _StoppedThread:
    def join(self, timeout=None) -> None:
        return None

    def is_alive(self) -> bool:
        return False


def test_runtime_stamps_telemetry_and_rank_finished(monkeypatch):
    monkeypatch.setattr(
        "traceml_ai.runtime.runtime.setup_error_logger", Mock()
    )
    monkeypatch.setattr(
        "traceml_ai.runtime.runtime.get_error_logger",
        lambda _name: _Logger(),
    )
    monkeypatch.setattr(
        "traceml_ai.runtime.runtime.build_samplers", lambda **_kw: []
    )
    monkeypatch.setattr("traceml_ai.runtime.runtime.TCPClient", _TCPClient)

    runtime = TraceMLRuntime(
        settings=TraceMLSettings(
            mode="summary", session_id="run-a", run_nonce="n1"
        )
    )

    identity = runtime._publisher._identity
    assert (identity.session_id, identity.run_nonce) == ("run-a", "n1")

    controls = []
    runtime._sampler_thread = _StoppedThread()
    runtime._publisher = Mock(send_control=controls.append)
    runtime._exporter = Mock()
    runtime.stop()

    assert controls[0]["session_id"] == "run-a"
    assert controls[0]["run_nonce"] == "n1"


def test_session_source_rides_with_the_stamp():
    identity = SenderIdentity(
        global_rank=0,
        local_rank=0,
        session_id="s",
        session_source="generated",
    )
    payload = build_telemetry_envelope(
        identity=identity,
        sampler_name="SystemSampler",
        tables={"SystemTable": [{"seq": 1}]},
        timestamp=1.0,
    )
    control = build_rank_finished_payload(
        global_rank=0,
        world_size=1,
        node_rank=0,
        hostname="h",
        session_id="s",
        session_source="generated",
    )

    assert payload["meta"]["session_source"] == "generated"
    assert TelemetryMeta.from_mapping(payload["meta"]).session_source == (
        "generated"
    )
    assert control["session_source"] == "generated"
    assert (
        SenderIdentity(global_rank=0, local_rank=0).to_payload_fields()[
            "session_source"
        ]
        is None
    )


def test_runtime_stamps_session_source(monkeypatch):
    monkeypatch.setattr(
        "traceml_ai.runtime.runtime.setup_error_logger", Mock()
    )
    monkeypatch.setattr(
        "traceml_ai.runtime.runtime.get_error_logger",
        lambda _name: _Logger(),
    )
    monkeypatch.setattr(
        "traceml_ai.runtime.runtime.build_samplers", lambda **_kw: []
    )
    monkeypatch.setattr("traceml_ai.runtime.runtime.TCPClient", _TCPClient)

    runtime = TraceMLRuntime(
        settings=TraceMLSettings(
            mode="summary", session_id="s", session_source="generated"
        )
    )
    assert runtime._publisher._identity.session_source == "generated"

    controls = []
    runtime._sampler_thread = _StoppedThread()
    runtime._publisher = Mock(send_control=controls.append)
    runtime._exporter = Mock()
    runtime.stop()
    assert controls[0]["session_source"] == "generated"
