# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Internal telemetry control messages.

Control messages travel over the same TCP transport as sampler telemetry, but
they are not sampler payloads and are never written to SQLite projection tables.
They let the aggregator distinguish "workers are still sending late telemetry"
from "all expected ranks have finished, so end-of-run finalization can begin".

The hello and its ack are the ``traceml.init()`` preflight: the runtime sends
one hello as the first frame of a probe connection, and an aggregator answers
with one ack saying whether it would admit this run's telemetry. An older
aggregator never answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

CONTROL_KIND_FIELD = "_traceml_control"
RANK_FINISHED = "rank_finished"
HELLO = "hello"
HELLO_ACK = "hello_ack"
HELLO_PROTOCOL = 1


@dataclass(frozen=True)
class RankFinishedControl:
    """Marker sent once by a runtime rank after its final telemetry publish."""

    global_rank: int
    world_size: int
    node_rank: int
    hostname: str


def build_rank_finished_payload(
    *,
    global_rank: int,
    world_size: int,
    node_rank: int,
    hostname: str,
    session_id: Optional[str] = None,
    run_nonce: Optional[str] = None,
    session_source: Optional[str] = None,
) -> dict[str, Any]:
    """Return the wire payload for one rank-finished control message.

    ``session_id``, ``run_nonce`` and ``session_source`` stamp the sender's
    run, the same optional keys telemetry envelopes carry.
    """
    return {
        CONTROL_KIND_FIELD: RANK_FINISHED,
        "global_rank": int(global_rank),
        "world_size": int(world_size),
        "node_rank": int(node_rank),
        "hostname": str(hostname),
        "session_id": session_id or None,
        "run_nonce": run_nonce or None,
        "session_source": session_source or None,
    }


def parse_rank_finished(
    payload: Any,
) -> Optional[RankFinishedControl]:
    """Parse a rank-finished control payload, returning ``None`` otherwise."""
    if not isinstance(payload, Mapping):
        return None
    if payload.get(CONTROL_KIND_FIELD) != RANK_FINISHED:
        return None
    try:
        return RankFinishedControl(
            global_rank=int(payload["global_rank"]),
            world_size=int(payload.get("world_size", 1)),
            node_rank=int(payload.get("node_rank", 0)),
            hostname=str(payload.get("hostname", "")),
        )
    except Exception:
        return None


@dataclass(frozen=True)
class HelloAck:
    """An aggregator's answer to a runtime's hello."""

    admitted: bool
    session_id: str
    traceml_version: str
    # The stamp key that names another run when ``admitted`` is False.
    reason: Optional[str]


def build_hello_payload(
    *,
    session_id: Optional[str],
    run_nonce: Optional[str],
    session_source: Optional[str],
) -> dict[str, Any]:
    """Return the hello a runtime sends before it starts sending telemetry.

    It carries the same stamp keys as the runtime's telemetry, so the
    aggregator can judge it with the rule it applies to that telemetry.
    """
    return {
        CONTROL_KIND_FIELD: HELLO,
        "protocol": HELLO_PROTOCOL,
        "session_id": session_id or None,
        "run_nonce": run_nonce or None,
        "session_source": session_source or None,
    }


def is_hello(payload: Any) -> bool:
    """Return True when ``payload`` is a hello control message."""
    return (
        isinstance(payload, Mapping)
        and payload.get(CONTROL_KIND_FIELD) == HELLO
    )


def build_hello_ack_payload(
    *,
    session_id: str,
    traceml_version: str,
    mismatch: Optional[str],
) -> dict[str, Any]:
    """Return the ack for one hello.

    ``mismatch`` is the stamp key that names another run, or None when the
    aggregator would admit the sender's telemetry.
    """
    return {
        CONTROL_KIND_FIELD: HELLO_ACK,
        "protocol": HELLO_PROTOCOL,
        "traceml_version": str(traceml_version),
        "session_id": str(session_id),
        "admitted": mismatch is None,
        "reason": mismatch,
    }


def parse_hello_ack(payload: Any) -> Optional[HelloAck]:
    """Parse a hello ack, returning ``None`` for anything else."""
    if not isinstance(payload, Mapping):
        return None
    if payload.get(CONTROL_KIND_FIELD) != HELLO_ACK:
        return None
    admitted = payload.get("admitted")
    if not isinstance(admitted, bool):
        return None
    reason = payload.get("reason")
    return HelloAck(
        admitted=admitted,
        session_id=str(payload.get("session_id") or ""),
        traceml_version=str(payload.get("traceml_version") or ""),
        reason=None if reason is None else str(reason),
    )


__all__ = [
    "CONTROL_KIND_FIELD",
    "HELLO",
    "HELLO_ACK",
    "HELLO_PROTOCOL",
    "HelloAck",
    "RANK_FINISHED",
    "RankFinishedControl",
    "build_hello_ack_payload",
    "build_hello_payload",
    "build_rank_finished_payload",
    "is_hello",
    "parse_hello_ack",
    "parse_rank_finished",
]
