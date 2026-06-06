# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Canonical telemetry envelope shared by runtime and aggregator code."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional


def _optional_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _optional_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return str(value)
    except Exception:
        return None


@dataclass(frozen=True)
class TelemetryMeta:
    """Stable metadata carried once per sampler payload."""

    rank: Optional[int]
    global_rank: Optional[int]
    local_rank: Optional[int]
    world_size: Optional[int]
    local_world_size: Optional[int]
    node_rank: Optional[int]
    hostname: Optional[str]
    pid: Optional[int]
    sampler: Optional[str]
    timestamp: Optional[float]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TelemetryMeta":
        """Create metadata from a transport mapping using best-effort coercion."""
        global_rank = _optional_int(data.get("global_rank"))
        rank = global_rank
        if rank is None:
            rank = _optional_int(data.get("rank"))

        return cls(
            rank=rank,
            global_rank=global_rank,
            local_rank=_optional_int(data.get("local_rank")),
            world_size=_optional_int(data.get("world_size")),
            local_world_size=_optional_int(data.get("local_world_size")),
            node_rank=_optional_int(data.get("node_rank")),
            hostname=_optional_str(data.get("hostname")),
            pid=_optional_int(data.get("pid")),
            sampler=_optional_str(data.get("sampler")),
            timestamp=_optional_float(data.get("timestamp")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a MessagePack/JSON-friendly metadata dictionary."""
        return {
            "rank": self.rank,
            "global_rank": self.global_rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "node_rank": self.node_rank,
            "hostname": self.hostname,
            "pid": self.pid,
            "sampler": self.sampler,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class TelemetryEnvelope:
    """
    Canonical sampler payload.

    `meta` contains fields shared by every row in the payload. `body.tables`
    contains the sampler-owned append-only rows grouped by table name.
    """

    meta: TelemetryMeta
    body: Mapping[str, Any]

    @property
    def tables(self) -> Mapping[str, Any]:
        """Return the body table mapping, or an empty mapping if malformed."""
        tables = (
            self.body.get("tables") if isinstance(self.body, Mapping) else None
        )
        return tables if isinstance(tables, Mapping) else {}

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical wire dictionary."""
        return {
            "meta": self.meta.to_dict(),
            "body": {"tables": dict(self.tables)},
        }


def build_telemetry_envelope(
    *,
    identity: Any,
    sampler_name: str,
    tables: Mapping[str, Any],
    timestamp: Optional[float] = None,
) -> dict[str, Any]:
    """Build the canonical wire payload emitted by runtime sampler senders."""
    fields = dict(identity.to_payload_fields())
    fields["sampler"] = str(sampler_name)
    fields["timestamp"] = (
        time.time() if timestamp is None else float(timestamp)
    )
    return TelemetryEnvelope(
        meta=TelemetryMeta.from_mapping(fields),
        body={"tables": dict(tables)},
    ).to_dict()


def normalize_telemetry_envelope(raw: Any) -> Optional[TelemetryEnvelope]:
    """
    Normalize one decoded telemetry payload.

    The canonical shape is `{"meta": {...}, "body": {"tables": {...}}}`.
    A temporary flat-envelope fallback remains for direct tests or external
    clients that still send the old shape during the transport migration.
    TODO: remove flat-envelope normalization after all direct callers emit
    canonical envelopes.
    """
    if not isinstance(raw, Mapping):
        return None

    meta_raw = raw.get("meta")
    body_raw = raw.get("body")
    if isinstance(meta_raw, Mapping) and isinstance(body_raw, Mapping):
        return TelemetryEnvelope(
            meta=TelemetryMeta.from_mapping(meta_raw),
            body=body_raw,
        )

    tables = raw.get("tables")
    if isinstance(tables, Mapping):
        return TelemetryEnvelope(
            meta=TelemetryMeta.from_mapping(raw),
            body={"tables": tables},
        )

    return None
