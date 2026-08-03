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
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

CONTROL_KIND_FIELD = "_traceml_control"
RANK_FINISHED = "rank_finished"


@dataclass(frozen=True)
class RankFinishedControl:
    """Marker sent once by a runtime rank after its final telemetry publish."""

    global_rank: int
    world_size: int
    node_rank: int
    hostname: str
    timestamp: float


def build_rank_finished_payload(
    *,
    global_rank: int,
    world_size: int,
    node_rank: int,
    hostname: str,
    timestamp: Optional[float] = None,
) -> dict[str, Any]:
    """Return the wire payload for one rank-finished control message."""
    return {
        CONTROL_KIND_FIELD: RANK_FINISHED,
        "global_rank": int(global_rank),
        "world_size": int(world_size),
        "node_rank": int(node_rank),
        "hostname": str(hostname),
        "timestamp": float(time.time() if timestamp is None else timestamp),
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
            timestamp=float(payload.get("timestamp", 0.0)),
        )
    except Exception:
        return None


__all__ = [
    "CONTROL_KIND_FIELD",
    "RANK_FINISHED",
    "RankFinishedControl",
    "build_rank_finished_payload",
    "parse_rank_finished",
]
