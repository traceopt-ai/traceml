# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Shared topology helpers for final-report sections."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Optional


def _identity_value(identity: Any, field: str) -> Optional[Any]:
    """Read an identity field from either a mapping or a dataclass-like object."""
    if identity is None:
        return None
    if isinstance(identity, Mapping):
        return identity.get(field)
    return getattr(identity, field, None)


def topology_mode_from_identities(
    identities: Iterable[Any],
    *,
    has_data: bool,
) -> str:
    """
    Infer run topology from runtime identity records.

    Sections represent identity rows with different concrete types, but the
    topology rule is shared: multiple observed node ranks, or a world size
    larger than local world size, means multi-node.
    """
    if not has_data:
        return "no_data"

    identity_rows = list(identities)
    node_ranks = {
        _identity_value(identity, "node_rank")
        for identity in identity_rows
        if _identity_value(identity, "node_rank") is not None
    }
    if len(node_ranks) > 1:
        return "multi_node"

    for identity in identity_rows:
        world_size = _identity_value(identity, "world_size")
        local_world_size = _identity_value(identity, "local_world_size")
        if (
            world_size is not None
            and local_world_size is not None
            and int(world_size) > int(local_world_size)
        ):
            return "multi_node"

    return "single_node"


__all__ = [
    "topology_mode_from_identities",
]
