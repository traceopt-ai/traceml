"""Shared training-strategy vocabulary and normalization."""

from __future__ import annotations

from typing import Any

DEFAULT_TRAINING_STRATEGY = "ddp"
KNOWN_TRAINING_STRATEGIES = frozenset(
    {"ddp", "fsdp", "distributed_unknown", "single_process", "unknown"}
)


def normalize_training_strategy(value: Any) -> str:
    """Return a known training strategy, defaulting to current DDP behavior."""
    strategy = str(value or DEFAULT_TRAINING_STRATEGY).strip().lower()
    if strategy in KNOWN_TRAINING_STRATEGIES:
        return strategy
    return DEFAULT_TRAINING_STRATEGY


__all__ = [
    "DEFAULT_TRAINING_STRATEGY",
    "KNOWN_TRAINING_STRATEGIES",
    "normalize_training_strategy",
]
