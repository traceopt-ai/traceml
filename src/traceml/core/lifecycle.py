"""
Small lifecycle protocols shared by runtime and display components.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Startable(Protocol):
    """Object with an explicit start lifecycle hook."""

    def start(self) -> None:
        """Start the component."""


@runtime_checkable
class Stoppable(Protocol):
    """Object with an explicit stop lifecycle hook."""

    def stop(self) -> None:
        """Stop the component and release owned resources."""


@runtime_checkable
class Tickable(Protocol):
    """Object that can perform one unit of periodic work."""

    def tick(self) -> None:
        """Run one best-effort lifecycle tick."""


__all__ = [
    "Startable",
    "Stoppable",
    "Tickable",
]
