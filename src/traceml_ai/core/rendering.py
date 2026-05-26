"""
Rendering and formatting contracts.

These protocols intentionally avoid importing concrete output libraries. Rich,
HTML, JSON-safe dictionaries, and plain text can all be represented by the
generic output type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Optional, Protocol, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True)
class RenderContext:
    """
    Optional metadata passed to renderers and formatters.

    The context is deliberately generic so display backends can pass backend
    names, window sizes, rank filters, or feature flags without creating a
    shared dependency on a concrete UI package.
    """

    backend: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)


class Formatter(Protocol, Generic[InputT, OutputT]):
    """Convert a domain payload into a backend-specific representation."""

    name: str

    def format(self, payload: InputT) -> OutputT:
        """Format one payload."""


class Renderer(Protocol, Generic[OutputT]):
    """Render current state for a display backend."""

    name: str

    def get_panel_renderable(self) -> OutputT:
        """Return the current live display representation."""


__all__ = [
    "Formatter",
    "InputT",
    "OutputT",
    "RenderContext",
    "Renderer",
]
