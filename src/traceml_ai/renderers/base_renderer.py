from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Protocol, runtime_checkable


class BaseRenderer(ABC):
    """
    Shared metadata/state base for live renderers.

    Display-specific contracts are intentionally separate:
    - CLIRenderer: implements get_panel_renderable()
    - DashboardRenderer: implements get_dashboard_renderable()
    """

    def __init__(self, name: str, layout_section_name: str) -> None:
        self.name = name
        self.layout_section_name = layout_section_name
        self._latest_data: Dict[str, Any] = {}


@runtime_checkable
class RendererMetadata(Protocol):
    """Shared renderer metadata required by display drivers."""

    name: str
    layout_section_name: str


@runtime_checkable
class CLIRenderer(RendererMetadata, Protocol):
    """Renderer contract used by the Rich CLI display driver."""

    def get_panel_renderable(self) -> Any:
        """Return a Rich-compatible renderable for the CLI layout."""


@runtime_checkable
class DashboardRenderer(RendererMetadata, Protocol):
    """Renderer contract used by the NiceGUI dashboard display driver."""

    def get_dashboard_renderable(self) -> Any:
        """Return a dashboard payload for the subscribed layout section."""


__all__ = [
    "BaseRenderer",
    "RendererMetadata",
    "CLIRenderer",
    "DashboardRenderer",
]
