# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Context renderer: which run this is, and how much of it is reporting.

Dashboard-first: the context strip subscribes to CONTEXT_LAYOUT and is fed
by ``get_dashboard_renderable()``. The CLI already prints run identity in
its own header, so the panel form is a minimal fallback, not a CLI section.
"""

from __future__ import annotations

from typing import Any, Dict

from rich.panel import Panel

from traceml_ai.aggregator.display_drivers.layout import CONTEXT_LAYOUT
from traceml_ai.loggers.error_log import get_error_logger
from traceml_ai.renderers.base_renderer import BaseRenderer

from .computer import ContextComputer


class ContextRenderer(BaseRenderer):
    """Renderer for the run-context payload (dashboard strip)."""

    NAME = "Context"

    def __init__(self, db_path: str):
        super().__init__(name=self.NAME, layout_section_name=CONTEXT_LAYOUT)
        self.db_path = db_path
        self._computer = ContextComputer(db_path=self.db_path)
        self._logger = get_error_logger(self.NAME + "Renderer")

    def get_panel_renderable(self) -> Panel:
        """Minimal Rich form; the CLI header carries run identity itself."""
        facts = self._computer.compute()
        ranks = facts.get("ranks_reporting")
        world = facts.get("world_size") or 0
        text = (
            f"ranks {ranks}/{world} reporting"
            if ranks is not None and world
            else "waiting for data"
        )
        return Panel(text, title="Context")

    def get_dashboard_renderable(self) -> Dict[str, Any]:
        """Return the flat context facts dict for CONTEXT_LAYOUT."""
        return self._computer.compute()
