"""Shared types for the heuristics package.

Kept in a separate module to prevent circular imports between engine.py
(which imports rule modules) and rule modules (which need Recommendation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Severity = Literal["info", "warn", "crit"]


@dataclass(frozen=True)
class Recommendation:
    kind: str
    severity: Severity
    category: str
    reason: str
    action: str
