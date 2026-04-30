"""
Final-report summary contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, Sequence


@dataclass(frozen=True)
class SummaryResult:
    """
    Output from one final-report section.

    Attributes
    ----------
    section:
        Stable section key, for example ``"system"`` or ``"step_time"``.
    payload:
        JSON-safe structured payload for programmatic consumers.
    text:
        Human-readable text card for CLI and report output.
    """

    section: str
    payload: Dict[str, Any] = field(default_factory=dict)
    text: str = ""


class SummarySection(Protocol):
    """Build one final-report section from persisted telemetry."""

    name: str

    def build(self, db_path: str) -> SummaryResult:
        """Build the section for a telemetry database."""


class ReportGenerator(Protocol):
    """Build a final report from registered summary sections."""

    sections: Sequence[SummarySection]

    def generate(self, db_path: str) -> Dict[str, Any]:
        """Generate a JSON-safe final report payload."""


__all__ = [
    "ReportGenerator",
    "SummaryResult",
    "SummarySection",
]
