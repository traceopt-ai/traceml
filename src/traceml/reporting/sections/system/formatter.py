"""
Text formatter for the final-report system section.
"""

from __future__ import annotations

from typing import Any, Dict


def format_system_section_text(payload: Dict[str, Any]) -> str:
    """
    Return the compact system card text from a system-section payload.
    """
    return str(payload.get("card", ""))


__all__ = [
    "format_system_section_text",
]
