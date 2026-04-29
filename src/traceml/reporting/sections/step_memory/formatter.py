"""
Text formatter for the final-report step-memory section.
"""

from __future__ import annotations

from typing import Any, Dict


def format_step_memory_section_text(payload: Dict[str, Any]) -> str:
    """
    Return the compact step-memory card text from a section payload.
    """
    return str(payload.get("card", ""))


__all__ = [
    "format_step_memory_section_text",
]
