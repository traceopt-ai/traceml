"""
Text formatter for the final-report process section.
"""

from __future__ import annotations

from typing import Any, Dict


def format_process_section_text(payload: Dict[str, Any]) -> str:
    """
    Return the compact process card text from a process-section payload.
    """
    return str(payload.get("card", ""))


__all__ = [
    "format_process_section_text",
]
