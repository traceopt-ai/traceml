"""
Shared text layout helpers for compact end-of-run summaries.

These helpers are intentionally presentation-only. They do not know anything
about metrics, diagnosis, or domain-specific summary logic.
"""

from typing import List


def border(width: int = 78) -> str:
    """
    Return a fixed-width outer border.
    """
    return "+" + "-" * (width - 2) + "+"


def row(text: str = "", width: int = 78) -> str:
    """
    Return one padded row inside the outer border.
    """
    inner_width = width - 4
    return f"|  {text:<{inner_width}}|"


def indented_block(text: str) -> List[str]:
    """
    Convert a multiline text block into rows suitable for the final summary.

    Empty lines are preserved as blank rows.
    """
    return [line.rstrip() for line in str(text).splitlines()]
