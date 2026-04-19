"""
Shared text layout helpers for compact end-of-run summaries.

These helpers are intentionally presentation-only. They do not know anything
about metrics, diagnosis, or domain-specific summary logic.
"""

from __future__ import annotations

import textwrap
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


def wrap_lines(text: str, width: int) -> List[str]:
    """
    Wrap one text line into a list of lines that fit within `width`.

    Parameters
    ----------
    text:
        Input text line to wrap.
    width:
        Maximum visible width of each wrapped line.

    Returns
    -------
    List[str]
        Wrapped lines. Blank input is preserved as a single blank line.

    Notes
    -----
    - Preserves whitespace minimally by trimming trailing spaces only.
    - Avoids breaking words when possible.
    - Keeps long summary lines inside the terminal card boundary.
    """
    raw = str(text).rstrip()

    if not raw:
        return [""]

    return textwrap.wrap(
        raw,
        width=max(1, int(width)),
        break_long_words=False,
        break_on_hyphens=False,
        replace_whitespace=False,
        drop_whitespace=True,
    )
