# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Numbers as the words a card prints.

Separate from ``theme`` on purpose. ``theme`` owns colours, fonts and CSS;
this module owns the far smaller question of how one already-decided value
is written down. The two were one module and the boundary blurred: chart
builders and value formatters accumulated beside the palette until a change
to how a byte count reads meant editing the file that also defines the
brand.

Nothing here decides what a number MEANS. That is settled in
``renderers/<domain>/`` before the payload is built. These functions only
choose decimals, units and phrasing.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

NA = "n/a"

_BYTES_PER_GB = float(1024**3)


def gb_si(value: Any) -> Optional[float]:
    """Bytes as decimal gigabytes (10^9), or ``None``.

    Distinct from :func:`gb`, which divides by 1024**3. The System card
    has always used this one and the Process card the other, so the same
    device renders 7.4% apart on one page depending on the card. Keeping
    both under names that say which is which is deliberate: collapsing
    them would move a user-visible number inside a file move. The
    unification is tracked separately.
    """
    if value is None:
        return None
    try:
        return float(value) / 1e9
    except (TypeError, ValueError):
        return None


def gb(value: Any) -> Optional[float]:
    """Bytes as gigabytes, or ``None`` when the value is not a number."""
    if value is None:
        return None
    try:
        return float(value) / _BYTES_PER_GB
    except (TypeError, ValueError):
        return None


def num(value: Any, fmt: str = "{:.0f}") -> str:
    """One number in the given format, or the absence marker."""
    if value is None:
        return NA
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return NA


def format_gb_pair(used_bytes: Any, total_bytes: Any) -> Tuple[str, str]:
    """A level against its capacity: ``('6.3', '/ 16.1 GB')``.

    A measured value never renders as "0.0": a trainer process holding
    27 MB is a real reading, and printing it with one decimal makes it
    indistinguishable from the marker this module uses for absence.
    Sub-gigabyte levels keep two decimals instead.
    """
    used = gb(used_bytes)
    if used is None:
        return (NA, "")
    shown = f"{used:.2f}" if 0 < used < 1 else f"{used:.1f}"
    total = gb(total_bytes)
    if total is None or total <= 0:
        return (shown, "GB")
    total_s = f"{total:.0f}" if total >= 100 else f"{total:.1f}"
    return (shown, f"/ {total_s} GB")


def format_span(seconds: Optional[float]) -> str:
    """The window a chart covers, as one phrase: ``'last 3 min'``.

    Seconds until a full two minutes. Switching at ninety and rounding to
    whole minutes announced a 105 second window as "last 2 min", which
    overstates what the chart shows by 14% in the label a reader uses to
    place it in time.
    """
    if not seconds or seconds <= 0:
        return ""
    if seconds < 120:
        return f"last {seconds:.0f} s"
    return f"last {seconds / 60.0:.0f} min"


def format_window(window_s: Optional[float]) -> str:
    """The rolling window in words: ``'30 s'``, ``'2 min'``.

    The compute layer picks round windows, so this reads as a duration a
    person recognises rather than an arbitrary number of seconds.
    """
    if not window_s or window_s <= 0:
        return ""
    if window_s < 60:
        return f"{window_s:.0f} s"
    return f"{window_s / 60.0:.0f} min"


def format_age(seconds: Any) -> str:
    """How long ago a rank last reported, in the strip's vocabulary."""
    if seconds is None:
        return NA
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return NA
    if value < 90:
        return f"{value:.0f} s"
    if value < 90 * 60:
        return f"{value / 60.0:.0f} min"
    return f"{value / 3600.0:.1f} h"


def format_percent(value: Optional[float], fmt: str = "{:.0f}") -> str:
    """A percentage, with a floor so a real small share is not lost.

    A reserved-memory spread of 0.4% is a genuine reading. Rounded to a
    whole percent it prints "0", which reads as balanced.
    """
    if value is None:
        return NA
    try:
        number = float(value)
    except (TypeError, ValueError):
        return NA
    if 0 < number < 1:
        return "<1"
    return fmt.format(number)


__all__ = [
    "NA",
    "format_age",
    "format_gb_pair",
    "gb_si",
    "format_percent",
    "format_span",
    "format_window",
    "gb",
    "num",
]
