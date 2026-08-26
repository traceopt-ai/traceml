# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Time-based policy for locally persisted telemetry history."""

from __future__ import annotations

import math
import re
from typing import Union

DurationValue = Union[str, int, float]

DEFAULT_HISTORY_RETENTION_S = 30.0 * 60.0
"""Logical history made available to final-report analysis."""

_DURATION = re.compile(
    r"^\s*(?P<value>(?:\d+(?:\.\d*)?|\.\d+))\s*(?P<unit>[smhdSMHD]?)\s*$"
)
_UNIT_SECONDS = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def parse_history_retention(value: DurationValue) -> float:
    """Parse a positive duration, treating a bare number as seconds."""
    if isinstance(value, bool):
        raise ValueError("history retention must be a positive duration")

    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        match = _DURATION.fullmatch(str(value))
        if match is None:
            raise ValueError(
                "history retention must be a positive duration such as "
                "30s, 30m, 2h, or 1d"
            )
        seconds = (
            float(match.group("value"))
            * _UNIT_SECONDS[match.group("unit").lower()]
        )

    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError(
            "history retention must be a finite positive duration"
        )
    return seconds


def retention_cutoff_recv_ts_ns(
    watermark_recv_ts_ns: int,
    retention_s: DurationValue,
) -> int:
    """Return the exclusive receive-time cutoff for one retention horizon."""
    return int(watermark_recv_ts_ns) - int(
        parse_history_retention(retention_s) * 1e9
    )


__all__ = [
    "DEFAULT_HISTORY_RETENTION_S",
    "DurationValue",
    "parse_history_retention",
    "retention_cutoff_recv_ts_ns",
]
