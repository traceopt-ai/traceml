# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Staleness labelling for the dashboard (TRA-68).

Every dashboard section silently re-renders its last-good payload when data
stops flowing, so a wedged display loop looks "frozen-fresh". This helper turns
an age (seconds since the dashboard last refreshed its payloads) into a short
indicator so a stalled dashboard is visible rather than misleading.

Scope note: ``age`` here measures time since the display driver last produced
payloads, which catches a wedged aggregator/display loop. It does not yet catch
"aggregator alive but telemetry stopped" (e.g. training ended) -- that needs a
per-payload data timestamp and is left for a follow-up.
"""

from __future__ import annotations


def format_staleness(
    age_seconds: float, threshold_seconds: float = 5.0
) -> str:
    """Return a short staleness label, or "" when data is fresh.

    Fresh (age below the threshold, or negative due to clock skew) yields an
    empty string so callers can hide the indicator entirely.
    """
    if age_seconds < threshold_seconds:
        return ""
    return f"stale {int(age_seconds)}s"
