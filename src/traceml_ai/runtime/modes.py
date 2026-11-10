# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Shared execution-mode policy, independent of configuration sources."""

DISPLAY_MODES = ("cli", "dashboard", "summary")
SUPPORTED_MODES = (*DISPLAY_MODES, "telemetry")


def resolve_history_enabled(mode: str, requested: bool) -> bool:
    """Keep telemetry history disabled regardless of inherited configuration."""
    return mode != "telemetry" and bool(requested)


def validate_aggregator_options(
    mode: str, exporter: str, html_report: bool
) -> None:
    """Reject incompatible aggregator options before resources are started."""
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"[TraceML] ERROR: Invalid mode {mode!r}. "
            f"Valid modes: {list(SUPPORTED_MODES)}"
        )
    if mode == "telemetry":
        if not str(exporter or "").strip():
            raise ValueError(
                "[TraceML] ERROR: mode=telemetry requires a configured "
                "exporter."
            )
        if html_report:
            raise ValueError(
                "[TraceML] ERROR: mode=telemetry does not support HTML reports "
                "because local history is disabled."
            )
