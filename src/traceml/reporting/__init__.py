# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""TraceML reporting package."""

from __future__ import annotations

from typing import Any

_FINAL_EXPORTS = {
    "DEFAULT_FINAL_REPORT_GENERATOR",
    "FinalReportGenerator",
    "build_summary_payload",
    "generate_summary",
    "write_summary_artifacts",
}


def __getattr__(name: str) -> Any:
    """Load final-summary exports only when callers request them."""
    if name not in _FINAL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from traceml.reporting import final

    value = getattr(final, name)
    globals()[name] = value
    return value


__all__ = [
    "DEFAULT_FINAL_REPORT_GENERATOR",
    "FinalReportGenerator",
    "build_summary_payload",
    "generate_summary",
    "write_summary_artifacts",
]
