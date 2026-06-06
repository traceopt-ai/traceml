# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Top-level summary view command implementation for TraceML."""

from __future__ import annotations

from pathlib import Path

from traceml_ai.reporting.summary_artifact import (
    extract_summary_text,
    load_summary_artifact,
)


def view_summary(
    summary_path: str | Path,
    *,
    print_to_stdout: bool = True,
) -> str:
    """
    Print and return the stored terminal summary from a summary JSON artifact.

    This is intentionally a read-only artifact view. It does not regenerate
    diagnostics, read telemetry databases, or write derived reports.
    """
    payload = load_summary_artifact(summary_path)
    text = extract_summary_text(payload, path=summary_path)

    if print_to_stdout:
        print(text)

    return text


__all__ = ["view_summary"]
