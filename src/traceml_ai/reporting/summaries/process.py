# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entry point for the process final summary."""

from __future__ import annotations

from typing import Any, Dict

from traceml_ai.reporting.sections.output import persist_section_summary
from traceml_ai.reporting.sections.process import ProcessSummarySection
from traceml_ai.reporting.sections.process.model import MAX_SUMMARY_ROWS


def generate_process_summary_card(
    db_path: str,
    *,
    print_to_stdout: bool = True,
    max_process_rows: int = MAX_SUMMARY_ROWS,
) -> Dict[str, Any]:
    """Generate and persist the end-of-run process summary."""
    result = ProcessSummarySection(
        max_process_rows=max_process_rows,
    ).build(db_path)
    summary = result.payload

    persist_section_summary(
        db_path,
        section_name="process",
        text=result.text,
        payload=summary,
    )

    if print_to_stdout:
        print(result.text)

    return summary


__all__ = [
    "MAX_SUMMARY_ROWS",
    "generate_process_summary_card",
]
