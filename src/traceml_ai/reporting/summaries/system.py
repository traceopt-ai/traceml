# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entry point for the system final summary."""

from __future__ import annotations

from typing import Any, Dict, Optional

from traceml_ai.reporting.sections.output import persist_section_summary
from traceml_ai.reporting.sections.system import SystemSummarySection
from traceml_ai.reporting.sections.system.model import MAX_SUMMARY_ROWS


def generate_system_summary_card(
    db_path: str,
    *,
    node_rank: Optional[int] = None,
    print_to_stdout: bool = True,
    max_system_rows: int = MAX_SUMMARY_ROWS,
) -> Dict[str, Any]:
    """Generate and persist the end-of-run system summary."""
    result = SystemSummarySection(
        node_rank=node_rank,
        max_system_rows=max_system_rows,
    ).build(db_path)
    summary = result.payload

    persist_section_summary(
        db_path,
        section_name="system",
        text=result.text,
        payload=summary,
        replace_text=True,
    )

    if print_to_stdout:
        print(result.text)

    return summary


__all__ = [
    "MAX_SUMMARY_ROWS",
    "generate_system_summary_card",
]
