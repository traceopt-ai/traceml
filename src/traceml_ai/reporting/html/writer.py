# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""File-writing entry points for the HTML report."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from ...utils.atomic_io import write_text_atomic
from ..summary_artifact import load_summary_artifact
from .document import render_html_report


def write_html_report(
    payload: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    source_label: Optional[str] = None,
) -> Path:
    """Render ``payload`` and write the HTML atomically to ``out_path``."""
    path = Path(out_path)
    html = render_html_report(payload, source_label=source_label)
    write_text_atomic(path, html)
    return path


def render_html_report_from_file(
    summary_json_path: Union[str, Path],
    out_path: Union[str, Path, None] = None,
) -> Path:
    """
    Render an HTML report from a saved ``final_summary.json`` on disk.

    Loads via the strict summary loader (RuntimeError on malformed input).
    Defaults the output to ``<input_stem>.html`` next to the input file.
    The footer source label is the input file name.
    """
    src = Path(summary_json_path)
    payload = load_summary_artifact(src)
    out = Path(out_path) if out_path is not None else src.with_suffix(".html")
    return write_html_report(payload, out, source_label=src.name)


__all__ = ["render_html_report_from_file", "write_html_report"]
