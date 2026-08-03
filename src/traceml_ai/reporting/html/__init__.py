# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained HTML run report rendered from final_summary.json."""

from .document import render_html_report
from .writer import render_html_report_from_file, write_html_report

__all__ = [
    "render_html_report",
    "render_html_report_from_file",
    "write_html_report",
]
