# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Text formatter for the final-report step-time section.

The current text card is assembled by the builder and stored in the payload.
Keeping this formatter as the public text boundary gives future rich text or
terminal-specific formatting a clear place to live.
"""

from __future__ import annotations

from typing import Any, Dict


def format_step_time_section_text(payload: Dict[str, Any]) -> str:
    """
    Return the compact step-time card text from a section payload.
    """
    return str(payload.get("card", ""))


__all__ = [
    "format_step_time_section_text",
]
