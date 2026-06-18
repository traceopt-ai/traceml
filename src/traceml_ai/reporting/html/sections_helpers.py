# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Small shared helpers for section/bar rendering (no rendering deps)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def sorted_rows(rows: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """
    Iterate ``groups.rows`` (a JSON object keyed by stringified rank/node).

    Sorted numerically when labels are integer-like, else lexicographically.
    """

    def key(label: str) -> Tuple[int, Any]:
        try:
            return (0, int(label))
        except (TypeError, ValueError):
            return (1, str(label))

    return [(label, rows[label]) for label in sorted(rows, key=key)]


__all__ = ["sorted_rows"]
