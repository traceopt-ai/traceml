# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Shared defaults for bounded final-report generation.

These values are intentionally centralized so runtime configuration can later
populate them from YAML without changing section internals.
"""

DEFAULT_SUMMARY_WINDOW_ROWS = 10_000
DEFAULT_SUMMARY_RETENTION_MULTIPLIER = 1.5
DEFAULT_SUMMARY_RETENTION_ROWS = int(
    DEFAULT_SUMMARY_WINDOW_ROWS * DEFAULT_SUMMARY_RETENTION_MULTIPLIER
)


def normalize_summary_window_rows(value: int) -> int:
    """Return a positive summary window row count."""
    return max(1, int(value))


def normalize_summary_retention_rows(value: int) -> int:
    """Return a positive retained row count."""
    return max(1, int(value))


def summary_retention_rows_for_window(window_rows: int) -> int:
    """Return retained rows derived from the active summary window."""
    active_window = normalize_summary_window_rows(window_rows)
    return normalize_summary_retention_rows(
        int(active_window * DEFAULT_SUMMARY_RETENTION_MULTIPLIER)
    )


__all__ = [
    "DEFAULT_SUMMARY_RETENTION_MULTIPLIER",
    "DEFAULT_SUMMARY_RETENTION_ROWS",
    "DEFAULT_SUMMARY_WINDOW_ROWS",
    "normalize_summary_retention_rows",
    "normalize_summary_window_rows",
    "summary_retention_rows_for_window",
]
