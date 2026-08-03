# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dashboard staleness helper (TRA-68 robustness pass)."""

from __future__ import annotations

from traceml_ai.aggregator.display_drivers.staleness import format_staleness


def test_fresh_data_has_no_indicator() -> None:
    assert format_staleness(1.0, threshold_seconds=5.0) == ""


def test_just_under_threshold_is_fresh() -> None:
    assert format_staleness(4.9, threshold_seconds=5.0) == ""


def test_at_threshold_is_stale() -> None:
    assert format_staleness(5.0, threshold_seconds=5.0) == "stale 5s"


def test_well_past_threshold_reports_age() -> None:
    assert format_staleness(12.7, threshold_seconds=5.0) == "stale 12s"


def test_negative_age_is_treated_as_fresh() -> None:
    # Clock skew should never produce a bogus "stale -3s".
    assert format_staleness(-3.0, threshold_seconds=5.0) == ""
