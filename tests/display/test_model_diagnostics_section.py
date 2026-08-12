# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostics rail last-known-state tests (issue #353)."""

from __future__ import annotations

import pytest

pytest.importorskip("nicegui")

from traceml_ai.aggregator.display_drivers.nicegui_sections import theme
from traceml_ai.aggregator.display_drivers.nicegui_sections.model_diagnostics_section import (
    build_model_diagnostics_section,
    update_model_diagnostics_section,
)

_HOLDING = "Holding last diagnosis - no update in the latest refresh"


class _FakeText:
    def __init__(self) -> None:
        self.text = ""
        self.styles: list[str] = []

    def style(self, value: str) -> "_FakeText":
        self.styles.append(value)
        return self


class _FakeHtml:
    def __init__(self) -> None:
        self.content = ""


def _panel() -> dict:
    return {
        "overall": _FakeText(),
        "body": _FakeHtml(),
        "hint": _FakeText(),
        "_last": None,
    }


def _payload(status: str = "INPUT-BOUND") -> dict:
    return {
        "overall_severity": "crit",
        "items": [
            {
                "source": "step_time",
                "status": status,
                "severity": "crit",
                "kind": "INPUT_BOUND",
                "reason": "dataloader starves the step",
                "confidence_label": "high",
                "evidence": {"window": "30 steps"},
            }
        ],
    }


def test_build_starts_with_an_empty_cache() -> None:
    panel = build_model_diagnostics_section()

    assert "_last" in panel
    assert panel["_last"] is None


def test_diagnosis_renders_and_is_cached() -> None:
    panel = _panel()

    update_model_diagnostics_section(panel, _payload())

    assert panel["overall"].text == "CRIT"
    assert "INPUT-BOUND" in panel["body"].content
    assert "dataloader starves the step" in panel["body"].content
    assert panel["hint"].text == ""
    assert panel["_last"] == _payload()


def test_empty_refresh_holds_the_last_diagnosis() -> None:
    # #353 regression: ranks settling, end of run, or a dying rank produce a
    # transiently empty read. That is not a new verdict, so the rail must
    # keep the last one instead of blanking to NO DATA.
    panel = _panel()
    update_model_diagnostics_section(panel, _payload())
    rendered = panel["body"].content

    update_model_diagnostics_section(panel, {"items": []})

    assert panel["overall"].text == "CRIT"
    assert panel["body"].content == rendered
    assert panel["hint"].text == _HOLDING


def test_empty_refresh_without_a_cache_shows_no_data() -> None:
    panel = _panel()

    update_model_diagnostics_section(panel, {"items": []})

    assert panel["overall"].text == "NO DATA"
    assert panel["body"].content == ""
    assert panel["hint"].text == "Waiting for diagnostics"
    assert theme.SEV["neutral"] in panel["overall"].styles[-1]


def test_repeated_empty_refresh_does_not_evict_the_cache() -> None:
    panel = _panel()
    update_model_diagnostics_section(panel, _payload())
    rendered = panel["body"].content

    update_model_diagnostics_section(panel, {"items": []})
    update_model_diagnostics_section(panel, {"items": []})

    assert panel["overall"].text == "CRIT"
    assert panel["body"].content == rendered
    assert panel["hint"].text == _HOLDING
    assert panel["_last"] == _payload()


def test_recovered_diagnosis_replaces_the_held_one() -> None:
    panel = _panel()
    update_model_diagnostics_section(panel, _payload())
    update_model_diagnostics_section(panel, {"items": []})
    assert panel["hint"].text == _HOLDING

    update_model_diagnostics_section(panel, _payload("COMPUTE-BOUND"))

    assert "COMPUTE-BOUND" in panel["body"].content
    assert "INPUT-BOUND" not in panel["body"].content
    assert panel["hint"].text == ""
