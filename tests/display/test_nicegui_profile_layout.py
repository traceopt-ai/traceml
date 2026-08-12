# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Profile-aware dashboard layout tests (#355).

The dashboard used to build the step-coupled panels for every profile, so a
``traceml watch --mode dashboard`` run showed a Step Time hero and a Step
Memory panel that nothing could ever fill. These tests pin the selection
(which panels exist per profile), the resulting subscriptions, and the one
boundary line that replaces them.
"""

from __future__ import annotations

import logging
import tempfile
from typing import Any, Dict, List, Tuple

import pytest

pytest.importorskip("nicegui")

from nicegui import ui  # noqa: E402

from traceml_ai.aggregator.display_drivers.layout import (  # noqa: E402
    MODEL_COMBINED_LAYOUT,
    MODEL_DIAGNOSTICS_LAYOUT,
    MODEL_MEMORY_LAYOUT,
    PROCESS_LAYOUT,
    SYSTEM_LAYOUT,
)
from traceml_ai.aggregator.display_drivers.nicegui import (  # noqa: E402
    NiceGUIDisplayDriver,
)
from traceml_ai.aggregator.display_drivers.nicegui_sections import (  # noqa: E402,E501
    pages,
)
from traceml_ai.reporting import summary_card  # noqa: E402
from traceml_ai.runtime.settings import TraceMLSettings  # noqa: E402

# The dashboard does not word its own boundary sentence: it renders the
# summary card's. These tests assert that single source rather than pinning
# the text, so the wording stays owned by one module and a change there needs
# no edit here.


def _driver(profile: str) -> NiceGUIDisplayDriver:
    settings = TraceMLSettings(
        mode="dashboard",
        db_path=tempfile.mktemp(suffix=".db"),
        profile=profile,
    )
    return NiceGUIDisplayDriver(logging.getLogger("test"), settings)


def _build_page(profile: str) -> Tuple[NiceGUIDisplayDriver, List[str]]:
    """Build the dashboard body headlessly; return the driver and new texts.

    NiceGUI builds into the auto-index client outside a page context, and that
    client is shared by the whole test session, so only the elements this call
    added are collected.
    """
    driver = _driver(profile)
    client = ui.context.client
    before = set(client.elements)
    pages.build_main_page(driver, profile)
    texts = [
        element.text
        for element_id, element in client.elements.items()
        if element_id not in before
        and isinstance(getattr(element, "text", None), str)
    ]
    return driver, texts


# --- selection ------------------------------------------------------------


def test_run_profile_builds_every_section() -> None:
    assert pages._sections_for_profile("run") == pages.ALL_SECTIONS


def test_watch_profile_excludes_only_the_step_coupled_sections() -> None:
    sections = pages._sections_for_profile("watch")

    assert pages.SECTION_HERO not in sections
    assert pages.SECTION_STEP_MEMORY not in sections
    assert sections == pages.ALL_SECTIONS - pages.STEP_COUPLED_SECTIONS


def test_watch_profile_keeps_the_resource_and_diagnostics_sections() -> None:
    sections = pages._sections_for_profile("watch")

    # The rail stays: it carries the System and Process findings too.
    assert pages.SECTION_DIAGNOSTICS in sections
    assert pages.SECTION_SYSTEM in sections
    assert pages.SECTION_PROCESS in sections
    assert pages.SECTION_GPU_GAUGE in sections


@pytest.mark.parametrize(
    "profile",
    ("run", "deep", "bogus", "", "   ", None),
)
def test_non_watch_profiles_build_the_full_run_layout(profile: Any) -> None:
    assert pages._sections_for_profile(profile) == pages.ALL_SECTIONS


@pytest.mark.parametrize("profile", ("watch", "WATCH", "  Watch  "))
def test_watch_profile_matching_is_normalized(profile: str) -> None:
    # Mirrors the summary card's own strip/lower profile comparison.
    assert pages._is_watch(profile) is True
    assert pages.SECTION_HERO not in pages._sections_for_profile(profile)


# --- boundary line --------------------------------------------------------


def test_watch_boundary_comes_from_the_summary_card() -> None:
    """The dashboard states the card's sentence, never one of its own."""
    card_line = getattr(summary_card, "_WATCH_BOUNDARY", None)

    assert card_line is not None, (
        "summary_card no longer exposes _WATCH_BOUNDARY; re-point the "
        "dashboard at wherever the watch card's boundary sentence now lives."
    )
    assert pages.WATCH_BOUNDARY_LINE is card_line


def test_watch_page_states_the_boundary_once() -> None:
    _driver_obj, texts = _build_page("watch")

    assert texts.count(pages.WATCH_BOUNDARY_LINE) == 1


def test_run_page_states_no_boundary() -> None:
    _driver_obj, texts = _build_page("run")

    assert pages.WATCH_BOUNDARY_LINE not in texts


# --- subscriptions --------------------------------------------------------


def test_watch_page_omits_the_step_layout_subscriptions() -> None:
    driver, _texts = _build_page("watch")

    assert set(driver._layout_subscribers) == {
        SYSTEM_LAYOUT,
        PROCESS_LAYOUT,
        MODEL_DIAGNOSTICS_LAYOUT,
    }


def test_run_page_subscribes_the_step_layouts() -> None:
    driver, _texts = _build_page("run")

    assert set(driver._layout_subscribers) == {
        SYSTEM_LAYOUT,
        PROCESS_LAYOUT,
        MODEL_DIAGNOSTICS_LAYOUT,
        MODEL_COMBINED_LAYOUT,
        MODEL_MEMORY_LAYOUT,
    }


def test_watch_diagnostics_subscriber_updates_the_rail_without_a_hero() -> (
    None
):
    """The shared subscriber must not reach for a hero that was not built."""
    driver, _texts = _build_page("watch")
    _client_id, cards, update_fn = driver._layout_subscribers[
        MODEL_DIAGNOSTICS_LAYOUT
    ][-1]
    payload: Dict[str, Any] = {
        "overall_severity": "warn",
        "items": [
            {
                "source": "system",
                "status": "MEMORY PRESSURE",
                "severity": "warn",
                "reason": "host RAM is nearly full",
            }
        ],
    }

    update_fn(cards, payload)

    assert cards["overall"].text == "WARN"
    assert "MEMORY PRESSURE" in cards["body"].content
