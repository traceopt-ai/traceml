# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Dashboard error surfacing tests (TRA-68 robustness pass)."""

from __future__ import annotations

import logging
import tempfile

import pytest

pytest.importorskip("nicegui")

from traceml_ai.aggregator.display_drivers.nicegui import (  # noqa: E402
    LayoutError,
    NiceGUIDisplayDriver,
    render_error,
)
from traceml_ai.database.remote_database_store import (  # noqa: E402
    RemoteDBStore,
)
from traceml_ai.runtime.settings import TraceMLSettings  # noqa: E402


class _FakeWidget:
    def __init__(self) -> None:
        self.text = ""


def _driver() -> NiceGUIDisplayDriver:
    settings = TraceMLSettings(
        mode="dashboard", db_path=tempfile.mktemp(suffix=".db")
    )
    return NiceGUIDisplayDriver(
        logging.getLogger("test"), RemoteDBStore(), settings
    )


def test_update_display_captures_error_as_layout_error() -> None:
    driver = _driver()
    driver._ui_ready = True

    def boom() -> None:
        raise RuntimeError("db is locked")

    driver._layout_content_fns = {"sys": boom}
    driver.update_display()
    payload = driver.latest_data["sys"]
    assert isinstance(payload, LayoutError)
    assert "db is locked" in payload.message


def test_render_error_sets_visible_message() -> None:
    cards = {"a": _FakeWidget(), "b": _FakeWidget()}
    render_error(cards, "db is locked")
    assert all("db is locked" in widget.text for widget in cards.values())


def test_ui_update_loop_surfaces_error_and_skips_update_fn() -> None:
    driver = _driver()
    driver._ui_ready = True
    cards = {"a": _FakeWidget()}
    called: list = []
    driver._layout_subscribers = {
        "sys": [(None, cards, lambda c, d: called.append(d))]
    }
    driver.latest_data = {"sys": LayoutError("db is locked")}
    driver._ui_update_loop()
    assert "db is locked" in cards["a"].text
    assert called == []  # update_fn is skipped for error payloads
