# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NiceGUI display driver (TRA-68 robustness pass)."""

from __future__ import annotations

import logging
import socket
import tempfile

import pytest

pytest.importorskip("nicegui")

from traceml_ai.aggregator.display_drivers.nicegui import (  # noqa: E402
    NiceGUIDisplayDriver,
)
from traceml_ai.database.remote_database_store import (  # noqa: E402
    RemoteDBStore,
)
from traceml_ai.runtime.settings import TraceMLSettings  # noqa: E402


def _driver(**settings_kwargs) -> NiceGUIDisplayDriver:
    settings = TraceMLSettings(
        mode="dashboard",
        db_path=tempfile.mktemp(suffix=".db"),
        **settings_kwargs,
    )
    return NiceGUIDisplayDriver(
        logging.getLogger("test"), RemoteDBStore(), settings
    )


def test_settings_dashboard_defaults() -> None:
    settings = TraceMLSettings()
    assert settings.dashboard_port == 8765
    assert settings.dashboard_auto_open is True


def test_driver_uses_settings_port() -> None:
    assert _driver(dashboard_port=9999)._port == 9999


def test_driver_uses_settings_auto_open() -> None:
    assert _driver(dashboard_auto_open=False)._show is False


def test_driver_defaults_match_settings_defaults() -> None:
    driver = _driver()
    assert driver._port == 8765
    assert driver._show is True


def test_staleness_text_empty_before_any_data() -> None:
    driver = _driver()
    assert driver._last_data_monotonic is None
    assert driver.staleness_text(now=100.0) == ""


def test_staleness_text_fresh_after_recent_update() -> None:
    driver = _driver()
    driver._last_data_monotonic = 100.0
    assert driver.staleness_text(now=101.0) == ""


def test_staleness_text_stale_after_gap() -> None:
    driver = _driver()
    driver._last_data_monotonic = 100.0
    assert driver.staleness_text(now=120.0) == "stale 20s"


def test_update_display_records_timestamp_on_success() -> None:
    driver = _driver()
    driver._ui_ready = True
    driver._layout_content_fns = {"sys": lambda: {"ok": 1}}
    assert driver._last_data_monotonic is None
    driver.update_display()
    assert driver._last_data_monotonic is not None


def test_update_display_keeps_stale_when_all_payloads_error() -> None:
    driver = _driver()
    driver._ui_ready = True

    def boom() -> None:
        raise RuntimeError("x")

    driver._layout_content_fns = {"sys": boom}
    driver.update_display()
    # All sections errored -> not "fresh", timestamp stays unset.
    assert driver._last_data_monotonic is None


def test_ui_loop_updates_staleness_label() -> None:
    driver = _driver()
    driver._ui_ready = True
    driver._last_data_monotonic = 0.0  # ancient -> stale

    class _FakeLabel:
        def __init__(self) -> None:
            self.text = ""

    label = _FakeLabel()
    driver._staleness_labels = {None: label}
    driver._ui_update_loop()
    assert "stale" in label.text


def test_start_returns_when_port_already_in_use() -> None:
    # TRA-68 integration: a held port must not hang start(); the pre-check
    # short-circuits to a failure without launching a doomed server thread.
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        driver = _driver(dashboard_port=port)
        driver.start()
        assert driver._ui_started is True
        assert driver._server_thread is None  # never launched
    finally:
        blocker.close()


def test_define_pages_registers_without_error() -> None:
    from traceml_ai.aggregator.display_drivers.nicegui_sections.pages import (
        define_pages,
    )

    # Smoke: page/route registration (incl. the staleness-label wiring path)
    # must import and run without raising.
    define_pages(_driver())
