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

from tests.step_time.factories import live_result_from_window  # noqa: E402
from traceml_ai.aggregator.display_drivers.nicegui import (  # noqa: E402
    LayoutError,
    NiceGUIDisplayDriver,
)
from traceml_ai.aggregator.display_drivers.layout import (  # noqa: E402
    MODEL_COMBINED_LAYOUT,
    MODEL_DIAGNOSTICS_LAYOUT,
)
from traceml_ai.runtime.settings import TraceMLSettings  # noqa: E402
from traceml_ai.step_time.model import StepTimeWindow  # noqa: E402
from traceml_ai.step_time.pipeline import (  # noqa: E402
    LiveStepTimeFreshness,
    LiveStepTimeResult,
)


def _driver(**settings_kwargs) -> NiceGUIDisplayDriver:
    settings = TraceMLSettings(
        mode="dashboard",
        db_path=tempfile.mktemp(suffix=".db"),
        **settings_kwargs,
    )
    return NiceGUIDisplayDriver(logging.getLogger("test"), settings)


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
    driver._compute_model_payloads = lambda: {}

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


def test_dashboard_sections_build_without_error() -> None:
    from traceml_ai.aggregator.display_drivers.nicegui_sections.model_combined_section import (
        build_model_combined_section,
    )
    from traceml_ai.aggregator.display_drivers.nicegui_sections.model_diagnostics_section import (
        build_model_diagnostics_section,
    )
    from traceml_ai.aggregator.display_drivers.nicegui_sections.process_section import (
        build_process_section,
    )
    from traceml_ai.aggregator.display_drivers.nicegui_sections.step_memory_section import (
        build_step_memory_section,
    )
    from traceml_ai.aggregator.display_drivers.nicegui_sections.system_section import (
        build_gpu_gauge_section,
        build_system_section,
    )

    # Smoke the actual section construction performed when a browser renders
    # the dashboard route. This catches NiceGUI element signature changes.
    build_model_combined_section()
    build_gpu_gauge_section()
    build_system_section()
    build_process_section()
    build_step_memory_section()
    build_model_diagnostics_section()


@pytest.mark.parametrize(
    "freshness",
    ("cold", "live", "bridged", "expired"),
)
def test_dashboard_tick_fans_out_one_step_time_result(
    monkeypatch: pytest.MonkeyPatch,
    freshness: LiveStepTimeFreshness,
) -> None:
    """The hero and rail share one result in every freshness state."""
    driver = _driver()
    result = live_result_from_window(
        StepTimeWindow(),
        freshness=freshness,
    )
    refreshes: list[object] = []
    diagnostics_inputs: list[object] = []

    def refresh() -> LiveStepTimeResult:
        refreshes.append(result)
        return result

    def render_diagnostics(
        step_time: LiveStepTimeResult,
    ) -> dict[str, list[object]]:
        diagnostics_inputs.append(step_time)
        return {"items": []}

    monkeypatch.setattr(driver._step_time_session, "refresh", refresh)
    monkeypatch.setattr(
        driver._model_diagnostics,
        "get_dashboard_renderable",
        render_diagnostics,
    )
    for renderer in driver._renderers:
        monkeypatch.setattr(
            renderer,
            "get_dashboard_renderable",
            lambda: {},
        )

    driver._ui_ready = True
    driver.tick()

    assert refreshes == [result]
    assert diagnostics_inputs == [result]
    assert driver.latest_data[MODEL_COMBINED_LAYOUT] is result
    assert driver.latest_data[MODEL_DIAGNOSTICS_LAYOUT] == {"items": []}


def test_diagnostics_failure_does_not_hide_shared_step_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rail presentation failure must not discard a valid hero result."""
    driver = _driver()
    result = live_result_from_window(StepTimeWindow())

    monkeypatch.setattr(
        driver._step_time_session,
        "refresh",
        lambda: result,
    )

    def fail(_step_time: LiveStepTimeResult) -> None:
        raise RuntimeError("rail failed")

    monkeypatch.setattr(
        driver._model_diagnostics,
        "get_dashboard_renderable",
        fail,
    )

    payloads = driver._compute_model_payloads()

    assert payloads[MODEL_COMBINED_LAYOUT] is result
    assert isinstance(payloads[MODEL_DIAGNOSTICS_LAYOUT], LayoutError)
