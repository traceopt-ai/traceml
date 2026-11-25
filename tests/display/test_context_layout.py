# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""The context strip has its own renderer, layout key, and subscription.

The strip must not ride on the System payload: a System window that is
empty or stale says nothing about which run this is or which ranks are
reporting. The driver feeds CONTEXT_LAYOUT from ``ContextRenderer`` on
every tick, in both profiles, and ``update_context_section`` consumes that
payload directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

pytest.importorskip("nicegui")

from tests.sqlite_fixtures import (  # noqa: E402
    init_summary_schema,
    insert_process_sample,
    insert_system_sample,
    sqlite_database,
)
from traceml_ai.aggregator.display_drivers.layout import (  # noqa: E402
    CONTEXT_LAYOUT,
    SYSTEM_LAYOUT,
)
from traceml_ai.aggregator.display_drivers.nicegui import (  # noqa: E402
    NiceGUIDisplayDriver,
)
from traceml_ai.aggregator.display_drivers.nicegui_sections.context_section import (  # noqa: E402,E501
    update_context_section,
)
from traceml_ai.renderers.base_renderer import DashboardRenderer  # noqa: E402
from traceml_ai.renderers.context.renderer import ContextRenderer  # noqa: E402
from traceml_ai.runtime.settings import TraceMLSettings  # noqa: E402


def _write(db: Path) -> None:
    with sqlite_database(db, init_summary_schema) as conn:
        for seq in range(3):
            insert_system_sample(
                conn,
                row_id=seq + 1,
                rank=0,
                ts=100.0 + 2.0 * seq,
                gpu_available=False,
                gpu_count=0,
                world_size=2,
                hostname="node-a",
                seq=seq,
            )
            for rank in range(2):
                insert_process_sample(
                    conn,
                    row_id=10 + seq * 2 + rank,
                    rank=rank,
                    ts=100.0 + 2.0 * seq,
                    gpu_available=False,
                    gpu_count=0,
                    global_rank=rank,
                    world_size=2,
                    hostname="node-a",
                    seq=seq,
                )


def _driver(db: Path, profile: str) -> NiceGUIDisplayDriver:
    settings = TraceMLSettings(
        mode="dashboard", profile=profile, db_path=str(db)
    )
    return NiceGUIDisplayDriver(logging.getLogger("test"), settings)


def test_context_renderer_owns_its_layout(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    _write(db)
    renderer = ContextRenderer(db_path=str(db))
    assert isinstance(renderer, DashboardRenderer)
    assert renderer.layout_section_name == CONTEXT_LAYOUT
    payload = renderer.get_dashboard_renderable()
    assert payload["world_size"] == 2
    assert payload["ranks_reporting"] == 2
    assert payload["last_data_ts"] == 104.0


@pytest.mark.parametrize("profile", ("run", "watch"))
def test_driver_feeds_context_layout_in_both_profiles(
    tmp_path: Path, profile: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "t.db"
    _write(db)
    driver = _driver(db, profile)
    monkeypatch.setattr(
        driver._step_time_session, "refresh", lambda: {"steps": []}
    )
    monkeypatch.setattr(
        driver._model_diagnostics,
        "get_dashboard_renderable",
        lambda _st: {"items": []},
    )
    driver._register_once()
    assert CONTEXT_LAYOUT in driver._layout_content_fns
    driver._ui_ready = True
    driver.tick()
    ctx = driver.latest_data[CONTEXT_LAYOUT]
    assert ctx["ranks_reporting"] == 2
    assert ctx["world_size"] == 2
    # The System payload no longer carries the strip's facts.
    system = driver.latest_data[SYSTEM_LAYOUT]
    # The System payload is typed now; its context carries only the fields
    # the System block owns, and the strip's cross-rank count is not one.
    assert not hasattr(system.rollups.ctx, "ranks_reporting")


def test_update_context_section_consumes_the_context_payload() -> None:
    class _Label:
        def __init__(self, text: str = "") -> None:
            self.text = text
            self.styles: list[str] = []

        def style(self, value: str) -> "_Label":
            self.styles.append(value)
            return self

    labels = (
        "Script",
        "Profile",
        "GPUs detected",
        "Observed duration",
        "Telemetry coverage",
        "Last sample",
    )
    cards = {
        "execution": _Label(),
        "status": _Label(),
        "dot": _Label(),
        "detail_values": {label: _Label() for label in labels},
        "identity": {"script_name": "train.py", "profile": "run"},
        "live_threshold_s": 6.0,
    }
    payload = {
        "world_size": 4,
        "ranks_reporting": 3,
        "gpu_count": 4,
        "gpus_observed": 4,
        "node_count": 1,
        "training_strategy": "ddp",
        "first_data_ts": 1000.0,
        "last_data_ts": 1146.0,
    }
    update_context_section(cards, payload, now=1147.0)
    assert cards["execution"].text == "DDP · 4 processes · 1 node"
    assert cards["status"].text == "Updating"
    details = {label: w.text for label, w in cards["detail_values"].items()}
    assert details == {
        "Script": "train.py",
        "Profile": "run",
        "GPUs detected": "4",
        "Observed duration": "2m 26s",
        "Telemetry coverage": "3 / 4 ranks",
        "Last sample": "1s ago",
    }
    # Stale data says so, with its age, and is not dressed as a failure.
    update_context_section(cards, payload, now=1146.0 + 47)
    assert cards["status"].text == "Delayed · 47s since last data"
    # A payload shaped like the old System rollups is ignored, not misread.
    update_context_section(cards, {"rollups": {"ctx": payload}})
    assert cards["execution"].text == "DDP · 4 processes · 1 node"


def _build(identity, **kwargs):
    from nicegui import ui

    from traceml_ai.aggregator.display_drivers.nicegui_sections.context_section import (  # noqa: E501
        build_context_section,
    )

    with ui.element("div"):
        return build_context_section(identity, lambda _proxy: None, **kwargs)


_LONG_PATH = (
    "/very/long/path/to/the/training/logs/directory/for/this/team/"
    "bert-baseline-with-a-long-run-name/final_summary.json"
)


def test_header_labels_run_execution_and_data_status() -> None:
    cards = _build(
        {
            "profile": "run",
            "run_name": "bert-baseline",
            "run_name_source": "run_name",
            "script_name": "train.py",
            "artifact_path": _LONG_PATH,
        }
    )
    assert cards["identity_label"].text == "Run:"
    assert cards["run_name"].text == "bert-baseline"
    assert cards["status"].text == "Waiting"
    assert cards["execution"].text == "unknown"
    # The field holds the complete path, not an abbreviation of it.
    assert cards["artifact"].value == _LONG_PATH
    assert "readonly" in cards["artifact"].props
    assert cards["copy"].text == "Copy"


def test_watch_profile_reads_session_and_unnamed_run_is_muted() -> None:
    cards = _build(
        {
            "profile": "watch",
            "run_name": "session_20260821_123224_1103e8",
            "run_name_source": "generated",
            "artifact_path": "",
        }
    )
    assert cards["identity_label"].text == "Session:"
    assert cards["run_name"].text == "Unnamed run"
    # No path yet: the row is hidden rather than showing an empty field.
    assert cards["artifact_row"]._style.get("display") == "none"


def test_details_start_collapsed_and_toggle() -> None:
    from traceml_ai.aggregator.display_drivers.nicegui_sections.context_section import (  # noqa: E501
        set_details_open,
    )

    cards = _build({"profile": "run", "run_name": "r", "artifact_path": ""})
    assert cards["details_open"] is False
    assert cards["details"]._style.get("display") == "none"
    assert cards["details_toggle"].props.get("aria-expanded") == "false"

    set_details_open(cards, True)
    assert cards["details"]._style.get("display") == "grid"
    assert cards["details_toggle"].text == "Details ▴"
    assert cards["details_toggle"].props.get("aria-expanded") == "true"

    set_details_open(cards, False)
    assert cards["details"]._style.get("display") == "none"
    assert cards["details_toggle"].text == "Details ▾"


def test_long_run_name_is_clipped_and_kept_whole() -> None:
    name = "x" * 300
    cards = _build(
        {"profile": "run", "run_name": name, "run_name_source": "run_name"}
    )
    style = cards["run_name"]._style
    assert style.get("white-space") == "nowrap"
    assert style.get("text-overflow") == "ellipsis"
    assert style.get("overflow") == "hidden"
    assert cards["run_name"].text == name
