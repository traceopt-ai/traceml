# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Subscriber/timer leak-prune tests (TRA-68 robustness pass)."""

from __future__ import annotations

import logging
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


def _driver() -> NiceGUIDisplayDriver:
    settings = TraceMLSettings(
        mode="dashboard", db_path=tempfile.mktemp(suffix=".db")
    )
    return NiceGUIDisplayDriver(
        logging.getLogger("test"), RemoteDBStore(), settings
    )


def _fn(cards, data) -> None:  # noqa: ANN001 - test stub
    pass


def test_prune_client_removes_only_that_clients_subscribers() -> None:
    driver = _driver()
    driver._layout_subscribers = {
        "sys": [("c1", {}, _fn), ("c2", {}, _fn)],
        "proc": [("c1", {}, _fn)],
    }
    driver._timer_clients = {"c1", "c2"}

    driver._prune_client("c1")

    # c1 gone everywhere; the now-empty "proc" layout is dropped.
    assert driver._layout_subscribers == {"sys": [("c2", {}, _fn)]}
    assert driver._timer_clients == {"c2"}


def test_prune_client_unknown_id_is_noop() -> None:
    driver = _driver()
    driver._layout_subscribers = {"sys": [("c1", {}, _fn)]}
    driver._timer_clients = {"c1"}

    driver._prune_client("does-not-exist")

    assert driver._layout_subscribers == {"sys": [("c1", {}, _fn)]}
    assert driver._timer_clients == {"c1"}


def test_handle_disconnect_prunes_via_client_id() -> None:
    driver = _driver()
    driver._layout_subscribers = {"sys": [("c1", {}, _fn)]}
    driver._timer_clients = {"c1"}

    class _FakeClient:
        id = "c1"

    driver._handle_disconnect(_FakeClient())

    assert driver._layout_subscribers == {}
    assert driver._timer_clients == set()


def test_prune_client_removes_staleness_label() -> None:
    driver = _driver()
    driver._staleness_labels = {"c1": object(), "c2": object()}
    driver._prune_client("c1")
    assert "c1" not in driver._staleness_labels
    assert "c2" in driver._staleness_labels


def test_repeated_connect_disconnect_does_not_leak() -> None:
    driver = _driver()
    for i in range(100):
        cid = f"client-{i}"
        driver._layout_subscribers.setdefault("sys", []).append((cid, {}, _fn))
        driver._timer_clients.add(cid)
        driver._prune_client(cid)

    assert driver._layout_subscribers == {}
    assert driver._timer_clients == set()
