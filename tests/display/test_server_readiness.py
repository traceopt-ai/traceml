# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dashboard server-readiness helpers (TRA-68)."""

from __future__ import annotations

import socket

from traceml_ai.aggregator.display_drivers.server_readiness import (
    ServerReadiness,
    socket_is_listening,
    wait_for_server_ready,
)


def test_socket_is_listening_true_for_bound_port() -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        assert socket_is_listening("127.0.0.1", port) is True
    finally:
        srv.close()


def test_socket_is_listening_false_for_free_port() -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()  # port is now free
    assert socket_is_listening("127.0.0.1", port, timeout=0.1) is False


def test_wait_ready_when_listening_after_lifespan() -> None:
    result = wait_for_server_ready(
        is_listening=lambda: True,
        is_alive=lambda: True,
        lifespan_started=lambda: True,
        timeout=1.0,
    )
    assert result is ServerReadiness.READY


def test_wait_failed_when_thread_dies() -> None:
    result = wait_for_server_ready(
        is_listening=lambda: False,
        is_alive=lambda: False,
        lifespan_started=lambda: False,
        timeout=1.0,
    )
    assert result is ServerReadiness.FAILED


def test_wait_failed_when_thread_dies_even_with_foreign_listener() -> None:
    # A foreign process holds the port (is_listening True) but our server
    # thread died and our lifespan never started -> FAILED, not false READY.
    result = wait_for_server_ready(
        is_listening=lambda: True,
        is_alive=lambda: False,
        lifespan_started=lambda: False,
        timeout=1.0,
    )
    assert result is ServerReadiness.FAILED


def test_wait_failed_when_thread_dies_after_lifespan_foreign_listener() -> (
    None
):
    # on_startup fired (lifespan True) but our bind then failed and the thread
    # died, while a foreign process holds the port (listening True). A live
    # thread is required for READY, so this must classify as FAILED.
    result = wait_for_server_ready(
        is_listening=lambda: True,
        is_alive=lambda: False,
        lifespan_started=lambda: True,
        timeout=1.0,
    )
    assert result is ServerReadiness.FAILED


def test_wait_not_ready_before_our_lifespan_started() -> None:
    # Socket is listening (foreign) and thread alive, but our lifespan has not
    # started -> must NOT report READY; bounded by the timeout.
    result = wait_for_server_ready(
        is_listening=lambda: True,
        is_alive=lambda: True,
        lifespan_started=lambda: False,
        timeout=0.0,
    )
    assert result is ServerReadiness.TIMEOUT


def test_wait_timeout_when_never_ready() -> None:
    result = wait_for_server_ready(
        is_listening=lambda: False,
        is_alive=lambda: True,
        lifespan_started=lambda: True,
        timeout=0.0,
    )
    assert result is ServerReadiness.TIMEOUT
