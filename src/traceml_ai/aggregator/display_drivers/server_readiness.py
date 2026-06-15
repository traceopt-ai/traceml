# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Server-readiness helpers for the NiceGUI dashboard driver (TRA-68).

The dashboard server runs ``ui.run(...)`` (blocking) inside a daemon thread.
Startup must wait for the *server socket to be listening*, never for a browser
to render a page, and must always return within a bounded time so a headless
or remote training run is never blocked.

Why a socket probe and not just ``app.on_startup``: NiceGUI runs its
``on_startup`` handlers inside the uvicorn ASGI lifespan, which completes
*before* the socket is bound (uvicorn ``Server.startup`` awaits
``lifespan.startup()`` and only then calls ``loop.create_server``). So an
``on_startup`` signal means "our lifespan ran", not "we are listening". We use
that signal only to gate the socket probe (so a foreign listener on the same
port can never be mistaken for our server), and the probe to confirm the bind.
On a bind failure uvicorn calls ``sys.exit(1)`` (``SystemExit``), killing the
daemon thread; ``is_alive`` becoming False is our failure signal.
"""

from __future__ import annotations

import enum
import socket
import time
from typing import Callable


class ServerReadiness(enum.Enum):
    """Outcome of waiting for the dashboard server to come up."""

    READY = "ready"  # our server's socket is accepting connections
    FAILED = "failed"  # the server thread exited (e.g. port already in use)
    TIMEOUT = "timeout"  # neither happened before the deadline


def socket_is_listening(host: str, port: int, timeout: float = 0.2) -> bool:
    """Return True if a TCP connection to ``host:port`` succeeds."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def wait_for_server_ready(
    *,
    is_listening: Callable[[], bool],
    is_alive: Callable[[], bool],
    lifespan_started: Callable[[], bool],
    timeout: float,
    interval: float = 0.05,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> ServerReadiness:
    """Wait for the dashboard server to become ready, fail, or time out.

    Returns:
        READY   - our lifespan ran and the socket is accepting connections.
        FAILED  - the server thread is no longer alive (e.g. port bind failed).
        TIMEOUT - neither condition held before ``timeout`` elapsed.

    Never blocks indefinitely. ``lifespan_started`` gates the listening check so
    a pre-existing foreign listener on the same port is not mistaken for ours.
    """
    deadline = monotonic() + timeout
    while True:
        alive = is_alive()
        # READY only if our server thread is alive, our lifespan has started,
        # and the socket accepts a connection. Requiring "alive" stops a dead
        # thread plus a foreign listener on the same port from looking ready.
        if alive and lifespan_started() and is_listening():
            return ServerReadiness.READY
        if not alive:
            return ServerReadiness.FAILED
        if monotonic() >= deadline:
            return ServerReadiness.TIMEOUT
        sleep(interval)
