# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Dashboard serving floor: the server must bind, and a server that does not
bind must say so at ERROR level with enough detail to diagnose it.

Background: on a GPU box the dashboard once printed NiceGUI's "ready" banner
and then never listened on its port, with empty error logs. The banner is
printed from the ASGI lifespan, which runs before the socket is bound, and
the driver only logged the not-yet-listening case at WARNING, below the error
logger's floor. These tests pin both halves of the fix.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest

pytest.importorskip("nicegui")

from traceml_ai.aggregator.display_drivers.nicegui import (  # noqa: E402
    NiceGUIDisplayDriver,
)
from traceml_ai.runtime.settings import TraceMLSettings  # noqa: E402


def _free_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _driver(port: int) -> NiceGUIDisplayDriver:
    settings = TraceMLSettings(
        mode="dashboard",
        db_path=tempfile.mktemp(suffix=".db"),
        dashboard_port=port,
        dashboard_auto_open=False,
    )
    return NiceGUIDisplayDriver(logging.getLogger("test.serving"), settings)


def test_never_binding_server_escalates_to_error_with_thread_stack(
    caplog: pytest.LogCaptureFixture,
) -> None:
    driver = _driver(_free_port())
    release = threading.Event()

    def _stuck_server_never_binds() -> None:
        driver._lifespan_started.set()  # banner-equivalent: lifespan ran
        release.wait(timeout=30)  # ...but the socket is never bound

    driver._start_ui_server = _stuck_server_never_binds  # type: ignore
    driver._startup_timeout_sec = 0.0
    # A short but non-zero grace: the watchdog keeps probing until it
    # elapses, so the stub thread has certainly run (and set the lifespan
    # flag) by the time the diagnostics are sampled.
    driver._startup_grace_sec = 1.0

    with caplog.at_level(logging.INFO, logger="test.serving"):
        driver.start()
        watchdog = driver._startup_watchdog
        assert watchdog is not None
        watchdog.join(timeout=10)
    release.set()

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors, "no ERROR was logged for a server that never bound"
    text = "\n".join(r.getMessage() for r in errors)
    assert "not listening" in text
    assert f"port={driver._port}" in text
    assert "server_thread_alive=True" in text
    assert "lifespan_started=True" in text
    # The diagnostics carry the server thread's live Python stack, so a stuck
    # startup names the function it is stuck in.
    assert "_stuck_server_never_binds" in text


# Runs in a child process on purpose: NiceGUI keeps process-global state
# (registered pages, script-mode detection, its pytest port hook), so an
# in-process start is order-dependent on other display tests. A child
# process is also exactly how the aggregator starts the dashboard.
_CHILD = """
import logging, sys, tempfile, urllib.request
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")
from traceml_ai.aggregator.display_drivers.nicegui import NiceGUIDisplayDriver
from traceml_ai.aggregator.display_drivers.server_readiness import (
    socket_is_listening,
)
from traceml_ai.runtime.settings import TraceMLSettings

port = int(sys.argv[1])
driver = NiceGUIDisplayDriver(
    logging.getLogger("child"),
    TraceMLSettings(
        mode="dashboard",
        db_path=tempfile.mktemp(suffix=".db"),
        dashboard_port=port,
        dashboard_auto_open=False,
    ),
)
driver.start()
listening = socket_is_listening("127.0.0.1", port)
status = None
if listening:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10) as r:
        status = r.status
print(f"LISTENING={listening} HTTP={status}", flush=True)
sys.exit(0 if listening and status == 200 else 1)
"""


def test_real_server_binds_and_serves_http() -> None:
    import traceml_ai

    port = _free_port()
    src_root = str(Path(traceml_ai.__file__).resolve().parents[1])
    # pytest exports PYTEST_* markers that NiceGUI reads as "running under
    # pytest" and then switches ui.run() to its screen-test port hook; the
    # child must look like a plain production process.
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (src_root, env.get("PYTHONPATH", "")) if p
    )
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(port)],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    detail = f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert proc.returncode == 0, detail
    assert "LISTENING=True HTTP=200" in proc.stdout, detail
    # The ready line names the serving stack so a field log is diagnosable.
    assert "Dashboard ready at" in proc.stderr, detail
    assert "nicegui" in proc.stderr, detail
