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

from traceml_ai.aggregator.display_drivers import (  # noqa: E402
    nicegui as nicegui_driver,
)
from traceml_ai.aggregator.display_drivers.nicegui import (  # noqa: E402
    NiceGUIDisplayDriver,
    ServerReadiness,
    ServerWatchOutcome,
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


# ---------------------------------------------------------------------------
# Console notices (#488): the error logger is file-only and ERROR-level, so
# every startup outcome is also printed to stderr, which the launcher mirrors
# to the terminal in dashboard mode. The error-log records stay as they were.
# ---------------------------------------------------------------------------

_LOGGER = "test.serving"


def _dashboard_lines(err: str) -> list[str]:
    return [
        line
        for line in err.splitlines()
        if line.startswith("[TraceML] Dashboard")
    ]


def _messages(caplog: pytest.LogCaptureFixture, level: int) -> str:
    return "\n".join(
        r.getMessage() for r in caplog.records if r.levelno == level
    )


def test_ready_prints_url_and_keeps_info_log(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    driver = _driver(_free_port())
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        driver._log_startup_result(ServerReadiness.READY)

    assert _dashboard_lines(capsys.readouterr().err) == [
        f"[TraceML] Dashboard ready at http://localhost:{driver._port}"
    ]
    assert "Dashboard ready at" in _messages(caplog, logging.INFO)


def test_taken_port_prints_port_and_flag_and_keeps_error_log(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    try:
        driver = _driver(port)
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            driver.start()
    finally:
        holder.close()

    assert driver._server_thread is None  # the pre-check short-circuited
    assert _dashboard_lines(capsys.readouterr().err) == [
        f"[TraceML] Dashboard could not start: port {port} is already in "
        f"use. Training continues without the dashboard. Pass "
        f"--dashboard-port <free port> to use another port."
    ]
    assert f"failed to start on port {port}" in _messages(
        caplog, logging.ERROR
    )


def test_server_thread_failing_early_prints_one_line_without_blaming_port(
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _broken_pages(_driver: object) -> None:
        raise RuntimeError("page setup broke")

    # The real _start_ui_server runs and hits its exception path; start()
    # then sees the thread gone and reports FAILED. Exactly one console
    # line, and it does not claim a port conflict it cannot know about.
    monkeypatch.setattr(nicegui_driver, "define_pages", _broken_pages)
    driver = _driver(_free_port())
    driver._startup_timeout_sec = 10.0
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        driver.start()

    lines = _dashboard_lines(capsys.readouterr().err)
    assert lines == [
        f"[TraceML] Dashboard did not start on port {driver._port}; "
        f"training continues without it. Details: see traceml_errors.log "
        f"in the session's aggregator directory."
    ]
    errors = _messages(caplog, logging.ERROR)
    assert "page setup broke" in errors
    assert f"failed to start on port {driver._port}" in errors


def test_timeout_prints_notice_and_keeps_warning_log(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    driver = _driver(_free_port())
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        driver._log_startup_result(ServerReadiness.TIMEOUT)

    assert _dashboard_lines(capsys.readouterr().err) == [
        f"[TraceML] Dashboard not confirmed within 10s; continuing. It may "
        f"still come up at http://localhost:{driver._port}."
    ]
    assert "not confirmed within" in _messages(caplog, logging.WARNING)


def _watch_with(
    monkeypatch: pytest.MonkeyPatch, outcome: ServerWatchOutcome
) -> NiceGUIDisplayDriver:
    monkeypatch.setattr(
        nicegui_driver, "watch_server_startup", lambda **_kw: outcome
    )
    return _driver(_free_port())


def test_watchdog_ready_late_prints_url_and_keeps_info_log(
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _watch_with(monkeypatch, ServerWatchOutcome.READY_LATE)
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        driver._watch_startup()

    assert _dashboard_lines(capsys.readouterr().err) == [
        f"[TraceML] Dashboard ready at http://localhost:{driver._port}"
    ]
    assert "Dashboard ready at" in _messages(caplog, logging.INFO)


@pytest.mark.parametrize(
    ("outcome", "logged"),
    [
        (ServerWatchOutcome.THREAD_DIED, "exited without binding"),
        (ServerWatchOutcome.STILL_NOT_LISTENING, "still not listening"),
    ],
)
def test_watchdog_failure_prints_short_line_and_keeps_error_log(
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    outcome: ServerWatchOutcome,
    logged: str,
) -> None:
    driver = _watch_with(monkeypatch, outcome)
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        driver._watch_startup()

    err = capsys.readouterr().err
    assert _dashboard_lines(err) == [
        f"[TraceML] Dashboard did not start on port {driver._port}; "
        f"training continues without it. Details: see traceml_errors.log "
        f"in the session's aggregator directory."
    ]
    # The stack and diagnostics stay in the log file, not on the console.
    assert "Diagnostics" not in err and "server_thread_alive" not in err
    errors = _messages(caplog, logging.ERROR)
    assert logged in errors and "server_thread_alive" in errors


def test_console_print_failure_never_raises(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _BrokenStream:
        def write(self, _text: str) -> int:
            raise OSError("stderr is gone")

        def flush(self) -> None:
            raise OSError("stderr is gone")

    monkeypatch.setattr(sys, "stderr", _BrokenStream())
    driver = _driver(_free_port())
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        driver._log_startup_result(ServerReadiness.READY)

    assert "Dashboard ready at" in _messages(caplog, logging.INFO)
