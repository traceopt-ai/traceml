"""The aggregator stops on EOF from its `traceml run` launcher's pipe."""

import os
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from traceml_ai.aggregator import aggregator_main
from traceml_ai.runtime.settings import (
    LAUNCHER_STDIN_WATCH_ENV,
    TraceMLSettings,
)


def test_run_aggregator_finalizes_when_launcher_goes_away(
    monkeypatch, tmp_path
) -> None:
    start = Mock()
    monkeypatch.setattr(aggregator_main, "start_aggregator", start)
    monkeypatch.setattr(
        aggregator_main, "_install_signal_handlers", lambda _: None
    )
    settings = TraceMLSettings(logs_dir=str(tmp_path), finalize_timeout_sec=1)
    read_fd, write_fd = os.pipe()
    result = {}
    runner = threading.Thread(
        target=lambda: result.update(
            rc=aggregator_main.run_aggregator(
                settings, logger=Mock(), launcher_fd=read_fd
            )
        ),
        daemon=True,
    )
    runner.start()
    runner.join(timeout=0.5)
    alive_with_launcher = runner.is_alive()
    # A killed launcher was the reader of the aggregator's stderr pipe too.
    broken = Mock(side_effect=BrokenPipeError(32, "Broken pipe"))
    monkeypatch.setattr(sys, "stderr", Mock(write=broken, flush=broken))
    os.close(write_fd)
    runner.join(timeout=5.0)
    stopped = not runner.is_alive()
    if not stopped:  # unblock a failing run so the suite does not hang
        start.call_args.kwargs["stop_event"].set()
        runner.join(timeout=5.0)
    os.close(read_fd)

    assert alive_with_launcher, "stopped while the launcher was alive"
    assert stopped, "aggregator kept running after the launcher went away"
    assert result.get("rc") == 0
    start.return_value.stop.assert_called_once_with(timeout_sec=1.0)


def test_watchdog_stops_when_read_fails(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stderr", sys.stderr)  # the watchdog swaps it
    stop_event = threading.Event()
    # -1 is never a valid fd, so the read raises EBADF.
    aggregator_main._install_launcher_watchdog(-1, stop_event, Mock())
    assert stop_event.wait(5.0)


def test_windows_watch_polls_instead_of_reading(monkeypatch) -> None:
    # A read left pending on the pipe blocks extension imports on Windows.
    results = [(0, 0), (0, 0), BrokenPipeError(109, "The pipe has ended")]

    def peek(handle):
        assert handle == 42
        result = results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(
        sys.modules, "_winapi", SimpleNamespace(PeekNamedPipe=peek)
    )
    monkeypatch.setitem(
        sys.modules, "msvcrt", SimpleNamespace(get_osfhandle=lambda _: 42)
    )
    monkeypatch.setattr(aggregator_main.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        aggregator_main.os, "read", Mock(side_effect=AssertionError)
    )

    with pytest.raises(BrokenPipeError):
        aggregator_main._wait_for_launcher_eof(7)
    assert results == []


@pytest.mark.parametrize(
    "marker, tty, expected",
    [("0", 0, None), ("1", 0, 7), ("1", 1, None), ("1", None, None)],
)
def test_launcher_fd(monkeypatch, marker, tty, expected) -> None:
    # Only a marked aggregator with a non-terminal stdin watches it.
    monkeypatch.setenv(LAUNCHER_STDIN_WATCH_ENV, marker)
    stdin = Mock(**{"isatty.return_value": tty, "fileno.return_value": 7})
    monkeypatch.setattr(sys, "stdin", None if tty is None else stdin)
    assert aggregator_main._launcher_fd() == expected
