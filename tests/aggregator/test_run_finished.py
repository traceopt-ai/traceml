# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""The aggregator tells its display driver once that the run finished.

The quiet after every expected rank sent ``rank_finished`` is expected,
so the terminal's run-wide "no new data" line must not call it a stall.
The aggregator says so from its loop thread, once, the first time every
expected rank has finished, after the arrivals that came with the
markers are stamped. A driver that fails on it costs nothing else.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, List
from unittest.mock import Mock

import pytest

from traceml_ai.aggregator.display_drivers.base import BaseDisplayDriver
from traceml_ai.aggregator.display_drivers.summary import (
    SummaryDisplayDriver,
)
from traceml_ai.aggregator.trace_aggregator import TraceMLAggregator
from traceml_ai.runtime.settings import TraceMLSettings
from traceml_ai.telemetry.control import build_rank_finished_payload

# Ends the loop: the poll that meets it sets the stop event and yields
# nothing, so any polls after it are the shutdown drain's.
STOP = object()


def _finished(rank: int) -> dict:
    return build_rank_finished_payload(
        global_rank=rank, world_size=2, node_rank=0, hostname="worker"
    )


def _telemetry(tag: str) -> dict:
    return {"sampler": "ProcessSampler", "tag": tag}


class _ScriptedTCP:
    """One scripted batch of messages per poll."""

    def __init__(self, polls: List[Any], stop_event: threading.Event):
        self._polls = list(polls)
        self._stop_event = stop_event

    def wait_for_data(self, timeout: float) -> bool:
        return bool(self._polls)

    def poll(self):
        if not self._polls:
            self._stop_event.set()
            return
        batch = self._polls.pop(0)
        if batch is STOP:
            self._stop_event.set()
            return
        yield from batch

    def remaining(self) -> int:
        return len(self._polls)


class _Writer:
    def __init__(self, events: List[str]) -> None:
        self._events = events

    def ingest(self, payload: dict) -> None:
        self._events.append(f"ingest:{payload['tag']}")

    def force_flush(self, timeout_sec: float) -> bool:
        self._events.append("flush")
        return True


class _Driver:
    def __init__(self, events: List[str], fail: bool = False) -> None:
        self._events = events
        self._fail = fail
        self.threads: List[threading.Thread] = []

    def tick(self) -> None:
        self._events.append("tick")

    def run_finished(self) -> None:
        self._events.append("run_finished")
        self.threads.append(threading.current_thread())
        if self._fail:
            raise RuntimeError("driver failed")


def _run_loop(tmp_path: Path, polls: List[Any], *, fail: bool = False):
    """Run the real loop, on the aggregator's own thread, over ``polls``."""
    agg = TraceMLAggregator(
        logger=Mock(),
        stop_event=threading.Event(),
        settings=TraceMLSettings(
            mode="summary",
            render_interval_sec=0.001,
            expected_world_size=2,
            history_enabled=False,
            logs_dir=str(tmp_path),
            db_path=str(tmp_path / "telemetry"),
        ),
    )
    events: List[str] = []
    tcp = _ScriptedTCP(polls, agg._stop_event)
    driver = _Driver(events, fail=fail)
    agg._tcp_server = tcp
    agg._sqlite_writer = _Writer(events)
    agg._summary_service = Mock()
    agg._display_driver = driver
    agg._thread.start()
    agg._thread.join(timeout=10.0)
    assert not agg._thread.is_alive()
    return agg, events, driver, tcp


def test_run_finished_is_called_once_when_every_expected_rank_finished(
    tmp_path,
) -> None:
    """Rank 0 finishes first; the call waits for rank 1, then never again."""
    agg, events, driver, tcp = _run_loop(
        tmp_path,
        [
            [_telemetry("a"), _finished(0)],
            [_telemetry("b")],
            [_finished(1)],
            [_telemetry("c")],
            [_finished(1)],
            [],
        ],
    )

    assert events.count("run_finished") == 1
    finished_at = events.index("run_finished")
    assert events.index("ingest:b") < finished_at < events.index("ingest:c")
    assert driver.threads == [agg._thread]
    assert tcp.remaining() == 0


def test_arrivals_are_flushed_before_the_driver_is_told(tmp_path) -> None:
    """The driver reads the newest arrival at that moment on the clock
    that stamps arrivals at flush, so the markers' batch is flushed first.
    """
    _, events, _, _ = _run_loop(
        tmp_path, [[_telemetry("last"), _finished(0), _finished(1)], []]
    )

    assert events.count("flush") == 1
    assert (
        events.index("ingest:last")
        < events.index("flush")
        < events.index("run_finished")
    )


def test_run_finished_is_never_called_while_a_rank_is_missing(
    tmp_path,
) -> None:
    _, events, _, tcp = _run_loop(
        tmp_path, [[_finished(0)], [_telemetry("a")], [_finished(0)], []]
    )

    assert "run_finished" not in events
    assert "flush" not in events
    assert tcp.remaining() == 0


def test_the_shutdown_drain_can_still_finish_the_run(tmp_path) -> None:
    """The last marker, taken by the drain after the loop, before the
    final tick.
    """
    agg, events, driver, _ = _run_loop(
        tmp_path, [[_finished(0)], [], STOP, [_finished(1)]]
    )

    assert events.count("run_finished") == 1
    assert events[-1] == "tick"
    assert events.index("run_finished") < len(events) - 1
    assert driver.threads == [agg._thread]


def test_a_raising_driver_does_not_break_the_loop(tmp_path) -> None:
    """Logged once, not retried, and the loop keeps draining."""
    agg, events, _, tcp = _run_loop(
        tmp_path,
        [
            [_finished(0), _finished(1)],
            [_telemetry("after")],
            [],
        ],
        fail=True,
    )

    assert events.count("run_finished") == 1
    assert "ingest:after" in events
    assert events[-1] == "tick"
    assert tcp.remaining() == 0
    labels = [c.args[0] for c in agg._logger.exception.call_args_list]
    assert any("run_finished" in label for label in labels)


# --- the drivers that do not care --------------------------------------
class _MinimalDriver(BaseDisplayDriver):
    def start(self) -> None:
        return None

    def tick(self) -> None:
        return None

    def stop(self) -> None:
        return None


def test_a_driver_without_run_finished_inherits_a_no_op() -> None:
    """Not abstract: an existing driver needs no change to keep working."""
    settings = TraceMLSettings()
    minimal = _MinimalDriver(logger=Mock(), settings=settings)
    assert minimal.run_finished() is None
    summary = SummaryDisplayDriver(logger=Mock(), settings=settings)
    assert "run_finished" not in vars(SummaryDisplayDriver)
    assert summary.run_finished() is None


def test_the_dashboard_driver_inherits_the_no_op(tmp_path) -> None:
    pytest.importorskip("nicegui")
    from traceml_ai.aggregator.display_drivers.nicegui import (
        NiceGUIDisplayDriver,
    )

    assert "run_finished" not in vars(NiceGUIDisplayDriver)
    driver = NiceGUIDisplayDriver(
        logging.getLogger("test"),
        TraceMLSettings(mode="dashboard", db_path=str(tmp_path / "t.db")),
    )
    assert driver.run_finished() is None
