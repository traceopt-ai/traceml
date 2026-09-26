# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""The aggregator tells its display driver when a run finished.

The quiet after every rank of the run sent ``rank_finished`` is expected,
so the terminal's run-wide "no new data" line must not call it a stall.
The aggregator says so from its loop thread, once per run, when every
rank the run expects has finished, after the arrivals that came with the
markers are stamped. ``traceml serve`` hosts one run after another on
one aggregator, so a rank that reports again after its marker has begun
the next run. A driver that fails on it costs nothing else.
"""

from __future__ import annotations

import itertools
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional
from unittest.mock import Mock

import pytest
from rich.console import Console

from traceml_ai.aggregator.display_drivers.base import BaseDisplayDriver
from traceml_ai.aggregator.display_drivers.summary import (
    SummaryDisplayDriver,
)
from traceml_ai.aggregator.trace_aggregator import TraceMLAggregator
from traceml_ai.runtime.sender import SenderIdentity
from traceml_ai.runtime.settings import TraceMLSettings
from traceml_ai.samplers.schema.process import ProcessSample
from traceml_ai.telemetry.control import build_rank_finished_payload
from traceml_ai.telemetry.envelope import build_telemetry_envelope

# Ends the loop: the poll that meets it sets the stop event and yields
# nothing, so any polls after it are the shutdown drain's.
STOP = object()


def _finished(rank: int, world_size: int = 2) -> dict:
    return build_rank_finished_payload(
        global_rank=rank,
        world_size=world_size,
        node_rank=0,
        hostname="worker",
    )


def _telemetry(tag: str, rank: Optional[int] = None) -> dict:
    payload = {"sampler": "ProcessSampler", "tag": tag}
    if rank is not None:
        payload["global_rank"] = rank
    return payload


def _process(rank: int, seq: int) -> dict:
    """One real ProcessSampler envelope, as a rank sends it."""
    sample = ProcessSample(
        sample_idx=seq,
        timestamp=time.time(),
        pid=4242 + rank,
        cpu_percent=50.0,
        cpu_logical_core_count=8,
        ram_used=1e9,
        ram_total=8e9,
        gpu_available=False,
        gpu_count=0,
        gpu=None,
    )
    return build_telemetry_envelope(
        identity=SenderIdentity(global_rank=rank, local_rank=rank),
        sampler_name="ProcessSampler",
        tables={"ProcessTable": [sample.to_wire()]},
    )


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

    def feed(self, batch: List[Any]) -> None:
        self._polls.append(batch)

    def remaining(self) -> int:
        return len(self._polls)


class _Writer:
    """Records ingests and flushes; each flush returns the next result."""

    def __init__(
        self, events: List[str], results: Iterable[bool] = ()
    ) -> None:
        self._events = events
        self._results = iter(results)

    def ingest(self, payload: dict) -> None:
        self._events.append(f"ingest:{payload['tag']}")

    def force_flush(self, timeout_sec: float) -> bool:
        self._events.append("flush")
        return next(self._results, True)


class _Clock:
    """A monotonic clock that moves only when told, or by ``step_s`` per
    read."""

    def __init__(self, now_s: float = 0.0, step_s: float = 0.0) -> None:
        self.now_s = now_s
        self._step_s = step_s

    def __call__(self) -> float:
        self.now_s += self._step_s
        return self.now_s


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


def _aggregator(
    tmp_path: Path,
    *,
    expected_world_size: int = 2,
    render_interval_sec: float = 0.001,
) -> TraceMLAggregator:
    return TraceMLAggregator(
        logger=Mock(),
        stop_event=threading.Event(),
        settings=TraceMLSettings(
            mode="summary",
            render_interval_sec=render_interval_sec,
            expected_world_size=expected_world_size,
            history_enabled=False,
            logs_dir=str(tmp_path),
            db_path=str(tmp_path / "telemetry"),
        ),
    )


def _run_loop(
    tmp_path: Path,
    polls: List[Any],
    *,
    fail: bool = False,
    expected_world_size: int = 2,
    flush_results: Iterable[bool] = (),
    retry_clock: Optional[_Clock] = None,
):
    """Run the real loop, on the aggregator's own thread, over ``polls``."""
    agg = _aggregator(tmp_path, expected_world_size=expected_world_size)
    events: List[str] = []
    tcp = _ScriptedTCP(polls, agg._stop_event)
    driver = _Driver(events, fail=fail)
    agg._tcp_server = tcp
    agg._sqlite_writer = _Writer(events, flush_results)
    agg._summary_service = Mock()
    agg._display_driver = driver
    if retry_clock is not None:
        agg._retry_clock = retry_clock
    agg._thread.start()
    agg._thread.join(timeout=10.0)
    assert not agg._thread.is_alive()
    return agg, events, driver, tcp


def _told(events: List[str]) -> List[int]:
    return [i for i, event in enumerate(events) if event == "run_finished"]


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


# --- one notification per run --------------------------------------------
def test_each_run_on_one_aggregator_is_told_once(tmp_path) -> None:
    """``traceml serve`` hosts run A, then run B, on one aggregator.

    Rank 0 reporting after its marker has begun run B, so run B's finish
    is told as well. Serve expects one rank unless told otherwise.
    """
    _, events, _, tcp = _run_loop(
        tmp_path,
        [
            [_telemetry("a", rank=0), _finished(0, world_size=1)],
            [],
            [_telemetry("b1", rank=0)],
            [_telemetry("b2", rank=0)],
            [_finished(0, world_size=1)],
            [_finished(0, world_size=1)],
            [],
        ],
        expected_world_size=1,
    )

    told = _told(events)
    assert len(told) == 2
    assert events.index("ingest:a") < told[0] < events.index("ingest:b1")
    assert told[1] > events.index("ingest:b2")
    assert tcp.remaining() == 0


def test_a_run_waits_for_the_world_size_its_markers_name(tmp_path) -> None:
    """A 4-rank ``torchrun`` against serve, which expects one rank."""
    _, events, _, _ = _run_loop(
        tmp_path,
        [
            [_telemetry(f"r{rank}", rank=rank) for rank in range(4)],
            [_finished(0, world_size=4)],
            [_finished(1, world_size=4)],
            [_telemetry("r3", rank=3), _finished(2, world_size=4)],
            [_telemetry("r3-last", rank=3)],
            [_finished(3, world_size=4)],
            [],
        ],
        expected_world_size=1,
    )

    assert events.count("run_finished") == 1
    assert events.index("run_finished") > events.index("ingest:r3-last")


def test_a_crashed_run_cannot_finish_the_next_one_early(tmp_path) -> None:
    """Run A's rank 1 died without a marker; run B follows on serve.

    Rank 0 reporting again begins run B, so run A's finished rank 0 no
    longer counts. Run B is told once both of its own ranks finished.
    """
    _, events, _, _ = _run_loop(
        tmp_path,
        [
            [_telemetry("a0", rank=0), _telemetry("a1", rank=1)],
            [_finished(0, world_size=2)],
            [_telemetry("b0", rank=0), _telemetry("b1", rank=1)],
            [_finished(1, world_size=2)],
            [_telemetry("b0-last", rank=0)],
            [_finished(0, world_size=2)],
            [],
        ],
        expected_world_size=1,
    )

    assert events.count("run_finished") == 1
    assert events.index("run_finished") > events.index("ingest:b0-last")


# --- a flush that does not complete ---------------------------------------
def test_a_failed_flush_tells_nothing_until_a_later_one_succeeds(
    tmp_path,
) -> None:
    """Rows still queued would be stamped after the moment told and read
    as a newer arrival, so the driver hears only after a flush completes.
    """
    _, events, _, _ = _run_loop(
        tmp_path,
        [
            [_finished(0), _finished(1)],
            [_telemetry("x")],
            [_telemetry("y")],
            [],
        ],
        flush_results=[False, True],
        retry_clock=_Clock(step_s=1.0),
    )

    flushes = [i for i, event in enumerate(events) if event == "flush"]
    assert len(flushes) == 2
    assert _told(events) == [flushes[1] + 1]
    assert flushes[0] < events.index("ingest:x") < flushes[1]


def test_a_writer_that_never_flushes_is_retried_with_backoff(
    tmp_path,
) -> None:
    """Each attempt may block the loop for up to twice the render
    interval, so attempts back off: one interval, doubling, and never
    more than 30 s apart. The driver is never told.
    """
    agg = _aggregator(tmp_path, expected_world_size=1, render_interval_sec=2.0)
    events: List[str] = []
    agg._sqlite_writer = _Writer(events, itertools.repeat(False))
    agg._display_driver = _Driver(events)
    clock = _Clock()
    agg._retry_clock = clock
    agg._split_telemetry_payloads(_finished(0, world_size=1))

    attempts: List[float] = []
    while clock.now_s < 300.0:
        before = events.count("flush")
        agg._notify_run_finished()
        if events.count("flush") > before:
            attempts.append(clock.now_s)
        clock.now_s += 0.5

    gaps = [later - earlier for earlier, later in zip(attempts, attempts[1:])]
    assert gaps[:5] == [2.0, 4.0, 8.0, 16.0, 30.0]
    assert set(gaps[4:]) == {30.0}
    assert "run_finished" not in events


# --- against the real writer and the real terminal driver ----------------
def _cli_aggregator(tmp_path: Path) -> TraceMLAggregator:
    """A serve-like aggregator: real SQLite writer, real CLI driver."""
    agg = TraceMLAggregator(
        logger=Mock(),
        stop_event=threading.Event(),
        settings=TraceMLSettings(
            mode="cli",
            render_interval_sec=2.0,
            expected_world_size=1,
            history_enabled=True,
            logs_dir=str(tmp_path),
            db_path=str(tmp_path / "telemetry"),
        ),
    )
    agg._tcp_server = _ScriptedTCP([], agg._stop_event)
    agg._sqlite_writer.start()
    driver = agg._display_driver
    driver._create_initial_layout()
    driver._live = Mock()
    return agg


def _step(agg: TraceMLAggregator, *payloads: Any) -> None:
    """One loop iteration over ``payloads``, then the writer's own flush."""
    agg._tcp_server.feed(list(payloads))
    agg._drain_tcp()
    agg._notify_run_finished()
    assert agg._sqlite_writer.force_flush(5.0)


def _screen(driver: Any) -> str:
    console = Console(
        force_terminal=True,
        color_system=None,
        width=160,
        height=60,
        record=True,
    )
    console.print(driver._layout)
    return console.export_text()


def test_rows_ingested_before_the_notification_are_stamped_before_it(
    tmp_path,
) -> None:
    """The writer stamps ``recv_ts_ns`` when it flushes a row; the driver
    records the finish on the same wall clock. Every row that came with
    the marker must be at or before that moment, or the finished run's
    own tail reads as a newer arrival.
    """
    agg = _cli_aggregator(tmp_path)
    try:
        agg._tcp_server.feed(
            [_process(rank=0, seq=seq) for seq in range(1, 301)]
            + [_finished(0, world_size=1)]
        )
        agg._drain_tcp()
        agg._notify_run_finished()
        finished_at_s = agg._display_driver._finished_at_s
    finally:
        assert agg._sqlite_writer.finalize(10.0).ok

    with sqlite3.connect(tmp_path / "telemetry") as conn:
        stamps = [
            row[0]
            for row in conn.execute("SELECT recv_ts_ns FROM process_samples")
        ]
    assert finished_at_s is not None
    assert len(stamps) == 300
    assert max(stamps) / 1e9 <= finished_at_s


class _WallClock:
    """The wall clock, shifted by ``offset_s`` to look ahead in time."""

    def __init__(self) -> None:
        self.offset_s = 0.0

    def __call__(self) -> float:
        return time.time() + self.offset_s


def test_serve_hides_the_line_after_each_run_finishes(tmp_path) -> None:
    """Run A finishes: hidden. Run B reports and goes quiet: live.
    Run B finishes: hidden again.
    """
    agg = _cli_aggregator(tmp_path)
    driver = agg._display_driver
    clock = _WallClock()
    driver._process._computer._cli._now_fn = clock
    driver._now_fn = clock

    def screen_ahead(offset_s: float) -> str:
        clock.offset_s = offset_s
        driver.tick()
        clock.offset_s = 0.0
        return _screen(driver)

    try:
        _step(agg, _process(rank=0, seq=1), _finished(0, world_size=1))
        assert "no new data" not in screen_ahead(60.0)

        _step(agg, _process(rank=0, seq=1))
        assert "no new data for 60s (stale)" in screen_ahead(60.0)

        _step(agg, _finished(0, world_size=1))
        assert "no new data" not in screen_ahead(120.0)
    finally:
        assert agg._sqlite_writer.finalize(10.0).ok


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
