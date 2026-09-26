# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""The terminal says so when the whole run stops reporting (issue #358).

The per-rank markers measure each rank against its peers, so a single-rank
run, or every rank stopping together, never looks stale to them. The
run-wide line covers that case: the newest arrival from any rank, on the
aggregator's own clock, judged by the shared freshness policy at the
cadence the ranks were observed to report at. It comes from the Process
panel's per-rank read, so it costs no read of its own. Once every expected
rank finished, the quiet that follows is expected and the line stays
hidden, until something newer arrives.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable, Dict
from unittest.mock import Mock

import pytest
from rich.console import Console
from tests.sqlite_fixtures import (
    init_summary_schema,
    insert_process_sample,
    sqlite_database,
)

from traceml_ai.aggregator.display_drivers.cli import CLIDisplayDriver
from traceml_ai.runtime.settings import TraceMLSettings

T0 = 1_700_000_000.0
INTERVAL_S = 2.0
SAMPLES = 20


def _write_run(
    path: Path,
    last_arrivals: Dict[int, float],
    *,
    cadence_s: float = INTERVAL_S,
    clock_offset_s: float = 0.0,
) -> None:
    """Each rank's samples arrived every ``cadence_s`` seconds.

    ``last_arrivals`` maps a rank to its last arrival, on the aggregator's
    clock, which stamps ``recv_ts_ns``. ``clock_offset_s`` skews the rank
    hosts' own clock, which stamps ``sample_ts_s``.
    """
    with sqlite_database(path, init_summary_schema) as conn:
        for rank, last_arrival in last_arrivals.items():
            for tick in range(SAMPLES):
                arrival = last_arrival - (SAMPLES - 1 - tick) * cadence_s
                insert_process_sample(
                    conn,
                    row_id=int(arrival * 1e9),
                    rank=rank,
                    ts=arrival + clock_offset_s,
                    seq=tick + 1,
                    gpu_available=False,
                    gpu_count=0,
                    cpu_percent=50.0,
                )


def _driver(
    db_path: Path, now_s: float, profile: str = "run"
) -> CLIDisplayDriver:
    """A driver whose Process read ages arrivals at ``now_s``."""
    driver = CLIDisplayDriver(
        logger=Mock(),
        settings=TraceMLSettings(
            profile=profile,
            mode="cli",
            db_path=str(db_path),
            sampler_interval_sec=INTERVAL_S,
        ),
    )
    _use_clock(driver, lambda: now_s)
    driver._create_initial_layout()
    driver._live = Mock()
    return driver


def _use_clock(driver: CLIDisplayDriver, now_fn: Callable[[], float]) -> None:
    """One aggregator clock for the Process read and the driver."""
    driver._process._computer._cli._now_fn = now_fn
    driver._now_fn = now_fn


class _Clock:
    def __init__(self, now_s: float) -> None:
        self.now_s = now_s

    def __call__(self) -> float:
        return self.now_s


def _arrive(path: Path, rank: int, arrival: float, seq: int) -> None:
    """One more sample from ``rank``, arriving at ``arrival``."""
    with sqlite_database(path) as conn:
        insert_process_sample(
            conn,
            row_id=int(arrival * 1e9),
            rank=rank,
            ts=arrival,
            seq=seq,
            gpu_available=False,
            gpu_count=0,
            cpu_percent=50.0,
        )


def _screen(driver: CLIDisplayDriver) -> str:
    console = Console(
        force_terminal=True,
        color_system=None,
        width=160,
        height=60,
        record=True,
    )
    console.print(driver._layout)
    return console.export_text()


@pytest.mark.parametrize("profile", ["run", "watch"])
def test_tick_prints_the_line_when_the_only_rank_stops(
    tmp_path, profile
) -> None:
    """The only rank stopped: nothing else would say so."""
    db_path = tmp_path / "run.db"
    _write_run(db_path, {0: T0})
    driver = _driver(db_path, now_s=T0 + 60.0, profile=profile)

    driver.tick()

    assert "no new data for 60s (stale)" in _screen(driver)


@pytest.mark.parametrize("profile", ["run", "watch"])
def test_tick_prints_the_line_when_every_rank_stops_together(
    tmp_path, profile
) -> None:
    """No rank is behind a peer, so no rank marker can say it."""
    db_path = tmp_path / "run.db"
    _write_run(db_path, {0: T0, 1: T0, 2: T0})
    driver = _driver(db_path, now_s=T0 + 41.6, profile=profile)

    driver.tick()

    screen = _screen(driver)
    assert "no new data for 42s (stale)" in screen
    assert "no data for" not in screen


def test_no_run_wide_line_while_another_rank_reports(tmp_path) -> None:
    """Rank 1 stopped 40 s ago; rank 0 is still reporting.

    The rank marker names rank 1, and the run is not quiet.
    """
    db_path = tmp_path / "run.db"
    _write_run(db_path, {0: T0, 1: T0 - 40.0})
    driver = _driver(db_path, now_s=T0 + 1.0)

    driver.tick()

    screen = _screen(driver)
    assert "rank 1: no data for 40s (stale)" in screen
    assert "no new data" not in screen


def test_tick_prints_no_line_while_data_flows(tmp_path) -> None:
    db_path = tmp_path / "run.db"
    _write_run(db_path, {0: T0})
    driver = _driver(db_path, now_s=T0 + 1.0)

    driver.tick()

    assert "no new data" not in _screen(driver)


def test_no_line_before_the_first_sample(tmp_path) -> None:
    """Nothing has arrived yet; the panels already say they are waiting."""
    db_path = tmp_path / "empty.db"
    with sqlite_database(db_path, init_summary_schema):
        pass
    driver = _driver(db_path, now_s=T0)

    driver.tick()

    assert "no new data" not in _screen(driver)


def test_a_skewed_rank_clock_raises_no_line(tmp_path) -> None:
    """A rank host an hour behind the aggregator is not a stopped run.

    Arrivals are stamped by the aggregator, so the rank's own
    ``sample_ts_s`` never enters the comparison.
    """
    db_path = tmp_path / "skewed.db"
    _write_run(db_path, {0: T0}, clock_offset_s=-3600.0)
    driver = _driver(db_path, now_s=T0 + 1.0)

    driver.tick()

    assert "no new data" not in _screen(driver)


def test_a_rank_slower_than_configured_raises_no_line(tmp_path) -> None:
    """Configured at 2 s, observed at 30 s: a 20 s gap is a healthy one.

    The threshold follows the cadence the rank was seen to report at,
    as the per-rank markers do, not the aggregator's configured one.
    """
    db_path = tmp_path / "slow.db"
    _write_run(db_path, {0: T0}, cadence_s=30.0)
    driver = _driver(db_path, now_s=T0 + 20.0)

    driver.tick()

    assert "no new data" not in _screen(driver)


def test_the_line_costs_no_database_read_of_its_own(
    tmp_path, monkeypatch
) -> None:
    """The verdict comes from the Process panel's read this tick."""
    db_path = tmp_path / "run.db"
    _write_run(db_path, {0: T0})
    driver = _driver(db_path, now_s=T0 + 60.0)
    driver._register_once()
    driver._update_all_sections()

    def no_more_reads(*_args, **_kwargs):
        raise AssertionError("the staleness line opened a connection")

    monkeypatch.setattr(sqlite3, "connect", no_more_reads)
    driver._update_staleness()

    assert "no new data for 60s (stale)" in _screen(driver)


# --- the end of the run --------------------------------------------------
def _finished_quiet_run(
    tmp_path: Path,
) -> tuple[Path, _Clock, CLIDisplayDriver]:
    """The only rank's last sample arrived at T0; it is now T0 + 60 s."""
    db_path = tmp_path / "run.db"
    _write_run(db_path, {0: T0})
    clock = _Clock(T0 + 60.0)
    driver = _driver(db_path, now_s=clock.now_s)
    _use_clock(driver, clock)
    return db_path, clock, driver


def test_run_finished_hides_the_line_from_then_on(tmp_path) -> None:
    """Every expected rank said it finished: the quiet is expected."""
    _, clock, driver = _finished_quiet_run(tmp_path)
    driver.tick()
    assert "no new data for 60s (stale)" in _screen(driver)

    driver.run_finished()
    driver.tick()
    assert "no new data" not in _screen(driver)

    clock.now_s += 600.0
    driver.tick()
    assert "no new data" not in _screen(driver)


def test_the_line_is_live_again_after_a_newer_arrival(tmp_path) -> None:
    """A second run on the same ``traceml serve`` starts, stalls, then
    finishes: each notification moves the finish moment forward.
    """
    db_path, clock, driver = _finished_quiet_run(tmp_path)
    driver.run_finished()

    _arrive(db_path, rank=0, arrival=T0 + 70.0, seq=1)
    clock.now_s = T0 + 71.0
    driver.tick()
    assert "no new data" not in _screen(driver)

    clock.now_s = T0 + 90.0
    driver.tick()
    assert "no new data for 20s (stale)" in _screen(driver)

    driver.run_finished()
    clock.now_s = T0 + 300.0
    driver.tick()
    assert "no new data" not in _screen(driver)


def test_an_arrival_stamped_before_the_finish_is_not_newer(tmp_path) -> None:
    """Rows that came with the finish markers are read on a later tick.

    They are measured against the moment the run finished, not against
    the newest arrival the last read happened to see.
    """
    db_path, clock, driver = _finished_quiet_run(tmp_path)
    driver.tick()
    driver.run_finished()

    _arrive(db_path, rank=0, arrival=T0 + 59.9, seq=SAMPLES + 1)
    clock.now_s = T0 + 80.0
    driver.tick()

    assert "no new data" not in _screen(driver)


def test_run_finished_before_the_first_tick_hides_the_line(tmp_path) -> None:
    """A run short enough to finish before the display first reads."""
    _, _, driver = _finished_quiet_run(tmp_path)
    driver.run_finished()

    driver.tick()

    assert "no new data" not in _screen(driver)
