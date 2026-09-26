"""Step Memory names a rank that stopped reporting (issue #358).

The combined result aligns on the slowest rank's latest completed step, so
a dead rank freezes the panel on its last step. Liveness comes from each
rank's process-sampler heartbeat, which keeps arriving while a surviving
rank blocks in a collective, and is judged by the same rule the Process
dashboard uses.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace
from typing import Mapping

import pytest
from rich.console import Console
from tests.sqlite_fixtures import (
    init_summary_schema,
    insert_process_sample,
    insert_step_memory_sample,
    sqlite_database,
)

from traceml_ai.renderers.shared.freshness import RankLiveness
from traceml_ai.renderers.step_memory.cli_compute import (
    StepMemoryCLIComputer,
)
from traceml_ai.renderers.step_memory.computer import (
    StepMemoryMetricsComputer,
)
from traceml_ai.renderers.step_memory.renderer import StepMemoryRenderer

GIB = 1024.0**3
T0 = 1_700_000_000.0
STEPS = 60


def _heartbeat(conn, *, rank: int, tick: int, world: int) -> None:
    """One 2 s process-sampler tick from ``rank``."""
    ts = T0 + tick * 2.0
    insert_process_sample(
        conn,
        row_id=int(ts * 1e9),
        rank=rank,
        ts=ts,
        seq=tick,
        gpu_available=True,
        gpu_count=world,
        world_size=world,
        local_world_size=world,
        gpu_mem_total_bytes=16.0 * GIB,
    )


def _write_run(
    path: str, *, last_heartbeat: Mapping[int, int], steps: int = STEPS
) -> None:
    """Two GPU ranks with ``steps`` aligned steps and per-rank heartbeats.

    ``last_heartbeat`` maps rank -> the last 2 s sampler tick it sent.
    """
    ranks = sorted(last_heartbeat)
    with sqlite_database(path, init_summary_schema) as conn:
        for rank in ranks:
            for tick in range(1, last_heartbeat[rank] + 1):
                _heartbeat(conn, rank=rank, tick=tick, world=len(ranks))
        row_id = 0
        for rank in ranks:
            for step in range(steps):
                row_id += 1
                insert_step_memory_sample(
                    conn,
                    row_id=row_id,
                    rank=rank,
                    step=step,
                    alloc=2.0 * GIB,
                    reserved=3.0 * GIB,
                    world_size=len(ranks),
                    local_world_size=len(ranks),
                )


def _render(renderable) -> str:
    console = Console(
        force_terminal=True, color_system=None, width=140, record=True
    )
    console.print(renderable)
    return console.export_text()


def _panel_text(db_path: str) -> str:
    return _render(StepMemoryRenderer(db_path).get_panel_renderable())


def test_combined_result_names_the_rank_that_stopped(tmp_path) -> None:
    db_path = str(tmp_path / "dead.db")
    _write_run(db_path, last_heartbeat={0: 60, 1: 20})

    out = StepMemoryMetricsComputer(db_path).compute_cli()

    by_rank = {r.global_rank: r for r in out.rank_liveness}
    assert sorted(by_rank) == [0, 1]
    assert by_rank[0].freshness == "fresh"
    assert by_rank[1].freshness == "stale"
    # Tick 20 arrived at T0+40 s, the newest heartbeat at T0+120 s.
    assert by_rank[1].age_s == pytest.approx(80.0)
    # The held figures are still the real ones.
    assert out.metrics
    assert out.metrics[0].summary.worst_peak == pytest.approx(2.0 * GIB)


def test_dashboard_result_carries_the_same_liveness(tmp_path) -> None:
    db_path = str(tmp_path / "dead.db")
    _write_run(db_path, last_heartbeat={0: 60, 1: 20})

    computer = StepMemoryMetricsComputer(db_path)
    assert computer.compute_dashboard().rank_liveness == (
        computer.compute_cli().rank_liveness
    )


def test_step_memory_panel_marks_the_stale_rank(tmp_path) -> None:
    db_path = str(tmp_path / "dead.db")
    _write_run(db_path, last_heartbeat={0: 60, 1: 20})

    text = _panel_text(db_path)

    assert "rank 1: no data for 80s (stale)" in text
    assert "rank 0:" not in text


def test_step_memory_panel_has_no_marker_when_every_rank_reports(
    tmp_path,
) -> None:
    db_path = str(tmp_path / "alive.db")
    _write_run(db_path, last_heartbeat={0: 60, 1: 60})

    text = _panel_text(db_path)

    assert "(stale)" not in text
    assert "no data for" not in text


def test_held_metrics_carry_this_ticks_liveness(tmp_path) -> None:
    """A tick with no complete step reuses the figures, not the verdict.

    Otherwise the rank that stopped would stay unnamed for as long as the
    last good figures are being held.
    """
    db_path = str(tmp_path / "dead.db")
    _write_run(db_path, last_heartbeat={0: 60, 1: 60})
    computer = StepMemoryCLIComputer(db_path)
    first = computer.compute()
    assert first.metrics and not any(r.is_stale for r in first.rank_liveness)

    now = (RankLiveness(global_rank=1, age_s=30.0, freshness="stale"),)
    held = computer._return_stale_or_empty(
        "STALE (no metrics this tick)", rank_liveness=now
    )
    assert held.metrics == first.metrics
    assert held.rank_liveness == now


def test_renderer_holds_its_figures_with_this_ticks_verdict(
    tmp_path, monkeypatch
) -> None:
    """The renderer's own cache holds the figures, never the verdict.

    Tick 1 has figures and every rank reporting. By tick 2 no step is
    complete, the computer's held figures have expired, and rank 1 has
    stopped: the renderer's cached figures must name it.
    """
    import traceml_ai.renderers.step_memory.cli_compute as step_memory_cli

    db_path = str(tmp_path / "dies_later.db")
    _write_run(db_path, last_heartbeat={0: 60, 1: 60})
    clock = [T0]
    monkeypatch.setattr(
        step_memory_cli, "time", SimpleNamespace(time=lambda: clock[0])
    )
    renderer = StepMemoryRenderer(db_path)
    first = _render(renderer.get_panel_renderable())
    assert "Peak Allocated" in first
    assert "(stale)" not in first

    with sqlite_database(db_path) as conn:
        conn.execute("DELETE FROM step_memory_samples")
        for tick in range(61, 101):
            _heartbeat(conn, rank=0, tick=tick, world=2)
    clock[0] += 31.0
    held = _render(renderer.get_panel_renderable())

    assert "Peak Allocated" in held
    assert "rank 1: no data for 80s (stale)" in held


def test_empty_panel_names_a_rank_that_stopped_before_the_first_step(
    tmp_path,
) -> None:
    """A rank dies before any step completes: the DDP startup hang."""
    db_path = str(tmp_path / "startup_hang.db")
    _write_run(db_path, last_heartbeat={0: 60, 1: 20}, steps=0)

    text = _panel_text(db_path)

    assert "Peak Allocated" not in text
    assert "No complete memory metrics available" in text
    assert "rank 1: no data for 80s (stale)" in text


def _add_unparseable_rank(path: str) -> None:
    """An old heartbeat row for rank 0 whose global_rank is not a number."""
    with sqlite_database(path) as conn:
        insert_process_sample(
            conn,
            row_id=10**12,
            rank=0,
            global_rank="not-a-rank",
            ts=T0 + 2.0,
            seq=1,
            gpu_available=True,
            gpu_count=2,
        )


def test_an_older_row_with_a_garbage_global_rank_moves_no_verdict(
    tmp_path,
) -> None:
    """One bad heartbeat row never costs every verdict.

    It falls back to its ``rank`` cell, rank 0, and arrived before rank
    0's newest row, so rank 0's last word stands.
    """
    db_path = str(tmp_path / "bad_rank.db")
    _write_run(db_path, last_heartbeat={0: 60, 1: 20})
    _add_unparseable_rank(db_path)

    out = StepMemoryCLIComputer(db_path).compute()
    text = _panel_text(db_path)

    assert out.metrics
    assert [(r.global_rank, r.freshness) for r in out.rank_liveness] == [
        (0, "fresh"),
        (1, "stale"),
    ]
    assert "rank 1: no data for 80s (stale)" in text
    assert "Peak" in text or "peak" in text


def test_unreadable_heartbeat_on_an_empty_tick_keeps_the_last_markers(
    tmp_path,
) -> None:
    """Unreadable is not "no rank stopped": the last verdict stands.

    Tick 1 reads rank 1 as stale. Tick 2 has no complete step and cannot
    read the heartbeat at all, so the held figures keep tick 1's verdict
    instead of silently dropping the marker.
    """
    db_path = str(tmp_path / "dead.db")
    _write_run(db_path, last_heartbeat={0: 60, 1: 20})
    computer = StepMemoryCLIComputer(db_path)
    first = computer.compute()
    assert [r.global_rank for r in first.rank_liveness if r.is_stale] == [1]

    with sqlite_database(db_path) as conn:
        conn.execute("DELETE FROM step_memory_samples")
        conn.execute("DROP TABLE process_samples")

    held = computer.compute()

    assert held.metrics == first.metrics
    assert held.rank_liveness == first.rank_liveness


def test_unreadable_heartbeat_on_a_metrics_tick_keeps_the_last_markers(
    tmp_path, monkeypatch
) -> None:
    """Fresh figures do not make an unreadable heartbeat a clean one.

    Tick 1 names rank 1. Tick 2 reads the metrics but not the heartbeat,
    so the panel keeps tick 1's marker instead of dropping it.
    """
    import traceml_ai.renderers.step_memory.common as step_memory_common

    db_path = str(tmp_path / "dead.db")
    _write_run(db_path, last_heartbeat={0: 60, 1: 20})
    renderer = StepMemoryRenderer(db_path)
    console = Console(
        force_terminal=True, color_system=None, width=140, record=True
    )
    console.print(renderer.get_panel_renderable())
    renderer.get_dashboard_renderable()
    assert "rank 1: no data for 80s (stale)" in console.export_text()

    def unreadable(*_args, **_kwargs):
        raise sqlite3.OperationalError("heartbeat unreadable")

    monkeypatch.setattr(step_memory_common, "read_rank_clock", unreadable)
    console.print(renderer.get_panel_renderable())
    dashboard = renderer.get_dashboard_renderable()

    assert "rank 1: no data for 80s (stale)" in console.export_text()
    assert dashboard.metrics
    assert [r.global_rank for r in dashboard.rank_liveness if r.is_stale] == [
        1
    ]


def test_heartbeat_read_with_no_ranks_is_an_empty_verdict(tmp_path) -> None:
    """Read fine, nobody reported: an empty tuple, not "unreadable"."""
    db_path = str(tmp_path / "no_heartbeat.db")
    with sqlite_database(db_path, init_summary_schema):
        pass

    out = StepMemoryCLIComputer(db_path).compute()

    assert out.rank_liveness == ()
