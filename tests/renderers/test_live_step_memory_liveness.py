"""Step Memory names a rank that stopped reporting (issue #358).

The combined result aligns on the slowest rank's latest completed step, so
a dead rank freezes the panel on its last step. Liveness comes from each
rank's process-sampler heartbeat, which keeps arriving while a surviving
rank blocks in a collective, and is judged by the same rule the Process
dashboard uses.
"""

from __future__ import annotations

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


def _write_run(path: str, *, last_heartbeat: Mapping[int, int]) -> None:
    """Two GPU ranks with `STEPS` aligned steps and per-rank heartbeats.

    ``last_heartbeat`` maps rank -> the last 2 s sampler tick it sent.
    """
    ranks = sorted(last_heartbeat)
    with sqlite_database(path, init_summary_schema) as conn:
        for rank in ranks:
            for tick in range(1, last_heartbeat[rank] + 1):
                ts = T0 + tick * 2.0
                insert_process_sample(
                    conn,
                    row_id=int(ts * 1e9),
                    rank=rank,
                    ts=ts,
                    seq=tick,
                    gpu_available=True,
                    gpu_count=len(ranks),
                    world_size=len(ranks),
                    local_world_size=len(ranks),
                    gpu_mem_total_bytes=16.0 * GIB,
                )
        row_id = 0
        for rank in ranks:
            for step in range(STEPS):
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


def _panel_text(db_path: str) -> str:
    console = Console(
        force_terminal=True, color_system=None, width=140, record=True
    )
    console.print(StepMemoryRenderer(db_path).get_panel_renderable())
    return console.export_text()


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
