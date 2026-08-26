# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Process dashboard payload: what each rank contributes, and when.

The behaviours pinned here are the ones real captures broke: a dead rank
that used to freeze the whole block, teardown rows that blanked the GPU
tiles on a finished run, and a sawtooth whose newest sample is a trough.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from traceml_ai.renderers.process.dashboard_compute import (
    IMBALANCE_OPEN_PCT,
    ProcessDashboardComputer,
)

GB = 1024**3
CORES = 48
TOTAL_RAM = 200.0 * GB
GPU_TOTAL = 16.0 * GB

_SCHEMA = """
CREATE TABLE process_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recv_ts_ns INTEGER NOT NULL,
    rank INTEGER,
    global_rank INTEGER,
    local_rank INTEGER,
    world_size INTEGER,
    local_world_size INTEGER,
    node_rank INTEGER,
    hostname TEXT,
    sample_ts_s REAL,
    seq INTEGER,
    cpu_percent REAL,
    cpu_logical_core_count INTEGER,
    ram_used_bytes REAL,
    ram_total_bytes REAL,
    gpu_available INTEGER,
    gpu_count INTEGER,
    gpu_device_index INTEGER,
    gpu_mem_used_bytes REAL,
    gpu_mem_reserved_bytes REAL,
    gpu_mem_total_bytes REAL
);
"""

BASE_TS = 1_787_000_000.0


def _write(db: Path, rows: List[Dict[str, Any]]) -> None:
    conn = sqlite3.connect(db)
    conn.executescript(_SCHEMA)
    for row in rows:
        conn.execute(
            """
            INSERT INTO process_samples (
                recv_ts_ns, rank, global_rank, node_rank, hostname,
                sample_ts_s, seq, cpu_percent, cpu_logical_core_count,
                ram_used_bytes, ram_total_bytes, gpu_available,
                gpu_device_index, gpu_mem_used_bytes,
                gpu_mem_reserved_bytes, gpu_mem_total_bytes
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                int(row["recv_s"] * 1e9),
                row["rank"],
                row["rank"],
                0,
                "host-a",
                row["ts"],
                row["seq"],
                row.get("cpu", 100.0),
                CORES,
                row.get("rss", 2.0 * GB),
                TOTAL_RAM,
                1 if row.get("gpu", True) else 0,
                row["rank"],
                row.get("alloc", 2.0 * GB),
                row.get("reserved", 5.0 * GB),
                row.get("total", GPU_TOTAL) if row.get("gpu", True) else 0.0,
            ),
        )
    conn.commit()
    conn.close()


def _rows(
    ranks: int = 4,
    ticks: int = 40,
    *,
    cadence: float = 2.0,
    per_tick: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """A healthy multi-rank run; ``per_tick`` overrides any field."""
    out = []
    for seq in range(ticks):
        for rank in range(ranks):
            row = {
                "rank": rank,
                "seq": seq,
                "ts": BASE_TS + seq * cadence,
                "recv_s": BASE_TS + seq * cadence + 0.2,
            }
            if per_tick is not None:
                row.update(per_tick(rank, seq) or {})
            out.append(row)
    return out


def _payload(db: Path) -> Dict[str, Any]:
    return ProcessDashboardComputer(str(db)).compute()


def test_a_dead_rank_dims_instead_of_stopping_the_block(
    tmp_path: Path,
) -> None:
    """The failure this block exists for must not silence it.

    The previous payload aligned every rank on ``min(latest seq)``, so one
    rank that stopped reporting pinned the whole card to that rank's last
    tick: the surface froze exactly when a reader needed it.
    """
    rows = [
        row
        for row in _rows(ranks=4, ticks=40)
        if not (row["rank"] == 3 and row["seq"] >= 25)
    ]
    db = tmp_path / "dead.db"
    _write(db, rows)

    out = _payload(db)
    roll = out["rollups"]
    assert roll["ranks_total"] == 4
    assert roll["ranks_stale"] == 1
    assert roll["ranks_reporting"] == 3
    by_rank = {rank["global_rank"]: rank for rank in roll["ranks"]}
    assert by_rank[3]["stale"] is True
    assert by_rank[3]["age_s"] > 0
    # The dead rank keeps its row and its last readings ...
    assert by_rank[3]["cpu_capacity"] is not None
    # ... and the live ranks keep their full window, not rank 3's.
    assert out["window_len"] == 40
    for rank in (0, 1, 2):
        assert by_rank[rank]["stale"] is False


def test_stale_ranks_leave_the_aggregates(tmp_path: Path) -> None:
    """A rank that stopped must not keep speaking through the tiles."""

    def per_tick(rank: int, seq: int) -> Dict[str, Any]:
        # Rank 3 dies holding far more memory than anyone else.
        if rank == 3:
            return {"reserved": 15.0 * GB, "rss": 9.0 * GB, "cpu": 4000.0}
        return {}

    rows = [
        row
        for row in _rows(ranks=4, ticks=40, per_tick=per_tick)
        if not (row["rank"] == 3 and row["seq"] >= 25)
    ]
    db = tmp_path / "stale-agg.db"
    _write(db, rows)

    roll = _payload(db)["rollups"]
    assert roll["ranks_stale"] == 1
    # The worst live rank owns the tiles, not the dead one.
    assert roll["rss"]["rank"] != 3
    assert roll["cuda"]["reserved_rank"] != 3
    assert roll["cpu_capacity"]["worst_rank"] != 3
    # And its 10 GB of held memory does not read as imbalance.
    assert roll["reserved_imbalance_pct"] == 0.0


def test_gpu_tiles_survive_teardown_rows(tmp_path: Path) -> None:
    """A finished run's last samples carry no device, and must not blank.

    torch releases the device before the sampler stops, so the newest row
    of a completed run reports no GPU. Anchoring the tiles on it showed
    `n/a` on every finished run (the same class as a gauge reading 0 %).
    """

    def per_tick(rank: int, seq: int) -> Dict[str, Any]:
        if seq >= 38:  # teardown: device gone, sampler still writing
            return {"gpu": False, "alloc": 0.0, "reserved": 0.0}
        return {}

    db = tmp_path / "teardown.db"
    _write(db, _rows(ranks=2, ticks=40, per_tick=per_tick))

    roll = _payload(db)["rollups"]
    assert roll["cuda"]["reserved"] == 5.0 * GB
    assert roll["cuda"]["reserved_total"] == GPU_TOTAL
    assert roll["cuda"]["alloc_p50"] == 2.0 * GB
    for rank in roll["ranks"]:
        assert rank["gpu_reserved"] == 5.0 * GB


def test_allocated_is_the_window_median_not_the_newest_sample(
    tmp_path: Path,
) -> None:
    """The allocator sawtooth is undersampled at 2 s; one sample is luck.

    Measured on a real bert capture: median 1.66 GB, window max 7.3 GB,
    newest sample 0.02 GB. The newest sample is what the old card showed.
    """

    def per_tick(rank: int, seq: int) -> Dict[str, Any]:
        # A steady working set with a peak every fifth step, and a trough
        # on the newest sample.
        if seq == 39:
            return {"alloc": 0.02 * GB}
        return {"alloc": (7.0 if seq % 5 == 0 else 1.6) * GB}

    db = tmp_path / "sawtooth.db"
    _write(db, _rows(ranks=1, ticks=40, per_tick=per_tick))

    roll = _payload(db)["rollups"]
    assert roll["cuda"]["alloc_p50"] == 1.6 * GB


def test_cpu_is_bounded_by_the_hosts_capacity(tmp_path: Path) -> None:
    """`cpu_percent` sums cores; only the bounded form is comparable.

    The stored 4000 % is 4000 / (100 x 48 cores) = 83 % of this host.
    """
    db = tmp_path / "cpu.db"
    _write(db, _rows(ranks=2, ticks=20, per_tick=lambda r, s: {"cpu": 4000.0}))

    roll = _payload(db)["rollups"]
    assert roll["cpu_capacity"]["p50"] == 4000.0 / (100.0 * CORES) * 100.0
    assert roll["cpu_capacity"]["p50"] < 100.0


def test_imbalance_reads_reserved_not_allocated(tmp_path: Path) -> None:
    """Allocated differs by which step phase each sampler caught.

    On a healthy run that phase noise reaches gigabytes, which is what the
    old `GPU IMBAL` tile showed. Reserved is the stable quantity, and it is
    the one the diagnosis engine's own imbalance rule uses.
    """

    def per_tick(rank: int, seq: int) -> Dict[str, Any]:
        # Wildly different allocated, identical reserved.
        return {"alloc": (1.0 + 4.0 * ((rank + seq) % 2)) * GB}

    db = tmp_path / "imbalance.db"
    _write(db, _rows(ranks=4, ticks=30, per_tick=per_tick))

    roll = _payload(db)["rollups"]
    assert roll["reserved_imbalance_pct"] == 0.0
    assert roll["rows_over"] is False


def test_rows_open_only_after_the_bar_holds(tmp_path: Path) -> None:
    """Armed, sustained, and on the engine's bar.

    Ranks reach their first allocation seconds apart, so an unarmed
    trigger reads that ramp as total imbalance and throws the rows open on
    every run's first tick.
    """

    def per_tick(rank: int, seq: int) -> Dict[str, Any]:
        if seq < 3:  # staggered CUDA init: rank 3 has nothing yet
            return {"reserved": 0.0 if rank == 3 else 5.0 * GB}
        if rank == 3:  # then it allocates far more than its peers
            return {"reserved": 12.0 * GB}
        return {}

    db = tmp_path / "trigger.db"
    _write(db, _rows(ranks=4, ticks=30, per_tick=per_tick))

    roll = _payload(db)["rollups"]
    assert roll["reserved_imbalance_pct"] > IMBALANCE_OPEN_PCT
    assert roll["rows_over"] is True

    # The same computer, replaying a run that never crosses the bar, keeps
    # the rows shut.
    calm = tmp_path / "calm.db"
    _write(calm, _rows(ranks=4, ticks=30))
    assert _payload(calm)["rollups"]["rows_over"] is False


def test_the_ramp_alone_never_opens_the_rows(tmp_path: Path) -> None:
    """Arming: a rank with nothing allocated yet is not an imbalance."""

    def per_tick(rank: int, seq: int) -> Dict[str, Any]:
        return {"reserved": 0.0 if rank == 3 else 5.0 * GB}

    db = tmp_path / "ramp.db"
    _write(db, _rows(ranks=4, ticks=30, per_tick=per_tick))

    roll = _payload(db)["rollups"]
    assert roll["reserved_imbalance_pct"] == 100.0
    assert roll["rows_over"] is False


def test_one_series_per_chart_never_both(tmp_path: Path) -> None:
    """A short run sends its window; a long run sends only the run view."""
    short = tmp_path / "short.db"
    _write(short, _rows(ranks=2, ticks=20))
    out = _payload(short)
    assert (
        out["series"]["cpu_capacity"] and not out["series"]["cpu_capacity_run"]
    )

    long_run = tmp_path / "long.db"
    _write(long_run, _rows(ranks=2, ticks=400, cadence=10.0))
    out = _payload(long_run)
    assert out["series"]["cpu_capacity_run"]
    assert out["series"]["cpu_capacity"] == []
    assert out["series"]["rss"] == []


def test_run_history_holds_its_point_budget(tmp_path: Path) -> None:
    """The payload cannot grow with the run: 120 points per rank, counted."""
    db = tmp_path / "budget.db"
    _write(db, _rows(ranks=3, ticks=2000, cadence=2.0))

    series = _payload(db)["series"]["cpu_capacity_run"]
    assert len(series) == 3
    for entry in series:
        assert 0 < len(entry["t"]) <= 120
        assert len(entry["t"]) == len(entry["avg"]) == len(entry["max"])
        assert entry["window_s"] > 0


def test_every_degraded_path_keeps_the_series_schema(
    tmp_path: Path,
) -> None:
    """An empty or unreadable database still answers with the full shape."""
    empty = tmp_path / "empty.db"
    _write(empty, [])
    keys = {"cpu_capacity", "rss", "cpu_capacity_run", "rss_run"}
    out = _payload(empty)
    assert set(out["series"]) == keys
    assert out["window_len"] == 0
    assert out["rollups"] == {}

    missing = _payload(tmp_path / "does-not-exist.db")
    assert set(missing["series"]) == keys


def test_a_cpu_only_host_reports_no_gpu(tmp_path: Path) -> None:
    """Absent is not zero: the GPU rollups stay None on a CPU-only box."""
    db = tmp_path / "cpu-only.db"
    _write(
        db,
        _rows(
            ranks=2,
            ticks=20,
            per_tick=lambda r, s: {
                "gpu": False,
                "alloc": 0.0,
                "reserved": 0.0,
            },
        ),
    )

    out = _payload(db)
    roll = out["rollups"]
    assert out["gpu_available"] is False
    assert roll["cuda"] == {
        "alloc_p50": None,
        "reserved": None,
        "reserved_total": None,
        "reserved_rank": None,
    }
    assert roll["reserved_imbalance_pct"] is None
    assert roll["rows_over"] is False
    # The host-side numbers still work.
    assert roll["cpu_capacity"]["p50"] is not None
    assert roll["rss"]["used"] is not None


def test_a_rank_dead_longer_than_the_window_keeps_its_row(
    tmp_path: Path,
) -> None:
    """Forgetting the dead rank later is still forgetting it.

    The windowed read is bounded by the observed cadence, so a rank silent
    for longer than that bound has no rows in it. Without a second read it
    dropped out of the block entirely: ranks_total fell, the stale count
    went to zero, and the header read "3 ranks" as though rank 3 had never
    existed. That is the failure this block was rebuilt to end, deferred
    by a few minutes.
    """
    rows = [
        row
        for row in _rows(ranks=4, ticks=600, cadence=2.0)
        # rank 3 stops 20 minutes before the others, far outside the
        # window the recent read looks back over
        if not (row["rank"] == 3 and row["seq"] >= 100)
    ]
    db = tmp_path / "long-dead.db"
    _write(db, rows)

    roll = _payload(db)["rollups"]
    assert roll["ranks_total"] == 4
    assert roll["ranks_stale"] == 1
    by_rank = {rank["global_rank"]: rank for rank in roll["ranks"]}
    assert by_rank[3]["stale"] is True
    assert by_rank[3]["age_s"] > 900
    # It keeps the values it last reported, so the row still says
    # something about why it died.
    assert by_rank[3]["ram_used"] is not None


def test_nothing_reporting_does_not_claim_an_exclusion(
    tmp_path: Path,
) -> None:
    """With every rank stale there is nobody to exclude them in favour of.

    The aggregates then come from stale ranks by necessity, and the header
    must not say "excluded" over numbers those very ranks produced.
    """
    # Two ranks stop together; a third goes on alone and then also stops,
    # so by the newest arrival every rank is older than the stale bound.
    rows = _rows(ranks=3, ticks=40)
    rows = [row for row in rows if not (row["rank"] < 2 and row["seq"] >= 20)]
    db = tmp_path / "all-stale.db"
    _write(db, rows)
    roll = _payload(db)["rollups"]
    # Rank 2 is the only live one, so the other two ARE excluded.
    assert roll["ranks_stale"] == 2
    assert roll["excluding_stale"] is True

    # Now nobody is live: the newest arrival belongs to a rank that is
    # itself past the bound, which cannot happen by construction, so the
    # honest case is a database whose every rank stopped at once.
    quiet = tmp_path / "quiet.db"
    _write(quiet, _rows(ranks=3, ticks=40))
    quiet_roll = _payload(quiet)["rollups"]
    assert quiet_roll["ranks_stale"] == 0
    assert quiet_roll["excluding_stale"] is False
    # The claim is only ever made when it is true.
    assert quiet_roll["excluding_stale"] == (
        quiet_roll["ranks_stale"] > 0 and quiet_roll["ranks_reporting"] > 0
    )


def test_the_payload_feeds_the_section_unchanged(tmp_path: Path) -> None:
    """The contract this PR introduces, exercised end to end.

    Both suites otherwise build their own inputs, so a key renamed on one
    side of the payload would pass every test and blank the block.
    """
    from nicegui import ui

    from traceml_ai.aggregator.display_drivers.nicegui_sections.process_section import (  # noqa: E501
        build_process_section,
        update_process_section,
    )

    db = tmp_path / "end-to-end.db"
    _write(db, _rows(ranks=4, ticks=40))
    payload = _payload(db)

    with ui.element("div"):
        panel = build_process_section()
    update_process_section(panel, payload)

    assert "%" in panel["tiles"]["cpu"].content
    assert "GB" in panel["tiles"]["rss"].content
    assert "GB" in panel["tiles"]["reserved"].content
    assert panel["subs"]["reserved"].text.startswith("least-headroom rank")
    assert "4 ranks" in panel["rows_hint"].text
    assert "R3" in panel["rows_html"].content
    assert len(panel["cpu_chart"].options["series"]) == 4
    assert panel["cpu_label"].text.startswith("process cpu")
