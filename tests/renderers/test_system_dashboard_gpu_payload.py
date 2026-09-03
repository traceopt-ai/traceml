# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Per-GPU facts in the SYSTEM dashboard payload: power and the GPU rows.

The settled System block shows the across-GPU average for utilisation and
pairs it with the per-GPU rows, so the payload has to carry what the rows
need: each GPU's own utilisation, memory, temperature, power and limit, plus
a power history per GPU for the power chart. Every aggregate cell is the
max GPU, never a sum, and a GPU that did not report in a tick is a gap in
its series, not a zero.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from tests.sqlite_fixtures import (
    init_summary_schema,
    insert_system_sample,
    sqlite_database,
)
from traceml_ai.renderers.system.dashboard_compute import (
    SystemDashboardComputer,
)
from traceml_ai.renderers.system.dashboard_models import (
    CpuRunSeries,
    GpuRow,
    PowerStat,
    SystemDashboardPayload,
)

GB = 1e9
TICKS = 20
LIMIT = 70.0


def _gpu(
    idx: int,
    util: float,
    power: Optional[float],
    *,
    temp: float,
    mem_used: float,
    mem_total: float = 16.1 * GB,
    limit: Optional[float] = LIMIT,
) -> Dict[str, Any]:
    return {
        "gpu_idx": idx,
        "util": util,
        "mem_used_bytes": mem_used,
        "mem_total_bytes": mem_total,
        "temperature_c": temp,
        "power_usage_w": power,
        "power_limit_w": limit,
    }


def _one_busy_of_four(power: bool = True) -> List[Dict[str, Any]]:
    busy = 68.0 if power else None
    idle = 33.0 if power else None
    limit = LIMIT if power else None
    return [
        _gpu(0, 100.0, busy, temp=54.0, mem_used=6.67 * GB, limit=limit),
        _gpu(1, 0.0, idle, temp=41.0, mem_used=0.47 * GB, limit=limit),
        _gpu(2, 0.0, idle, temp=40.0, mem_used=0.47 * GB, limit=limit),
        _gpu(3, 0.0, idle, temp=39.0, mem_used=0.47 * GB, limit=limit),
    ]


def _all_busy() -> List[Dict[str, Any]]:
    return [
        _gpu(i, 100.0, 66.0 + i, temp=53.0 + i, mem_used=6.31 * GB)
        for i in range(4)
    ]


def _write(
    path: Path,
    rows_for_tick,
    *,
    gpu_available: bool = True,
    ticks: int = TICKS,
) -> None:
    with sqlite_database(path, init_summary_schema) as conn:
        for seq in range(ticks):
            insert_system_sample(
                conn,
                row_id=seq + 1,
                rank=0,
                ts=1000.0 + 2.0 * seq,
                gpu_available=gpu_available,
                gpu_count=4 if gpu_available else 0,
                seq=seq,
                cpu_percent=8.0,
                ram_used_bytes=9.0 * GB,
                ram_total_bytes=200.0 * GB,
                gpu_samples=rows_for_tick(seq) if gpu_available else (),
            )


def _payload(path: Path) -> SystemDashboardPayload:
    """The payload the card is handed.

    Read through the types, not through a mapping: a field that is absent
    is a construction error here rather than a missing key discovered at
    render time, which is the whole point of the payload being typed.
    """
    return SystemDashboardComputer(str(path)).compute(window_n=100)


def test_one_busy_of_four_reads_average_with_full_spread(
    tmp_path: Path,
) -> None:
    db = tmp_path / "b.db"
    _write(db, lambda seq: _one_busy_of_four())
    out = _payload(db)
    roll = out.rollups

    # The headline stays the across-GPU average (25) and the spread that
    # opens the per-GPU rows reads the full 100 points.
    assert roll.gpu_util.p50 == 25.0
    assert roll.gpu_delta.p95 == 100.0

    # Every aggregate cell is the max GPU, never a sum.
    assert roll.gpu_power == PowerStat(
        now=68.0,
        p50=68.0,
        limit=LIMIT,
        floor=33.0,  # the idle GPUs' draw, the run's lowest
    )
    assert roll.gpu_mem.now == 6.67 * GB
    assert roll.gpu_mem.total == 16.1 * GB
    assert roll.temp.now == 54.0

    gpus = roll.gpus
    assert [g.gpu_idx for g in gpus] == [0, 1, 2, 3]
    assert gpus[0] == GpuRow(
        gpu_idx=0,
        util_now=100.0,
        util_p50=100.0,
        mem_used=6.67 * GB,
        mem_total=16.1 * GB,
        temp=54.0,
        power=68.0,
        power_limit=LIMIT,
        # Added deliberately. The card used to work out whether a device
        # had reported by testing which of its fields were None, using a
        # rule that disagreed with the computer's about a GPU carrying a
        # power limit and no memory total. The computer states it here so
        # there is one answer rather than two.
        reported=True,
    )
    assert gpus[3].util_p50 == 0.0
    assert gpus[3].power == 33.0

    power = out.series.gpu_power
    assert [p["gpu_idx"] for p in power] == [0, 1, 2, 3]
    assert power[0]["values"] == [68.0] * TICKS
    assert power[1]["values"] == [33.0] * TICKS
    assert len(out.series.x_time) == TICKS
    assert out.series.cpu_run.t == ()
    assert out.series.gpu_power_run == ()


def test_all_busy_reads_zero_spread(tmp_path: Path) -> None:
    db = tmp_path / "a.db"
    _write(db, lambda seq: _all_busy())
    roll = _payload(db).rollups
    assert roll.gpu_util.p50 == 100.0
    assert roll.gpu_delta.p95 == 0.0
    assert roll.gpu_power.now == 69.0  # max GPU (gpu3), not 270
    assert roll.temp.now == 56.0


def test_missing_gpu_row_is_a_gap_not_a_zero(tmp_path: Path) -> None:
    db = tmp_path / "g.db"

    def rows(seq: int):
        full = _all_busy()
        return full[:3] if seq == 10 else full

    _write(db, rows)
    out = _payload(db)
    gpu3 = out.series.gpu_power[3]["values"]
    assert len(gpu3) == TICKS
    assert gpu3[10] is None
    assert gpu3[9] == 69.0
    # The rows still list every GPU seen in the window.
    assert [g.gpu_idx for g in out.rollups.gpus] == [0, 1, 2, 3]


def test_no_gpu_payload_carries_empty_gpu_lists(tmp_path: Path) -> None:
    db = tmp_path / "c.db"
    _write(db, lambda seq: (), gpu_available=False)
    out = _payload(db)
    assert out.gpu_available is False
    assert out.rollups.gpus == ()
    assert out.rollups.gpu_power == PowerStat()
    assert out.series.gpu_power == ()
    assert list(out.series.cpu) == [8.0] * TICKS


def test_unreported_power_stays_none(tmp_path: Path) -> None:
    db = tmp_path / "p.db"
    _write(db, lambda seq: _one_busy_of_four(power=False))
    out = _payload(db)
    assert out.rollups.gpu_power == PowerStat()
    assert out.rollups.gpus[0].power is None
    assert out.series.gpu_power[0]["values"] == [None] * TICKS
    # Utilisation is unaffected by a missing power column.
    assert out.rollups.gpu_util.p50 == 25.0


def test_empty_database_payload_keeps_the_schema(tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    with sqlite_database(db, init_summary_schema):
        pass
    out = _payload(db)
    assert out.window_len == 0
    assert out.series.gpu_power == ()
    assert out.series.cpu_run == CpuRunSeries()
    assert out.series.gpu_power_run == ()


def test_failed_first_read_keeps_the_series_schema(tmp_path: Path) -> None:
    db = tmp_path / "missing" / "run.db"
    out = _payload(db)

    assert out.window_len == 0
    assert out.rollups.status == "No fresh system data"
    assert out.series.cpu_run == CpuRunSeries()
    assert out.series.gpu_power_run == ()


def test_two_node_window_shows_the_leader_node_only(tmp_path: Path) -> None:
    """System telemetry is per machine: no pooled series, no merged gpu0."""
    db = tmp_path / "n.db"
    with sqlite_database(db, init_summary_schema) as conn:
        row_id = 0
        for seq in range(TICKS):
            for node, (host, cpu, util, power) in enumerate(
                (("node-a", 10.0, 100.0, 66.0), ("node-b", 90.0, 0.0, 33.0))
            ):
                row_id += 1
                insert_system_sample(
                    conn,
                    row_id=row_id,
                    rank=node,
                    ts=1000.0 + 2.0 * seq + 0.3 * node,
                    gpu_available=True,
                    gpu_count=1,
                    global_rank=node,
                    node_rank=node,
                    hostname=host,
                    seq=seq,
                    cpu_percent=cpu,
                    ram_used_bytes=2.0 * GB,
                    ram_total_bytes=16.0 * GB,
                    gpu_samples=[
                        _gpu(0, util, power, temp=45.0, mem_used=6.3 * GB)
                    ],
                )
    out = _payload(db)
    roll = out.rollups
    assert roll.ctx.system_node == {
        "hostname": "node-a",
        "node_rank": 0,
        "nodes_in_window": 2,
    }
    # Node 0's own window, not a zig-zag between the two hosts.
    assert set(out.series.cpu) == {10.0}
    assert roll.cpu.p50 == 10.0
    assert roll.gpu_util.p50 == 100.0
    assert [g.gpu_idx for g in roll.gpus] == [0]
    assert roll.gpus[0].power == 66.0
    assert out.series.gpu_power[0]["values"] == [66.0] * TICKS
    assert len(out.series.x_time) == TICKS


def test_null_util_and_temp_stay_none_not_zero(tmp_path: Path) -> None:
    """A NULL column is unreported: no phantom 0 in the rows, no phantom
    spread opening them."""
    db = tmp_path / "u.db"

    def rows(seq: int):
        full = _all_busy()
        if seq >= TICKS - 6:
            full[1]["util"] = None
            full[1]["temperature_c"] = None
        return full

    _write(db, rows)
    out = _payload(db)
    g1 = out.rollups.gpus[1]
    assert g1.util_now is None
    assert g1.temp is None
    assert g1.power == 67.0  # still reported
    assert g1.util_p50 == 100.0  # the median ignores the unreported ticks


def test_legacy_zero_placeholder_tick_is_unreported(tmp_path: Path) -> None:
    """Older all-zero placeholders are not 0 W, 0 degrees, or 0 GB."""
    db = tmp_path / "z.db"

    def rows(seq: int):
        if seq == TICKS - 1:
            return [
                {
                    "gpu_idx": i,
                    "util": 0.0,
                    "mem_used_bytes": 0.0,
                    "mem_total_bytes": 0.0,
                    "temperature_c": 0.0,
                    "power_usage_w": 0.0,
                    "power_limit_w": 0.0,
                }
                for i in range(4)
            ]
        return _all_busy()

    _write(db, rows)
    out = _payload(db)
    roll = out.rollups
    # The limit is a constant seen earlier in the window: it stays.
    assert roll.gpu_power.limit == LIMIT
    assert roll.gpu_power.now is None
    assert roll.gpu_power.floor == 66.0
    for g in roll.gpus:
        assert g.power is None and g.power_limit is None
        assert g.mem_total is None and g.temp is None
        assert g.util_p50 == 100.0
    assert out.series.gpu_power[0]["values"][-1] is None
    assert out.series.gpu_power[0]["values"][-2] == 66.0


def test_reported_zero_power_remains_in_whole_run_history(
    tmp_path: Path,
) -> None:
    db = tmp_path / "zero-watts.db"
    _write(
        db,
        lambda seq: [
            _gpu(0, 0.0, 0.0, temp=35.0, mem_used=0.5 * GB),
        ],
        ticks=130,
    )

    out = _payload(db)
    assert out.rollups.gpu_power.floor == 0.0
    assert set(out.series.gpu_power_run[0]["min"]) == {0.0}


def test_cpu_only_run_does_not_read_gpu_power_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "cpu-only.db"
    _write(db, lambda seq: (), gpu_available=False)
    computer = SystemDashboardComputer(str(db))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("CPU-only runs must not query GPU power history")

    # 5b renamed this read. #421 deliberately left the name alone so this
    # spy kept binding; renaming it here is the stated change, and the spy
    # now watches the FIRST GPU-power read rather than the second.
    monkeypatch.setattr(
        computer._db,
        "gpu_power_run_stats",
        fail_if_called,
    )
    out = computer.compute(window_n=100)
    assert out.series.gpu_power_run == ()


def test_empty_retained_cpu_read_falls_back_to_recent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "cpu-retained-unavailable.db"
    _write(db, lambda seq: (), gpu_available=False, ticks=130)
    computer = SystemDashboardComputer(str(db))
    monkeypatch.setattr(
        computer._db,
        "fetch_cpu_run",
        lambda *_args, **_kwargs: [],
    )

    out = computer.compute(window_n=100)

    assert out.series.cpu_run_mode == "recent"
    assert out.series.cpu_run == CpuRunSeries()
    assert len(out.series.cpu) == 100


def test_empty_retained_power_read_falls_back_to_recent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "power-retained-unavailable.db"
    _write(db, lambda seq: _all_busy(), ticks=130)
    computer = SystemDashboardComputer(str(db))
    monkeypatch.setattr(
        computer._db,
        "fetch_gpu_power_run",
        lambda *_args, **_kwargs: [],
    )

    out = computer.compute(window_n=100)

    assert out.series.power_run_mode == "recent"
    assert out.series.gpu_power_run == ()
    assert out.series.gpu_power


def test_rows_without_a_hostname_are_not_a_node(tmp_path: Path) -> None:
    """A database upgraded from a pre-0.3.0 writer has NULL hostnames on
    its oldest rows; they are not a second machine and are not dropped."""
    db = tmp_path / "h.db"
    with sqlite_database(db, init_summary_schema) as conn:
        for seq in range(TICKS):
            named = seq >= 5
            insert_system_sample(
                conn,
                row_id=seq + 1,
                rank=0,
                ts=1000.0 + 2.0 * seq,
                gpu_available=True,
                gpu_count=4,
                node_rank=0 if named else None,
                hostname="box" if named else None,
                seq=seq,
                cpu_percent=8.0,
                ram_used_bytes=9.0 * GB,
                ram_total_bytes=200.0 * GB,
                gpu_samples=_all_busy(),
            )
    out = _payload(db)
    assert out.window_len == TICKS
    assert out.rollups.ctx.system_node == {
        "hostname": "box",
        "node_rank": 0,
        "nodes_in_window": 1,
    }


def test_whole_run_series_are_decimated_and_keep_peaks(tmp_path: Path) -> None:
    """The window says what the host is doing now; these say what it did.

    Both are decimated in SQL so the payload stays a fixed size however long
    the run is, and every slice keeps its max as well as its mean, because a
    mean alone erases what matters: a CPU spike, a power peak.
    """
    db = tmp_path / "run.db"
    ticks = 600  # 20 minutes at a 2 s cadence
    with sqlite_database(db, init_summary_schema) as conn:
        for seq in range(ticks):
            cpu = 10.0 + 30.0 * (seq / ticks)  # drifts 10 -> 40%
            if seq % 50 == 0:
                cpu = 95.0  # with a periodic spike
            insert_system_sample(
                conn,
                row_id=seq + 1,
                rank=0,
                ts=1000.0 + 2.0 * seq,
                gpu_available=True,
                gpu_count=1,
                seq=seq,
                cpu_percent=cpu,
                ram_used_bytes=9.0 * GB,
                ram_total_bytes=200.0 * GB,
                gpu_samples=[
                    _gpu(
                        0,
                        100.0,
                        60.0 if seq % 2 else 100.0,  # a sawtooth
                        temp=54.0,
                        mem_used=6.3 * GB,
                    )
                ],
            )
    out = _payload(db)

    run = out.series.cpu_run
    assert 2 < len(run.t) <= 181  # decimated, not one point per sample
    assert len(run.avg) == len(run.t) == len(run.max)
    assert run.span_s == pytest.approx(2.0 * (ticks - 1))
    # The drift survives. Measured against the run's floor, not slice 0:
    # the spike at tick 0 lands inside the first slice and lifts its mean.
    assert run.avg[-1] > min(run.avg) + 20
    assert max(run.max) >= 95.0  # and so do the spikes
    # The rolling mean never reaches the spike: it is smoothed away
    # there and preserved in "max", which is the point of carrying both.
    assert max(run.avg) < 95.0

    power = out.series.gpu_power_run
    assert [p["gpu_idx"] for p in power] == [0]
    e = power[0]
    assert 2 < len(e["t"]) <= 181
    assert max(e["max"]) == pytest.approx(100.0)  # the sawtooth's peak
    assert 60.0 < max(e["avg"]) < 100.0  # its mean sits between the levels
    assert e["span_s"] == pytest.approx(2.0 * (ticks - 1))


def test_cpu_whole_run_series_honors_point_cap(tmp_path: Path) -> None:
    """A run inside the old floor-division gap never exceeds 120 points."""
    db = tmp_path / "cpu-cap.db"
    ticks = 200
    with sqlite_database(db, init_summary_schema) as conn:
        for seq in range(ticks):
            insert_system_sample(
                conn,
                row_id=seq + 1,
                rank=0,
                ts=1000.0 + 2.0 * seq,
                gpu_available=False,
                gpu_count=0,
                seq=seq,
                cpu_percent=float(seq % 100),
                ram_used_bytes=9.0 * GB,
                ram_total_bytes=200.0 * GB,
                gpu_samples=(),
            )

    run = _payload(db).series.cpu_run
    assert 2 < len(run.t) <= 120
    assert len(run.t) == len(run.avg) == len(run.max)


def test_cpu_whole_run_keeps_all_eligible_points_under_cap(
    tmp_path: Path,
) -> None:
    """Do not over-decimate valid samples after the initial rolling window."""
    db = tmp_path / "cpu-near-cap.db"
    ticks = 130
    with sqlite_database(db, init_summary_schema) as conn:
        for seq in range(ticks):
            insert_system_sample(
                conn,
                row_id=seq + 1,
                rank=0,
                ts=1000.0 + 2.0 * seq,
                gpu_available=False,
                gpu_count=0,
                seq=seq,
                cpu_percent=float(seq % 100),
                ram_used_bytes=9.0 * GB,
                ram_total_bytes=200.0 * GB,
                gpu_samples=(),
            )

    run = _payload(db).series.cpu_run
    # A 30-second rolling window at a 2-second cadence excludes the first
    # 14 samples. The remaining 116 fit under the cap and should all survive.
    assert len(run.t) == 116
    assert len(run.t) == len(run.avg) == len(run.max)


def test_whole_run_series_follow_the_node_scope(tmp_path: Path) -> None:
    """A two-host window shows one node, and its whole-run view must too."""
    db = tmp_path / "n.db"
    with sqlite_database(db, init_summary_schema) as conn:
        row_id = 0
        # Longer than the recent 100-sample window, so whole-run mode is
        # active and its node scoping is exercised rather than bypassed.
        for seq in range(160):
            for node, (host, cpu) in enumerate(
                (("node-a", 10.0), ("node-b", 90.0))
            ):
                row_id += 1
                insert_system_sample(
                    conn,
                    row_id=row_id,
                    rank=node,
                    ts=1000.0 + 2.0 * seq + 0.3 * node,
                    gpu_available=True,
                    gpu_count=1,
                    global_rank=node,
                    node_rank=node,
                    hostname=host,
                    seq=seq,
                    cpu_percent=cpu,
                    ram_used_bytes=2.0 * GB,
                    ram_total_bytes=16.0 * GB,
                    gpu_samples=[
                        _gpu(0, 100.0, 66.0, temp=45.0, mem_used=6.3 * GB)
                    ],
                )
    out = _payload(db)
    assert out.rollups.ctx.system_node["hostname"] == "node-a"
    # node-a alone: never blended with node-b's 90%
    assert max(out.series.cpu_run.max) == pytest.approx(10.0)


# --- an unreported GPU is not a measurement of zero -----------------------
def test_an_unreported_gpu_does_not_drag_the_average_down(tmp_path: Path):
    """Four GPUs pinned at 100%, one of them not reporting utilisation.

    The aggregates used to substitute 0.0 for a missing reading, so the
    headline tile read 75% on a fully busy host and the across-GPU spread
    read 100 points where there was none. The spread is what opens the
    per-GPU rows, so the card announced an imbalance and then printed
    "util 100 to 100%" directly beside it, because that text is built on
    the rollup path which already excluded the device.

    Ten lines below the defect, the same loop builds the per-GPU histories
    correctly: it asks `_gpu_reported` and stores None for a gap. This
    makes the aggregates ask the same question.
    """
    db = tmp_path / "null-util.db"

    def rows(seq: int):
        out = [
            _gpu(i, 100.0, 66.0, temp=45.0, mem_used=6.3 * GB)
            for i in range(4)
        ]
        out[1]["util"] = None  # reporting device, absent metric
        return out

    _write(db, rows)
    roll = _payload(db).rollups

    assert roll.gpu_util.p50 == 100.0
    assert roll.gpu_delta.p95 == 0.0
    assert roll.rows_over is False
    # The count is the MEAN's denominator, not the number of devices that
    # reported. gpu1 reported, so it is live, but its util is absent, so
    # the mean covers three. A label saying "avg of 4 GPUs" over three
    # readings is the same defect this test exists to prevent, and every
    # assertion above passes with that label.
    assert roll.util_gpu_count == 3


def test_a_legacy_zero_placeholder_is_not_an_idle_gpu(tmp_path: Path):
    """An older all-zero placeholder remains distinguishable from idle.

    In traces recorded before sampling failures became NULL, the stored
    values are real 0.0 values. Only ``reported`` separates such a row
    from a genuinely idle device, and the payload must carry that fact to
    the aggregates.
    """
    db = tmp_path / "nvml.db"

    def rows(seq: int):
        out = [
            _gpu(i, 100.0, 66.0, temp=45.0, mem_used=6.3 * GB)
            for i in range(4)
        ]
        out[1] = {
            "gpu_idx": 1,
            "util": 0.0,
            "mem_used_bytes": 0.0,
            "mem_total_bytes": 0.0,
            "temperature_c": 0.0,
            "power_usage_w": 0.0,
            "power_limit_w": 0.0,
        }
        return out

    _write(db, rows)
    out = _payload(db)
    roll = out.rollups

    assert [g.gpu_idx for g in roll.gpus if not g.reported] == [1]
    assert roll.gpu_util.p50 == 100.0
    assert roll.gpu_delta.p95 == 0.0
    assert roll.rows_over is False
    # The tile must not claim to average devices it did not read.
    assert roll.util_gpu_count == 3


def test_all_reporting_leaves_the_count_at_the_device_count(tmp_path: Path):
    db = tmp_path / "healthy.db"
    _write(
        db,
        lambda seq: [
            _gpu(i, 100.0, 66.0, temp=45.0, mem_used=6.3 * GB)
            for i in range(4)
        ],
    )
    roll = _payload(db).rollups
    assert roll.util_gpu_count == 4
    assert roll.gpu_util.p50 == 100.0


def test_a_host_where_no_gpu_reports_has_no_average_to_show(tmp_path: Path):
    """Trigger (b) at full scale, which had no coverage at all.

    This fixture uses legacy all-zero placeholders for a host-wide failure.
    The mean then covers nothing. A tile reading 0% would be a measurement
    that no device made, so the payload states a count of zero and the card
    reads n/a rather than inventing an idle host.
    """
    db = tmp_path / "all-failed.db"
    zero = {
        "gpu_idx": 0,
        "util": 0.0,
        "mem_used_bytes": 0.0,
        "mem_total_bytes": 0.0,
        "temperature_c": 0.0,
        "power_usage_w": 0.0,
        "power_limit_w": 0.0,
    }
    _write(db, lambda seq: [dict(zero, gpu_idx=i) for i in range(4)])
    roll = _payload(db).rollups

    assert roll.util_gpu_count == 0
    assert roll.gpus_unreported is True
    assert all(not g.reported for g in roll.gpus)


def test_memory_pairs_stay_with_their_own_device(tmp_path: Path):
    """Keep used memory paired with the reporting device's capacity."""
    db = tmp_path / "mem-pairs.db"

    def rows(seq: int):
        return [
            _gpu(0, 100.0, 66.0, temp=45.0, mem_used=6.0 * GB),
            _gpu(
                1,
                100.0,
                66.0,
                temp=45.0,
                mem_used=7.0 * GB,
                mem_total=8.0 * GB,
            ),
        ]

    _write(db, rows)
    rollups = _payload(db).rollups
    entry = next(row for row in rollups.gpus if row.gpu_idx == 1)
    assert (entry.mem_used, entry.mem_total) == (7.0 * GB, 8.0 * GB)

    mem = rollups.gpu_mem
    assert mem is not None
    assert mem.now == 7.0 * GB
    assert mem.total == 8.0 * GB
    assert mem.headroom == 1.0 * GB


def test_a_gpu_that_goes_quiet_midway_only_leaves_its_own_ticks(tmp_path):
    """Mixed reporting across ticks: every tick averages its own reporters.

    Each of the ten earlier tests holds all ticks identical, so nothing
    covered a device that reports for part of a window and not the rest.
    """
    db = tmp_path / "midway.db"
    zero = {
        "gpu_idx": 1,
        "util": 0.0,
        "mem_used_bytes": 0.0,
        "mem_total_bytes": 0.0,
        "temperature_c": 0.0,
        "power_usage_w": 0.0,
        "power_limit_w": 0.0,
    }

    def rows(seq: int):
        out = [
            _gpu(i, 100.0, 66.0, temp=45.0, mem_used=6.3 * GB)
            for i in range(4)
        ]
        if seq >= TICKS // 2:
            out[1] = dict(zero)
        return out

    _write(db, rows)
    roll = _payload(db).rollups
    # The mean never dips: the ticks gpu1 missed averaged the other three.
    assert roll.gpu_util.p50 == 100.0
    assert roll.gpu_delta.p95 == 0.0
    # The newest tick is one of the ones it missed.
    assert roll.util_gpu_count == 3


# --- a tick where nothing reported is a gap, not a zero (#430) -----------
def test_a_tick_where_no_gpu_reported_is_a_gap_in_the_chart(tmp_path: Path):
    """A transient host-wide NVML failure is an absence, not 0% util.

    #432 stopped a single unreported DEVICE being averaged in as a real
    zero. This is the same idea one level up: a tick in which NO device
    reported left its slot at the pre-filled 0.0, so the chart drew a
    utilisation point, a memory point and a temperature point for a
    moment nothing was measured.

    The correct convention already exists four lines away in the same
    loop, where the per-GPU power history stores None for a gap.
    """
    db = tmp_path / "blind-tick.db"
    zero = {
        "gpu_idx": 0,
        "util": 0.0,
        "mem_used_bytes": 0.0,
        "mem_total_bytes": 0.0,
        "temperature_c": 0.0,
        "power_usage_w": 0.0,
        "power_limit_w": 0.0,
    }

    def rows(seq: int):
        if seq == TICKS // 2:
            return [dict(zero, gpu_idx=i) for i in range(4)]
        return [
            _gpu(i, 100.0, 66.0, temp=45.0, mem_used=6.3 * GB)
            for i in range(4)
        ]

    _write(db, rows)
    out = _payload(db)

    series = list(out.series.gpu_avg)
    assert series[TICKS // 2] is None
    assert all(v == 100.0 for i, v in enumerate(series) if i != TICKS // 2)


def test_a_blind_tick_does_not_drag_the_window_statistics(tmp_path: Path):
    """The tile summarises what was measured, not what was not.

    A zero in the array is not only drawn, it is also fed to the window
    percentiles, so repeated blind ticks pull the headline number down on
    a host that never left 100%.
    """
    db = tmp_path / "blind-many.db"
    zero = {
        "gpu_idx": 0,
        "util": 0.0,
        "mem_used_bytes": 0.0,
        "mem_total_bytes": 0.0,
        "temperature_c": 0.0,
        "power_usage_w": 0.0,
        "power_limit_w": 0.0,
    }

    def rows(seq: int):
        # Half the window blind, so the median moves if the blind ticks
        # are counted: ten zeros and ten hundreds median to 50.
        if seq % 2 == 0:
            return [dict(zero, gpu_idx=i) for i in range(4)]
        return [
            _gpu(i, 100.0, 66.0, temp=45.0, mem_used=6.3 * GB)
            for i in range(4)
        ]

    _write(db, rows)
    roll = _payload(db).rollups
    assert roll.gpu_util.p50 == 100.0
    assert roll.gpu_delta.p95 == 0.0
    assert roll.temp.p95 == 45.0


def test_missing_host_readings_are_not_counted_as_zero(tmp_path: Path):
    """Missing CPU and RAM readings stay absent from host rollups.

    `cpu_percent` is nullable, and the window was built with
    `float(x or 0.0)`, so a missing reading entered the percentile as a
    measured 0%. A host sitting at a steady 80% with half its rows
    lacking the reading showed 40% on the tile: the exact arithmetic of
    counting absences as zeros. RAM follows the same rule.
    """
    import sqlite3

    from tests.sqlite_fixtures import init_summary_schema, insert_system_sample
    from traceml_ai.renderers.system.dashboard_compute import (
        SystemDashboardComputer,
    )

    db = tmp_path / "null-cpu.db"
    conn = sqlite3.connect(db)
    init_summary_schema(conn)
    for seq in range(20):
        insert_system_sample(
            conn,
            row_id=seq + 1,
            rank=0,
            ts=1000.0 + 2.0 * seq,
            gpu_available=False,
            gpu_count=0,
            seq=seq,
            cpu_percent=80.0,
            ram_used_bytes=8.0 * GB,
            ram_total_bytes=16.0 * GB,
        )
    conn.commit()
    conn.execute(
        "UPDATE system_samples SET cpu_percent = NULL WHERE seq % 2 = 0"
    )
    # Leave the newest RAM reading absent to cover both the percentile and
    # the newest-measured value used to derive headroom.
    conn.execute(
        "UPDATE system_samples SET ram_used_bytes = NULL WHERE seq % 2 = 1"
    )
    conn.commit()
    conn.close()

    out = SystemDashboardComputer(str(db)).compute(window_n=100)
    assert out.rollups.cpu.p50 == 80.0
    assert out.rollups.cpu.p95 == 80.0
    # And the chart shows the gaps rather than drawing them at zero.
    assert list(out.series.cpu).count(None) == 10
    assert out.rollups.ram is not None
    assert out.rollups.ram.now == 8.0 * GB
    assert out.rollups.ram.p95 == 8.0 * GB
    assert out.rollups.ram.headroom == 8.0 * GB


def test_memory_and_its_capacity_come_from_the_same_tick(tmp_path: Path):
    """A level and the capacity it is measured against must be one moment.

    Skipping backwards past a blind tick for the value while leaving the
    capacity at the newest tick pairs two different moments, and when the
    newest tick is blind the capacity is zero, so the tile silently drops
    its denominator and renders a bare number.
    """
    db = tmp_path / "paired.db"
    zero = {
        "gpu_idx": 0,
        "util": 0.0,
        "mem_used_bytes": 0.0,
        "mem_total_bytes": 0.0,
        "temperature_c": 0.0,
        "power_usage_w": 0.0,
        "power_limit_w": 0.0,
    }

    def rows(seq: int):
        if seq == TICKS - 1:  # the NEWEST tick is blind
            return [dict(zero, gpu_idx=i) for i in range(4)]
        return [
            _gpu(i, 100.0, 66.0, temp=45.0, mem_used=6.3 * GB)
            for i in range(4)
        ]

    _write(db, rows)
    mem = _payload(db).rollups.gpu_mem
    assert mem is not None
    assert mem.now == 6.3 * GB
    assert mem.total == 16.1 * GB


def test_a_tick_with_no_gpu_rows_at_all_is_also_a_gap(tmp_path: Path):
    """The branch the all-zero fixtures do not reach.

    A row that is present and unreadable takes the reported-ness filter;
    a tick with NO rows takes a different branch entirely. Both must leave
    the slot absent, and only the first was covered.
    """
    db = tmp_path / "norows.db"

    def rows(seq: int):
        if seq == TICKS // 2:
            return ()
        return [
            _gpu(i, 100.0, 66.0, temp=45.0, mem_used=6.3 * GB)
            for i in range(4)
        ]

    _write(db, rows)
    out = _payload(db)
    assert list(out.series.gpu_avg)[TICKS // 2] is None
    assert out.rollups.gpu_util.p50 == 100.0


def _all_null_host(path: Path, *, gpu_available: bool) -> None:
    """A window in which the host reported no CPU and no RAM at all."""
    _write(path, lambda seq: (), gpu_available=gpu_available)
    with sqlite_database(path, init_summary_schema) as conn:
        conn.execute("UPDATE system_samples SET cpu_percent = NULL")
        conn.execute("UPDATE system_samples SET ram_used_bytes = NULL")
        conn.commit()


def test_a_window_with_no_host_readings_abstains(tmp_path: Path) -> None:
    """No CPU or RAM reading at all is not a host sitting idle.

    Every tick absent used to collapse to 0.0, which the card renders as a
    real measurement: the CPU label read "0%" and the RAM tile read
    "0.0 / 200 GB". Both describe a machine doing nothing, which is the
    opposite of what an unmeasured window means.
    """
    db = tmp_path / "host-absent.db"
    _all_null_host(db, gpu_available=False)

    roll = _payload(db).rollups

    assert roll.cpu is not None
    assert roll.cpu.now is None
    assert roll.cpu.p50 is None
    assert roll.cpu.p95 is None
    assert roll.ram is not None
    assert roll.ram.now is None
    assert roll.ram.p95 is None
    # Headroom is derived from a level nothing measured, so it has no value
    # either. A zero here would read as a full machine.
    assert roll.ram.headroom is None
    # The capacity is a different reading and these rows still carry it.
    # Abstaining here too would be over-reach, not a fix.
    assert roll.ram.total == 200.0 * GB


def _all_devices_failed(path: Path) -> None:
    """Every device writing the sampler's NVML-failure row.

    This is the reachable trigger: the sampler reads a device's metrics
    under one try and appends an all-zero placeholder when that raises,
    so a device reports everything or nothing. A row with capacity
    present and metrics missing does not occur.
    """

    def rows(_seq):
        return [
            {
                "gpu_idx": i,
                "util": 0.0,
                "mem_used_bytes": 0.0,
                "mem_total_bytes": 0.0,
                "temperature_c": 0.0,
                "power_usage_w": 0.0,
                "power_limit_w": 0.0,
            }
            for i in range(4)
        ]

    _write(path, rows)


def test_a_host_where_no_device_reported_abstains(tmp_path: Path) -> None:
    """Nothing was measured, so nothing is stated.

    The zeros are the sampler's failure marker, not readings, so every
    across-device aggregate has no value and the verdict derived from
    the temperature has nothing to be OK about.
    """
    db = tmp_path / "all-failed.db"
    _all_devices_failed(db)

    roll = _payload(db).rollups

    assert roll.gpus_unreported is True
    assert roll.gpu_mem is not None
    assert roll.gpu_mem.now is None
    assert roll.gpu_mem.total is None
    assert roll.gpu_mem.headroom is None
    assert roll.temp is not None
    assert roll.temp.now is None
    # A verdict computed from no temperature is not a verdict.
    assert roll.temp.status is None
    assert roll.gpu_util is not None
    assert roll.gpu_util.p50 is None


def test_a_partly_measured_window_still_reports(tmp_path: Path) -> None:
    """Abstention is for the all-absent window, not for any absence.

    The narrower behaviour is the point: one measured tick is enough for a
    percentile and for a newest-value read, and this is what separates this
    change from suppressing the whole card whenever a sample goes missing.
    """
    db = tmp_path / "partly.db"
    _all_null_host(db, gpu_available=False)
    with sqlite_database(db, init_summary_schema) as conn:
        conn.execute(
            "UPDATE system_samples SET cpu_percent = 80.0, "
            "ram_used_bytes = ? WHERE seq = ?",
            (9.0 * GB, TICKS - 1),
        )
        conn.commit()

    roll = _payload(db).rollups

    assert roll.cpu.now == 80.0
    assert roll.cpu.p50 == 80.0
    assert roll.ram.now == 9.0 * GB
    assert roll.ram.headroom == 191.0 * GB


def test_a_genuine_zero_reading_is_reported_as_zero(tmp_path: Path) -> None:
    """The same invariant on the dashboard payload.

    Absence and zero must stay distinct in BOTH directions. Every value
    is deliberately 0 here: a device really can sit at 0% drawing no
    power, and turning that into an abstention would hide a real
    measurement, which is worse than the defect being fixed.
    """
    db = tmp_path / "idle.db"

    def rows(_seq):
        return [_gpu(i, 0.0, 0.0, temp=0.0, mem_used=0.0) for i in range(4)]

    _write(db, rows)
    with sqlite_database(db, init_summary_schema) as conn:
        conn.execute("UPDATE system_samples SET cpu_percent = 0.0")
        conn.commit()

    roll = _payload(db).rollups

    assert roll.cpu.now == 0.0
    assert roll.cpu.p50 == 0.0
    assert roll.gpu_util.p50 == 0.0
    assert roll.gpu_mem.now == 0.0
    assert roll.temp.now == 0.0
    # A verdict computed from a real 0 degrees is still a verdict.
    assert roll.temp.status == "OK"
    # The capacity pairs with a measured level, so it is present.
    assert roll.gpu_mem.total == 16.1 * GB
    assert roll.util_gpu_count == 4


def test_a_held_over_payload_carries_why_it_is_held(tmp_path: Path) -> None:
    """The whole point of this change, end to end.

    A first compute succeeds and is cached. The next read fails. The
    boundary must serve the cached numbers AND say they are held over,
    because numbers without that marker are indistinguishable from live
    ones.

    Pinned here because the two ends were already covered and the middle
    was not: the first-failure case has a test, and the card has a test
    that is handed a literal status string. Dropping the status write in
    `_return_stale` would leave both of those green while restoring the
    original defect.
    """
    db = tmp_path / "held.db"
    _write(db, lambda seq: (), gpu_available=False)

    computer = SystemDashboardComputer(str(db))
    good = computer.compute(window_n=100)
    assert good.rollups.cpu is not None
    assert good.rollups.status is None
    live_value = good.rollups.cpu.p50

    with sqlite_database(db, init_summary_schema) as conn:
        conn.execute("DROP TABLE system_samples")
        conn.commit()

    held = computer.compute(window_n=100)

    # The numbers are the cached ones, not zeros and not absent.
    assert held.rollups.cpu is not None
    assert held.rollups.cpu.p50 == live_value
    # And they arrive saying why they are still here.
    assert held.rollups.status is not None
    assert "STALE" in held.rollups.status

    # And the card actually shows it, which is the other half of the
    # wiring: the boundary writing a marker nothing renders is the
    # defect this replaces, not a fix for it.
    pytest.importorskip("nicegui")
    from nicegui import ui

    from traceml_ai.aggregator.display_drivers.nicegui_sections.system_section import (  # noqa: E501
        build_system_section,
        update_system_section,
    )

    with ui.element("div"):
        panel = build_system_section()
    update_system_section(panel, held)
    assert held.rollups.status in panel["note"].text

def _with_arrival_clock(path: Path) -> None:
    """Give the rows a real arrival clock.

    The fixture writes `recv_ts_ns` as a synthetic row id, which is fine
    for ordering and useless as a clock. Liveness is measured against the
    AGGREGATOR's arrival time rather than the sampler's `sample_ts_s`,
    because the sampler may be a different machine and a skewed remote
    clock would otherwise read as a dead node. Process measures age the
    same way.
    """
    with sqlite_database(path, init_summary_schema) as conn:
        conn.execute(
            "UPDATE system_samples "
            "SET recv_ts_ns = CAST(sample_ts_s * 1000000000 AS INTEGER)"
        )
        conn.commit()


def _payload_at(path: Path, now_s: float) -> SystemDashboardPayload:
    """The payload as it would be computed at a given wall-clock moment."""
    return SystemDashboardComputer(str(path), now_fn=lambda: now_s).compute(
        window_n=100
    )


def test_a_node_that_stopped_reporting_is_marked_stale(
    tmp_path: Path,
) -> None:
    """System is single-node, so the question is about the payload.

    Process asks which of N ranks stopped, and answers per rank. There is
    only one host here, so the question is whether this payload still
    describes a live machine, and comparing the node against itself would
    always say yes. The reference is the wall clock.
    """
    db = tmp_path / "quiet.db"
    _write(db, lambda seq: (), gpu_available=False)
    _with_arrival_clock(db)

    fresh = _payload_at(db, 1000.0 + 2.0 * (TICKS - 1) + 1.0)
    assert fresh.rollups.node_liveness is not None
    assert fresh.rollups.node_liveness.state == "fresh"

    # Two minutes later nothing has arrived.
    quiet = _payload_at(db, 1000.0 + 2.0 * (TICKS - 1) + 120.0)
    assert quiet.rollups.node_liveness.state == "stale"
    assert quiet.rollups.node_liveness.age_s == pytest.approx(120.0, abs=1.0)


def test_liveness_reads_the_shared_threshold_not_a_local_one(
    tmp_path: Path,
) -> None:
    """One rule for one question, and it is the shared module's.

    This series has already produced two contradictions from answering
    one question in two places. The threshold here is
    `FreshnessPolicy.stale_after_s`, which the Process block already
    reads; nothing about the System card defines its own.
    """
    from traceml_ai.renderers.shared.freshness import FreshnessPolicy

    db = tmp_path / "threshold.db"
    _write(db, lambda seq: (), gpu_available=False)
    _with_arrival_clock(db)
    newest = 1000.0 + 2.0 * (TICKS - 1)

    # The fixture samples every 2 s, so the shared policy's floor of 5 s
    # is what applies, not the cadence multiple.
    policy = FreshnessPolicy.from_interval(2.0)
    edge = policy.stale_after_s

    assert _payload_at(
        db, newest + edge - 0.5
    ).rollups.node_liveness.state == ("fresh")
    assert _payload_at(
        db, newest + edge + 0.5
    ).rollups.node_liveness.state == ("stale")
