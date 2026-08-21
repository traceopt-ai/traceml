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
    limit: Optional[float] = LIMIT,
) -> Dict[str, Any]:
    return {
        "gpu_idx": idx,
        "util": util,
        "mem_used_bytes": mem_used,
        "mem_total_bytes": 16.1 * GB,
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


def _write(path: Path, rows_for_tick, *, gpu_available: bool = True) -> None:
    with sqlite_database(path, init_summary_schema) as conn:
        for seq in range(TICKS):
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


def _payload(path: Path) -> Dict[str, Any]:
    return SystemDashboardComputer(str(path)).compute(window_n=100)


def test_one_busy_of_four_reads_average_with_full_spread(
    tmp_path: Path,
) -> None:
    db = tmp_path / "b.db"
    _write(db, lambda seq: _one_busy_of_four())
    out = _payload(db)
    roll = out["rollups"]

    # The headline stays the across-GPU average (25) and the spread that
    # opens the per-GPU rows reads the full 100 points.
    assert roll["gpu_util"]["p50"] == 25.0
    assert roll["gpu_delta"]["p95"] == 100.0

    # Every aggregate cell is the max GPU, never a sum.
    assert roll["gpu_power"] == {
        "now": 68.0,
        "p50": 68.0,
        "limit": LIMIT,
        "floor": 33.0,  # the idle GPUs' draw, the run's lowest
    }
    assert roll["gpu_mem"]["now"] == 6.67 * GB
    assert roll["gpu_mem"]["total"] == 16.1 * GB
    assert roll["temp"]["now"] == 54.0

    gpus = roll["gpus"]
    assert [g["gpu_idx"] for g in gpus] == [0, 1, 2, 3]
    assert gpus[0] == {
        "gpu_idx": 0,
        "util_now": 100.0,
        "util_p50": 100.0,
        "mem_used": 6.67 * GB,
        "mem_total": 16.1 * GB,
        "temp": 54.0,
        "power": 68.0,
        "power_limit": LIMIT,
    }
    assert gpus[3]["util_p50"] == 0.0
    assert gpus[3]["power"] == 33.0

    power = out["series"]["gpu_power"]
    assert [p["gpu_idx"] for p in power] == [0, 1, 2, 3]
    assert power[0]["values"] == [68.0] * TICKS
    assert power[1]["values"] == [33.0] * TICKS
    assert len(out["series"]["x_time"]) == TICKS


def test_all_busy_reads_zero_spread(tmp_path: Path) -> None:
    db = tmp_path / "a.db"
    _write(db, lambda seq: _all_busy())
    roll = _payload(db)["rollups"]
    assert roll["gpu_util"]["p50"] == 100.0
    assert roll["gpu_delta"]["p95"] == 0.0
    assert roll["gpu_power"]["now"] == 69.0  # max GPU (gpu3), not 270
    assert roll["temp"]["now"] == 56.0


def test_missing_gpu_row_is_a_gap_not_a_zero(tmp_path: Path) -> None:
    db = tmp_path / "g.db"

    def rows(seq: int):
        full = _all_busy()
        return full[:3] if seq == 10 else full

    _write(db, rows)
    out = _payload(db)
    gpu3 = out["series"]["gpu_power"][3]["values"]
    assert len(gpu3) == TICKS
    assert gpu3[10] is None
    assert gpu3[9] == 69.0
    # The rows still list every GPU seen in the window.
    assert [g["gpu_idx"] for g in out["rollups"]["gpus"]] == [0, 1, 2, 3]


def test_no_gpu_payload_carries_empty_gpu_lists(tmp_path: Path) -> None:
    db = tmp_path / "c.db"
    _write(db, lambda seq: (), gpu_available=False)
    out = _payload(db)
    assert out["gpu_available"] is False
    assert out["rollups"]["gpus"] == []
    assert out["rollups"]["gpu_power"] == {
        "now": None,
        "p50": None,
        "limit": None,
        "floor": None,
    }
    assert out["series"]["gpu_power"] == []
    assert out["series"]["cpu"] == [8.0] * TICKS


def test_unreported_power_stays_none(tmp_path: Path) -> None:
    db = tmp_path / "p.db"
    _write(db, lambda seq: _one_busy_of_four(power=False))
    out = _payload(db)
    assert out["rollups"]["gpu_power"] == {
        "now": None,
        "p50": None,
        "limit": None,
        "floor": None,
    }
    assert out["rollups"]["gpus"][0]["power"] is None
    assert out["series"]["gpu_power"][0]["values"] == [None] * TICKS
    # Utilisation is unaffected by a missing power column.
    assert out["rollups"]["gpu_util"]["p50"] == 25.0


def test_empty_database_payload_keeps_the_schema(tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    with sqlite_database(db, init_summary_schema):
        pass
    out = _payload(db)
    assert out["window_len"] == 0
    assert out["series"]["gpu_power"] == []


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
    roll = out["rollups"]
    assert roll["ctx"]["system_node"] == {
        "hostname": "node-a",
        "node_rank": 0,
        "nodes_in_window": 2,
    }
    # Node 0's own window, not a zig-zag between the two hosts.
    assert set(out["series"]["cpu"]) == {10.0}
    assert roll["cpu"]["p50"] == 10.0
    assert roll["gpu_util"]["p50"] == 100.0
    assert [g["gpu_idx"] for g in roll["gpus"]] == [0]
    assert roll["gpus"][0]["power"] == 66.0
    assert out["series"]["gpu_power"][0]["values"] == [66.0] * TICKS
    assert len(out["series"]["x_time"]) == TICKS


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
    g1 = out["rollups"]["gpus"][1]
    assert g1["util_now"] is None
    assert g1["temp"] is None
    assert g1["power"] == 67.0  # still reported
    assert g1["util_p50"] == 100.0  # the median ignores the unreported ticks


def test_sampler_zero_fallback_tick_is_unreported(tmp_path: Path) -> None:
    """The sampler writes all-zero GPU rows when NVML fails on a device;
    those zeros are not a 0 W limit, 0 degrees or 0 GB."""
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
    roll = out["rollups"]
    # The limit is a constant seen earlier in the window: it stays.
    assert roll["gpu_power"]["limit"] == LIMIT
    assert roll["gpu_power"]["now"] is None
    for g in roll["gpus"]:
        assert g["power"] is None and g["power_limit"] is None
        assert g["mem_total"] is None and g["temp"] is None
        assert g["util_p50"] == 100.0
    assert out["series"]["gpu_power"][0]["values"][-1] is None
    assert out["series"]["gpu_power"][0]["values"][-2] == 66.0


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
    assert out["window_len"] == TICKS
    assert out["rollups"]["ctx"]["system_node"] == {
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

    run = out["series"]["cpu_run"]
    assert 2 < len(run["t"]) <= 181  # decimated, not one point per sample
    assert len(run["avg"]) == len(run["t"]) == len(run["max"])
    assert run["span_s"] == pytest.approx(2.0 * (ticks - 1))
    # The drift survives. Measured against the run's floor, not slice 0:
    # the spike at tick 0 lands inside the first slice and lifts its mean.
    assert run["avg"][-1] > min(run["avg"]) + 20
    assert max(run["max"]) >= 95.0  # and so do the spikes
    # The rolling mean never reaches the spike: it is smoothed away
    # there and preserved in "max", which is the point of carrying both.
    assert max(run["avg"]) < 95.0

    power = out["series"]["gpu_power_run"]
    assert [p["gpu_idx"] for p in power] == [0]
    e = power[0]
    assert 2 < len(e["t"]) <= 181
    assert max(e["max"]) == pytest.approx(100.0)  # the sawtooth's peak
    assert 60.0 < max(e["avg"]) < 100.0  # its mean sits between the levels
    assert e["span_s"] == pytest.approx(2.0 * (ticks - 1))


def test_whole_run_series_follow_the_node_scope(tmp_path: Path) -> None:
    """A two-host window shows one node, and its whole-run view must too."""
    db = tmp_path / "n.db"
    with sqlite_database(db, init_summary_schema) as conn:
        row_id = 0
        for seq in range(60):
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
    assert out["rollups"]["ctx"]["system_node"]["hostname"] == "node-a"
    # node-a alone: never blended with node-b's 90%
    assert max(out["series"]["cpu_run"]["max"]) == pytest.approx(10.0)
