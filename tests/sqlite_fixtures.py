"""Shared production-schema builders for database-backed tests."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from traceml_ai.aggregator.sqlite_writers import (
    process as process_projection,
    runtime_environment as runtime_environment_projection,
    step_memory as step_memory_projection,
    step_time as step_time_projection,
    system as system_projection,
)
from traceml_ai.samplers.schema.step_time_schema import StepTimeEventSample
from traceml_ai.telemetry.envelope import TelemetryEnvelope, TelemetryMeta

_UNSET = object()


def init_summary_schema(conn: sqlite3.Connection) -> None:
    """Create every projection table consumed by final-summary sections."""
    for initialize in (
        system_projection.init_schema,
        process_projection.init_schema,
        step_time_projection.init_schema,
        step_memory_projection.init_schema,
        runtime_environment_projection.init_schema,
    ):
        initialize(conn)


def init_step_time_schema(conn: sqlite3.Connection) -> None:
    """Create only the projection tables read by the Step Time repository."""
    step_time_projection.init_schema(conn)
    runtime_environment_projection.init_schema(conn)


@contextmanager
def sqlite_database(
    path: str | Path,
    *initializers: Callable[[sqlite3.Connection], None],
) -> Iterator[sqlite3.Connection]:
    """Open, initialize, commit, and close one temporary test database."""
    conn = sqlite3.connect(path)
    try:
        for initialize in initializers:
            initialize(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


@contextmanager
def summary_database(path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open a temporary database with every summary projection initialized."""
    with sqlite_database(path, init_summary_schema) as conn:
        yield conn


def _insert_row(
    conn: sqlite3.Connection,
    table: str,
    **values: object,
) -> None:
    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    conn.execute(
        f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
        tuple(values.values()),
    )


def _defaulted(value: Any, fallback: Any) -> Any:
    return fallback if value is _UNSET else value


def insert_system_sample(
    conn: sqlite3.Connection,
    *,
    row_id: int,
    rank: int,
    ts: float,
    gpu_available: bool,
    gpu_count: int,
    gpu_util: float | None = None,
    world_size: int = 1,
    local_world_size: int = 1,
    local_rank: int | None = 0,
    node_rank: int | None | object = _UNSET,
    global_rank: int | None | object = _UNSET,
    hostname: str | None | object = _UNSET,
    seq: int | None | object = _UNSET,
    cpu_percent: float | None = None,
    ram_used_bytes: float | None = None,
    ram_total_bytes: float | None = 16_000.0,
    gpu_samples: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Insert one node sample and its optional per-GPU children."""
    resolved_node = _defaulted(node_rank, rank)
    resolved_global = _defaulted(global_rank, rank)
    resolved_hostname = _defaulted(hostname, f"worker-{rank}")
    resolved_seq = _defaulted(seq, row_id)
    _insert_row(
        conn,
        "system_samples",
        recv_ts_ns=row_id,
        global_rank=resolved_global,
        local_rank=local_rank,
        world_size=world_size,
        local_world_size=local_world_size,
        node_rank=resolved_node,
        hostname=resolved_hostname,
        sample_ts_s=ts,
        seq=resolved_seq,
        cpu_percent=30.0 + rank if cpu_percent is None else cpu_percent,
        ram_used_bytes=(
            4_000.0 + rank if ram_used_bytes is None else ram_used_bytes
        ),
        ram_total_bytes=ram_total_bytes,
        gpu_available=int(gpu_available),
        gpu_count=gpu_count,
    )
    if gpu_samples is None:
        gpu_samples = (
            ({"gpu_idx": rank, "util": gpu_util},)
            if gpu_available and gpu_count > 0
            else ()
        )
    for offset, gpu in enumerate(gpu_samples):
        _insert_row(
            conn,
            "system_gpu_samples",
            recv_ts_ns=int(gpu.get("recv_ts_ns", row_id + offset)),
            global_rank=gpu.get("global_rank", resolved_global),
            local_rank=gpu.get("local_rank", local_rank),
            world_size=gpu.get("world_size", world_size),
            local_world_size=gpu.get("local_world_size", local_world_size),
            node_rank=gpu.get("node_rank", resolved_node),
            hostname=gpu.get("hostname", resolved_hostname),
            sample_ts_s=gpu.get("sample_ts_s", ts),
            seq=gpu.get("seq", resolved_seq),
            gpu_idx=gpu.get("gpu_idx", rank),
            util=gpu.get("util", gpu_util),
            mem_used_bytes=gpu.get("mem_used_bytes", 2_500.0),
            mem_total_bytes=gpu.get("mem_total_bytes", 10_000.0),
            temperature_c=gpu.get("temperature_c"),
            power_usage_w=gpu.get("power_usage_w"),
            power_limit_w=gpu.get("power_limit_w"),
        )


def insert_process_sample(
    conn: sqlite3.Connection,
    *,
    row_id: int,
    rank: int,
    ts: float,
    gpu_available: bool,
    gpu_count: int,
    global_rank: int | None | object = _UNSET,
    local_rank: int | None = 0,
    world_size: int = 1,
    local_world_size: int = 1,
    node_rank: int | None = 0,
    hostname: str | None | object = _UNSET,
    seq: int | None | object = _UNSET,
    cpu_percent: float | None = None,
    cpu_logical_core_count: int | None = 8,
    ram_used_bytes: float | None = None,
    ram_total_bytes: float | None = 16_000.0,
    gpu_device_index: int | None | object = _UNSET,
    gpu_mem_used_bytes: float | None | object = _UNSET,
    gpu_mem_reserved_bytes: float | None | object = _UNSET,
    gpu_mem_total_bytes: float | None | object = _UNSET,
) -> None:
    """Insert one process projection row with distributed identity."""
    resolved_global = _defaulted(global_rank, rank)
    _insert_row(
        conn,
        "process_samples",
        recv_ts_ns=row_id,
        rank=rank,
        global_rank=resolved_global,
        local_rank=local_rank,
        world_size=world_size,
        local_world_size=local_world_size,
        node_rank=node_rank,
        hostname=_defaulted(hostname, f"worker-{rank}"),
        sample_ts_s=ts,
        seq=_defaulted(seq, row_id),
        cpu_percent=50.0 + rank if cpu_percent is None else cpu_percent,
        cpu_logical_core_count=cpu_logical_core_count,
        ram_used_bytes=(
            1_000.0 + rank * 100.0
            if ram_used_bytes is None
            else ram_used_bytes
        ),
        ram_total_bytes=ram_total_bytes,
        gpu_available=int(gpu_available),
        gpu_count=gpu_count,
        gpu_device_index=_defaulted(
            gpu_device_index,
            rank if gpu_available else None,
        ),
        gpu_mem_used_bytes=_defaulted(
            gpu_mem_used_bytes,
            2_000.0 + rank * 500.0 if gpu_available else None,
        ),
        gpu_mem_reserved_bytes=_defaulted(
            gpu_mem_reserved_bytes,
            2_500.0 + rank * 600.0 if gpu_available else None,
        ),
        gpu_mem_total_bytes=_defaulted(
            gpu_mem_total_bytes,
            10_000.0 if gpu_available else None,
        ),
    )


def step_time_events(
    *,
    dataloader: float,
    forward: float,
    backward: float,
    optimizer: float,
    traced_step_time: float | None = None,
    h2d: float | None = None,
    clock: str = "cpu",
) -> dict[str, dict[str, dict[str, float | bool | int | None]]]:
    """Encode concise phase values in the production event payload shape.

    A ``None`` traced step time or H2D value omits that event, which is how
    the sampler reports an unavailable signal.
    """
    values = {
        "_traceml_internal:dataloader_next": dataloader,
        "_traceml_internal:forward_time": forward,
        "_traceml_internal:backward_time": backward,
        "_traceml_internal:optimizer_step": optimizer,
    }
    if traced_step_time is not None:
        values["_traceml_internal:step_time"] = traced_step_time
    if h2d is not None:
        values["_traceml_internal:h2d_time"] = h2d
    is_gpu = clock == "gpu"
    device = "cuda:0" if is_gpu else "cpu"
    return {
        event: {
            device: {
                "is_gpu": is_gpu,
                "duration_ms": value,
                "cpu_ms": value,
                "gpu_ms": value if is_gpu else None,
                "n_calls": 1,
            }
        }
        for event, value in values.items()
    }


def insert_step_time_sample(
    conn: sqlite3.Connection,
    *,
    row_id: int,
    rank: int,
    step: int,
    traced_step_time: float | None = None,
    events: Mapping[str, Any] | None = None,
    dataloader: float = 1.0,
    h2d: float | None = None,
    forward: float | None = None,
    backward: float | None = None,
    optimizer: float = 1.0,
    clock: str = "cpu",
    local_rank: int | None = 0,
    world_size: int = 1,
    local_world_size: int = 1,
    node_rank: int | None | object = _UNSET,
    hostname: str | None | object = _UNSET,
    ts: float | None = None,
    seq: int | object = _UNSET,
) -> None:
    """Insert one Step Time row through the production projection writer.

    The sample travels as a wire envelope through the writer's
    ``build_rows`` and ``insert_rows``, so the persisted ``events_json``
    shape and column list are the writer's own. ``events`` replaces the
    generated wire payload, so an empty mapping persists no phases instead
    of falling back to default timings.
    """
    payload = (
        events
        if events is not None
        else step_time_events(
            dataloader=dataloader,
            h2d=h2d,
            forward=2.0 + rank if forward is None else forward,
            backward=3.0 + rank if backward is None else backward,
            optimizer=optimizer,
            traced_step_time=traced_step_time,
            clock=clock,
        )
    )
    sample = StepTimeEventSample(
        seq=_defaulted(seq, row_id),
        timestamp=float(step) if ts is None else ts,
        step=step,
        events=dict(payload),
    )
    envelope = TelemetryEnvelope(
        meta=TelemetryMeta.from_mapping(
            {
                "sampler": step_time_projection.SAMPLER_NAME,
                "rank": rank,
                "global_rank": rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "local_world_size": local_world_size,
                "node_rank": _defaulted(node_rank, rank),
                "hostname": _defaulted(hostname, f"worker-{rank}"),
            }
        ),
        body={"tables": {"StepTimeTable": [sample.to_wire()]}},
    )
    step_time_projection.insert_rows(
        conn,
        step_time_projection.build_rows(envelope, recv_ts_ns=row_id),
    )


def insert_step_memory_sample(
    conn: sqlite3.Connection,
    *,
    row_id: int,
    rank: int,
    step: int,
    alloc: float | None,
    reserved: float | None,
    device: str | None = "cuda:0",
    world_size: int = 1,
    local_world_size: int = 1,
    local_rank: int | None = 0,
    node_rank: int | None | object = _UNSET,
    hostname: str | None | object = _UNSET,
    ts: float | None = None,
    seq: int | None | object = _UNSET,
) -> None:
    """Insert one canonical Step Memory projection row."""
    _insert_row(
        conn,
        "step_memory_samples",
        recv_ts_ns=row_id,
        rank=rank,
        global_rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        local_world_size=local_world_size,
        node_rank=_defaulted(node_rank, rank),
        hostname=_defaulted(hostname, f"worker-{rank}"),
        sample_ts_s=float(step) if ts is None else ts,
        seq=_defaulted(seq, row_id),
        device=device,
        step=step,
        peak_alloc_bytes=alloc,
        peak_reserved_bytes=reserved,
    )


def insert_training_strategy(
    conn: sqlite3.Connection,
    *strategies: str,
) -> None:
    """Append runtime strategy rows in observation order.

    ``recv_ts_ns`` continues after the latest stored row, so repeated calls
    keep the receive clock strictly increasing as the aggregator's does.
    """
    (next_recv_ts_ns,) = conn.execute(
        "SELECT COALESCE(MAX(recv_ts_ns), 0) + 1 FROM runtime_environment"
    ).fetchone()
    conn.executemany(
        "INSERT INTO runtime_environment(recv_ts_ns, training_strategy) "
        "VALUES (?, ?)",
        [
            (recv_ts_ns, strategy)
            for recv_ts_ns, strategy in enumerate(
                strategies, start=next_recv_ts_ns
            )
        ],
    )


__all__ = [
    "init_step_time_schema",
    "init_summary_schema",
    "insert_process_sample",
    "insert_step_memory_sample",
    "insert_step_time_sample",
    "insert_system_sample",
    "insert_training_strategy",
    "sqlite_database",
    "step_time_events",
    "summary_database",
]
