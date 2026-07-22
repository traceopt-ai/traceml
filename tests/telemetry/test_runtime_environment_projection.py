import sqlite3

from traceml_ai.aggregator.sqlite_writers import (
    runtime_environment as runtime_environment_projection,
)
from traceml_ai.telemetry.envelope import normalize_telemetry_envelope


def _runtime_environment_payload(
    *,
    sampler: str = "RuntimeEnvironmentSampler",
    tables: dict | None = None,
) -> dict:
    return {
        "rank": 5,
        "global_rank": 5,
        "local_rank": 1,
        "world_size": 8,
        "local_world_size": 4,
        "node_rank": 1,
        "hostname": "worker-1",
        "pid": 4321,
        "sampler": sampler,
        "timestamp": 123.0,
        "tables": (
            tables
            if tables is not None
            else {
                "RuntimeEnvironmentTable": [
                    {
                        "seq": 0,
                        "ts": 123.25,
                        "topology": "single_node_multi_process",
                        "distributed_initialized": True,
                        "distributed_backend": "nccl",
                        "training_strategy": "ddp",
                        "strategy_source": "runtime_model",
                        "strategy_confidence": "high",
                    }
                ]
            }
        ),
    }


def _envelope(payload: dict):
    envelope = normalize_telemetry_envelope(payload)
    assert envelope is not None
    return envelope


def test_runtime_environment_projection_creates_table() -> None:
    conn = sqlite3.connect(":memory:")
    runtime_environment_projection.init_schema(conn)

    tables = {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table';"
        )
    }

    assert "runtime_environment" in tables


def test_runtime_environment_projection_inserts_identity_and_strategy() -> (
    None
):
    conn = sqlite3.connect(":memory:")
    runtime_environment_projection.init_schema(conn)

    rows = runtime_environment_projection.build_rows(
        _envelope(_runtime_environment_payload()),
        recv_ts_ns=999,
    )
    runtime_environment_projection.insert_rows(conn, rows)

    row = conn.execute(
        """
        SELECT recv_ts_ns, rank, global_rank, local_rank, world_size,
               local_world_size, node_rank, hostname, pid, sample_ts_s, seq,
               topology, distributed_initialized, distributed_backend,
               training_strategy, strategy_source, strategy_confidence
        FROM runtime_environment;
        """
    ).fetchone()

    assert row == (
        999,
        5,
        5,
        1,
        8,
        4,
        1,
        "worker-1",
        4321,
        123.25,
        0,
        "single_node_multi_process",
        1,
        "nccl",
        "ddp",
        "runtime_model",
        "high",
    )


def test_runtime_environment_projection_ignores_non_matching_sampler() -> None:
    rows = runtime_environment_projection.build_rows(
        _envelope(_runtime_environment_payload(sampler="ProcessSampler")),
        recv_ts_ns=999,
    )

    assert rows == {"runtime_environment": []}


def test_runtime_environment_projection_ignores_malformed_payload() -> None:
    rows = runtime_environment_projection.build_rows(
        _envelope(
            _runtime_environment_payload(
                tables={"UnexpectedTable": [{"training_strategy": "ddp"}]}
            )
        ),
        recv_ts_ns=999,
    )

    assert rows == {"runtime_environment": []}
