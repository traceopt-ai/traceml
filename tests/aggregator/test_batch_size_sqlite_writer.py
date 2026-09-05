import sqlite3

from traceml_ai.aggregator.sqlite_writers import batch_size as bs_writer
from traceml_ai.samplers.schema.batch_size_schema import BatchSizeSample
from traceml_ai.telemetry.envelope import TelemetryEnvelope, TelemetryMeta


def _make_envelope(
    *,
    sampler: str = "BatchSizeSampler",
    step: int = 3,
    bytes_total: int = 2_105_344,
    n_fetches: int = 4,
) -> TelemetryEnvelope:
    meta = TelemetryMeta.from_mapping(
        {
            "global_rank": 1,
            "local_rank": 1,
            "world_size": 2,
            "local_world_size": 2,
            "node_rank": 0,
            "hostname": "node-a",
            "pid": 4242,
            "sampler": sampler,
            "timestamp": 1723.5,
        }
    )
    sample = BatchSizeSample(
        seq=7,
        timestamp=1723.5,
        step=step,
        bytes_total=bytes_total,
        n_fetches=n_fetches,
    )
    return TelemetryEnvelope(
        meta=meta,
        body={"tables": {"BatchSizeTable": [sample.to_wire()]}},
    )


def test_build_rows_projects_envelope_into_batch_size_row() -> None:
    """
    Pin the full projection path with the exact call form used by
    ``SQLiteWriter._collect_projection_rows``: build_rows(envelope=..., ...)
    -> insert_rows -> one queryable row per (rank, step).
    """
    envelope = _make_envelope()

    rows_by_table = bs_writer.build_rows(envelope=envelope, recv_ts_ns=123)

    with sqlite3.connect(":memory:") as conn:
        bs_writer.init_schema(conn)
        bs_writer.insert_rows(conn, rows_by_table)
        rows = conn.execute(
            """
            SELECT recv_ts_ns, rank, global_rank, hostname, runtime_pid,
                   step, bytes_total, n_fetches
            FROM batch_size_samples;
            """
        ).fetchall()

    assert rows == [(123, 1, 1, "node-a", 4242, 3, 2_105_344, 4)]


def test_build_rows_ignores_other_samplers() -> None:
    envelope = _make_envelope(sampler="StepTimeSampler")

    rows_by_table = bs_writer.build_rows(envelope=envelope, recv_ts_ns=123)

    assert rows_by_table == {"batch_size_samples": []}


def test_writer_is_registered_with_the_aggregator() -> None:
    """
    Pin the registration itself: without these entries the aggregator
    never creates or fills batch_size_samples, while every module-level
    writer test stays green.
    """
    from traceml_ai.aggregator import sqlite_writer

    assert bs_writer in sqlite_writer._PROJECTION_WRITERS
    assert "batch_size_samples" in sqlite_writer._RETENTION_PARTITION_SQL
