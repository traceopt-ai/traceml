from __future__ import annotations

import sqlite3
import time
from collections import deque

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import traceml_ai.utils.batch_size as bs_module
from traceml_ai.runtime.state import (
    configure_trace_recording,
    mark_trace_step_flushed,
)
from traceml_ai.aggregator.sqlite_writers import batch_size as bs_writer
from traceml_ai.samplers.batch_size_sampler import BatchSizeSampler
from traceml_ai.samplers.schema.batch_size_schema import BatchSizeSample
from traceml_ai.telemetry.envelope import TelemetryEnvelope, TelemetryMeta
from traceml_ai.utils.batch_size import (
    BatchSizeBatch,
    flush_batch_size_buffer,
    get_batch_size_queue,
    record_batch_size_bytes,
    tensor_bytes,
)


@pytest.fixture(autouse=True)
def _isolate_batch_size_state():
    """
    Reset the module-level buffer and drain the queue so each test starts
    from a clean slate, regardless of previous test ordering. The step-time
    buffer is cleared too: the dataloader tests iterate through the real
    timed_region, which would otherwise leak fetch TimeEvents into later
    step-time tests in the same process.
    """
    import traceml_ai.utils.timing as timing_module

    bs_module._BATCH_SIZE_BUFFER.clear()
    timing_module._STEP_BUFFER.clear()
    q = get_batch_size_queue()
    while not q.empty():
        try:
            q.get_nowait()
        except Exception:
            break
    yield
    bs_module._BATCH_SIZE_BUFFER.clear()
    timing_module._STEP_BUFFER.clear()
    while not q.empty():
        try:
            q.get_nowait()
        except Exception:
            break


# tensor_bytes


class TestTensorBytes:
    def test_tensor(self):
        t = torch.zeros(4, 8, dtype=torch.float32)
        assert tensor_bytes(t) == 4 * 8 * 4  # 4 bytes/elem

    def test_dict_of_tensors(self):
        t1 = torch.zeros(3, dtype=torch.float32)  # 12
        t2 = torch.zeros(2, dtype=torch.int64)  # 16
        batch = {"x": t1, "y": t2, "label": "not a tensor"}
        assert tensor_bytes(batch) == 12 + 16

    def test_list_of_tensors(self):
        t1 = torch.zeros(2, dtype=torch.float32)  # 8
        t2 = torch.zeros(2, dtype=torch.float32)  # 8
        assert tensor_bytes([t1, t2, "skip"]) == 16

    def test_tuple_of_tensors(self):
        t1 = torch.zeros(1, dtype=torch.float64)  # 8
        assert tensor_bytes((t1,)) == 8

    def test_unknown_returns_zero(self):
        assert tensor_bytes(object()) == 0

    def test_raising_container_returns_zero(self):
        # Instrumentation must never crash user training: a container
        # whose iteration raises is sized as 0, not propagated.
        class ExplodingBatch(dict):
            def values(self):
                raise RuntimeError("lazy mapping not materialized")

        assert tensor_bytes(ExplodingBatch({"x": 1})) == 0


# record_batch_size_bytes


class TestRecordBatchSizeBytes:
    def test_positive_records_event(self):
        record_batch_size_bytes(100)
        assert len(bs_module._BATCH_SIZE_BUFFER) == 1
        assert bs_module._BATCH_SIZE_BUFFER[0].bytes_count == 100

    def test_zero_is_ignored(self):
        record_batch_size_bytes(0)
        assert len(bs_module._BATCH_SIZE_BUFFER) == 0

    def test_negative_is_ignored(self):
        record_batch_size_bytes(-5)
        assert len(bs_module._BATCH_SIZE_BUFFER) == 0

    def test_invalid_is_ignored(self):
        record_batch_size_bytes("not an int")  # type: ignore[arg-type]
        assert len(bs_module._BATCH_SIZE_BUFFER) == 0


# flush_batch_size_buffer


class TestFlushBatchSizeBuffer:
    def test_no_events_no_enqueue(self):
        flush_batch_size_buffer(step=1)
        assert get_batch_size_queue().empty()

    def test_flush_drains_buffer_and_enqueues_one_batch(self):
        record_batch_size_bytes(100)
        record_batch_size_bytes(200)
        flush_batch_size_buffer(step=7)

        assert len(bs_module._BATCH_SIZE_BUFFER) == 0

        batch = get_batch_size_queue().get_nowait()
        assert isinstance(batch, BatchSizeBatch)
        assert batch.step == 7
        assert [e.bytes_count for e in batch.events] == [100, 200]
        assert all(e.step == 7 for e in batch.events)

    def test_flush_stamps_step_end_timestamp(self):
        # The schema documents the row timestamp as the step timestamp,
        # so the batch is stamped when the step flushes, not when the
        # sampler thread later drains the queue.
        record_batch_size_bytes(100)
        before = time.time()
        flush_batch_size_buffer(step=7)
        after = time.time()

        batch = get_batch_size_queue().get_nowait()
        assert before <= batch.timestamp <= after


# BatchSizeSampler


class TestBatchSizeSampler:
    def test_sums_events_per_step(self):
        sampler = BatchSizeSampler()

        # Two fetches in step 3, one in step 4
        record_batch_size_bytes(100)
        record_batch_size_bytes(250)
        flush_batch_size_buffer(step=3)

        record_batch_size_bytes(50)
        flush_batch_size_buffer(step=4)

        sampler.sample()

        rows = list(sampler.db.get_table("BatchSizeTable") or [])
        assert len(rows) == 2
        assert rows[0]["step"] == 3
        assert rows[0]["bytes_total"] == 350
        assert rows[0]["n_fetches"] == 2
        assert rows[1]["step"] == 4
        assert rows[1]["bytes_total"] == 50
        assert rows[1]["n_fetches"] == 1

    def test_empty_queue_no_records(self):
        sampler = BatchSizeSampler()
        sampler.sample()
        assert sampler.db.get_table("BatchSizeTable") in (None, deque())

    def test_row_timestamp_comes_from_the_flush_not_the_drain(self):
        sampler = BatchSizeSampler()

        record_batch_size_bytes(100)
        flush_batch_size_buffer(step=3)
        flushed_at = get_batch_size_queue().queue[0].timestamp

        sampler.sample()

        rows = list(sampler.db.get_table("BatchSizeTable") or [])
        assert rows[0]["timestamp"] == flushed_at

    def test_has_pending_recording_data_reflects_pending_batches(self):
        # Mirrors StepTimeSampler: the final-drain loop must keep
        # retrying while batches remain buffered.
        sampler = BatchSizeSampler()
        assert sampler.has_pending_recording_data() is False

        sampler._pending.append(BatchSizeBatch(step=1))
        assert sampler.has_pending_recording_data() is True


# Schema round trip


class TestBatchSizeSchema:
    def test_to_wire_and_from_wire(self):
        sample = BatchSizeSample(
            seq=11,
            timestamp=1234.5,
            step=9,
            bytes_total=4096,
            n_fetches=3,
        )
        wire = sample.to_wire()
        assert wire == {
            "seq": 11,
            "timestamp": 1234.5,
            "step": 9,
            "bytes_total": 4096,
            "n_fetches": 3,
        }
        round_trip = BatchSizeSample.from_wire(wire)
        assert round_trip == sample


# SQLite projection writer


class TestBatchSizeSqlWriter:
    def test_accepts_sampler(self):
        assert bs_writer.accepts_sampler("BatchSizeSampler") is True
        assert bs_writer.accepts_sampler("StepTimeSampler") is False
        assert bs_writer.accepts_sampler(None) is False

    def test_init_schema_creates_table(self):
        conn = sqlite3.connect(":memory:")
        try:
            bs_writer.init_schema(conn)
            cols = [
                r[1]
                for r in conn.execute(
                    "PRAGMA table_info(batch_size_samples);"
                ).fetchall()
            ]
            for expected in (
                "id",
                "recv_ts_ns",
                "rank",
                "global_rank",
                "local_rank",
                "world_size",
                "local_world_size",
                "node_rank",
                "hostname",
                "runtime_pid",
                "sample_ts_s",
                "seq",
                "step",
                "bytes_total",
                "n_fetches",
            ):
                assert expected in cols
        finally:
            conn.close()

    def test_build_rows_produces_correct_tuple(self):
        envelope = TelemetryEnvelope(
            meta=TelemetryMeta.from_mapping(
                {
                    "rank": 1,
                    "global_rank": 1,
                    "local_rank": 1,
                    "world_size": 2,
                    "local_world_size": 2,
                    "node_rank": 0,
                    "hostname": "host-a",
                    "pid": 42,
                    "sampler": "BatchSizeSampler",
                    "timestamp": 1000.0,
                }
            ),
            body={
                "tables": {
                    "BatchSizeTable": [
                        {
                            "seq": 5,
                            "timestamp": 1001.5,
                            "step": 3,
                            "bytes_total": 2048,
                            "n_fetches": 2,
                        }
                    ]
                }
            },
        )
        out = bs_writer.build_rows(envelope=envelope, recv_ts_ns=10**12)
        rows = out["batch_size_samples"]
        assert len(rows) == 1
        row = rows[0]
        # (recv_ts_ns, rank, global_rank, local_rank, world_size,
        #  local_world_size, node_rank, hostname, runtime_pid,
        #  sample_ts_s, seq, step, bytes_total, n_fetches)
        assert row[0] == 10**12
        assert row[1] == 1
        assert row[2] == 1
        assert row[7] == "host-a"
        assert row[8] == 42
        assert row[9] == 1001.5
        assert row[10] == 5
        assert row[11] == 3
        assert row[12] == 2048
        assert row[13] == 2

    def test_build_rows_rejects_other_sampler(self):
        envelope = TelemetryEnvelope(
            meta=TelemetryMeta.from_mapping({"sampler": "StepTimeSampler"}),
            body={"tables": {"x": [{"seq": 1}]}},
        )
        out = bs_writer.build_rows(envelope=envelope, recv_ts_ns=0)
        assert out == {"batch_size_samples": []}

    def test_insert_rows_writes_to_table(self):
        conn = sqlite3.connect(":memory:")
        try:
            bs_writer.init_schema(conn)
            envelope = TelemetryEnvelope(
                meta=TelemetryMeta.from_mapping(
                    {
                        "rank": 0,
                        "global_rank": 0,
                        "sampler": "BatchSizeSampler",
                    }
                ),
                body={
                    "tables": {
                        "BatchSizeTable": [
                            {
                                "seq": 1,
                                "timestamp": 100.0,
                                "step": 1,
                                "bytes_total": 64,
                                "n_fetches": 1,
                            },
                            {
                                "seq": 2,
                                "timestamp": 101.0,
                                "step": 2,
                                "bytes_total": 128,
                                "n_fetches": 4,
                            },
                        ]
                    }
                },
            )
            rows = bs_writer.build_rows(envelope=envelope, recv_ts_ns=1)
            bs_writer.insert_rows(conn, rows)
            persisted = conn.execute(
                "SELECT step, bytes_total, n_fetches FROM batch_size_samples "
                "ORDER BY step;"
            ).fetchall()
            assert persisted == [(1, 64, 1), (2, 128, 4)]
        finally:
            conn.close()


# Dataloader auto-patch -> bytes recording
#
# The batch is sized as it leaves the dataloader, so the metric works the
# same on CPU-only and GPU training. _traceml_dataloader_iter is called
# directly to avoid installing the global DataLoader patch in tests.


class TestDataloaderPatchRecordsBytes:
    def test_records_batch_bytes_per_fetch(self, monkeypatch):
        import traceml_ai.instrumentation.patches.dataloader_patch as dl_patch

        monkeypatch.setattr(dl_patch, "is_tracing_armed", lambda: True)

        ds = TensorDataset(torch.zeros(8, 4, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=4)

        batches = list(dl_patch._traceml_dataloader_iter(loader))

        # Two fetches of [tensor(4, 4) float32] = 64 bytes each.
        assert len(batches) == 2
        assert [e.bytes_count for e in bs_module._BATCH_SIZE_BUFFER] == [
            64,
            64,
        ]

    def test_does_not_record_when_not_armed(self, monkeypatch):
        import traceml_ai.instrumentation.patches.dataloader_patch as dl_patch

        monkeypatch.setattr(dl_patch, "is_tracing_armed", lambda: False)

        ds = TensorDataset(torch.zeros(4, 2, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=2)

        batches = list(dl_patch._traceml_dataloader_iter(loader))

        assert len(batches) == 2
        assert len(bs_module._BATCH_SIZE_BUFFER) == 0

    def test_does_not_record_after_recording_stops(self, monkeypatch):
        import traceml_ai.instrumentation.patches.dataloader_patch as dl_patch

        monkeypatch.setattr(dl_patch, "is_tracing_armed", lambda: True)

        configure_trace_recording(max_steps=1)
        mark_trace_step_flushed(1)
        try:
            ds = TensorDataset(torch.zeros(4, 2, dtype=torch.float32))
            loader = DataLoader(ds, batch_size=2)
            batches = list(dl_patch._traceml_dataloader_iter(loader))

            assert len(batches) == 2
            assert len(bs_module._BATCH_SIZE_BUFFER) == 0
        finally:
            configure_trace_recording()


# Manual wrapper (wrap_dataloader_fetch) -> bytes recording


class TestWrapDataloaderFetchRecordsBytes:
    def test_dict_batch_records_total_tensor_bytes(self):
        from traceml_ai.sdk.wrappers import wrap_dataloader_fetch

        class DictLoader:
            def __iter__(self):
                yield {
                    "x": torch.zeros(3, dtype=torch.float32),  # 12
                    "y": torch.zeros(2, dtype=torch.int64),  # 16
                }

        wrapped = wrap_dataloader_fetch(DictLoader())
        batches = list(wrapped)

        assert len(batches) == 1
        assert [e.bytes_count for e in bs_module._BATCH_SIZE_BUFFER] == [28]

    def test_opaque_batch_records_nothing(self):
        # When a batch is not a tensor or dict/list/tuple of tensors,
        # tensor_bytes cannot inspect it. No event is recorded (no spurious
        # zero rows), which is the safe behavior.
        from traceml_ai.sdk.wrappers import wrap_dataloader_fetch

        class OpaqueLoader:
            def __iter__(self):
                yield object()

        wrapped = wrap_dataloader_fetch(OpaqueLoader())
        batches = list(wrapped)

        assert len(batches) == 1
        assert len(bs_module._BATCH_SIZE_BUFFER) == 0

    def test_wrapper_defers_to_the_patched_iterator(self, monkeypatch):
        # wrap-then-init race: when the wrapped iterator is the auto
        # patch's own generator, the patch already records bytes, so the
        # wrapper must not record a second time.
        import traceml_ai.instrumentation.patches.dataloader_patch as dlp
        from traceml_ai.sdk.wrappers import _WrappedDataLoaderIterator

        monkeypatch.setattr(dlp, "is_tracing_armed", lambda: True)

        ds = TensorDataset(torch.zeros(4, 2, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=2)

        patched_it = dlp._traceml_dataloader_iter(loader)
        wrapped = _WrappedDataLoaderIterator(patched_it)
        batches = list(wrapped)

        # Two fetches of [tensor(2, 2) float32] = 16 bytes each,
        # recorded once each, not twice.
        assert len(batches) == 2
        assert [e.bytes_count for e in bs_module._BATCH_SIZE_BUFFER] == [
            16,
            16,
        ]


# flush_step_events -> flush_batch_size_buffer wiring
#
# This is the only production producer of the batch-size queue; pin it so
# removing the call in utils/flush_buffers.py cannot stay green.


class TestFlushStepEventsWiring:
    def test_flush_step_events_flushes_batch_size_buffer(self):
        import torch.nn as nn

        from traceml_ai.utils.flush_buffers import flush_step_events

        record_batch_size_bytes(100)
        flush_step_events(nn.Linear(1, 1), step=5)

        batch = get_batch_size_queue().get_nowait()
        assert batch.step == 5
        assert [e.bytes_count for e in batch.events] == [100]
