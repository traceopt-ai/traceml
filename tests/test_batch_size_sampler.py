from __future__ import annotations

import sqlite3
import sys
from collections import deque
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import traceml_ai.utils.batch_size as bs_module
from traceml_ai.aggregator.sqlite_writers import batch_size as bs_writer
from traceml_ai.samplers.batch_size_sampler import BatchSizeSampler
from traceml_ai.samplers.schema.batch_size_schema import BatchSizeSample
from traceml_ai.utils.batch_size import (
    BatchSizeBatch,
    BatchSizeEvent,
    flush_batch_size_buffer,
    get_batch_size_queue,
    record_batch_size_bytes,
    tensor_bytes,
)


@pytest.fixture(autouse=True)
def _isolate_batch_size_state():
    """
    Reset the module-level buffer and drain the queue so each test starts
    from a clean slate, regardless of previous test ordering.
    """
    bs_module._BATCH_SIZE_BUFFER.clear()
    q = get_batch_size_queue()
    while not q.empty():
        try:
            q.get_nowait()
        except Exception:
            break
    yield
    bs_module._BATCH_SIZE_BUFFER.clear()
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


# BatchSizeSampler


class TestBatchSizeSampler:
    def test_sums_events_per_step(self):
        sampler = BatchSizeSampler()

        # Two transfers in step 3, one in step 4
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
        assert rows[0]["n_transfers"] == 2
        assert rows[1]["step"] == 4
        assert rows[1]["bytes_total"] == 50
        assert rows[1]["n_transfers"] == 1

    def test_empty_queue_no_records(self):
        sampler = BatchSizeSampler()
        sampler.sample()
        assert sampler.db.get_table("BatchSizeTable") in (None, deque())


# Schema round trip


class TestBatchSizeSchema:
    def test_to_wire_and_from_wire(self):
        sample = BatchSizeSample(
            seq=11,
            timestamp=1234.5,
            step=9,
            bytes_total=4096,
            n_transfers=3,
        )
        wire = sample.to_wire()
        assert wire == {
            "seq": 11,
            "timestamp": 1234.5,
            "step": 9,
            "bytes_total": 4096,
            "n_transfers": 3,
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
                "n_transfers",
            ):
                assert expected in cols
        finally:
            conn.close()

    def test_build_rows_produces_correct_tuple(self):
        payload = {
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
            "tables": {
                "BatchSizeTable": [
                    {
                        "seq": 5,
                        "timestamp": 1001.5,
                        "step": 3,
                        "bytes_total": 2048,
                        "n_transfers": 2,
                    }
                ]
            },
        }
        out = bs_writer.build_rows(payload, recv_ts_ns=10**12)
        rows = out["batch_size_samples"]
        assert len(rows) == 1
        row = rows[0]
        # (recv_ts_ns, rank, global_rank, local_rank, world_size,
        #  local_world_size, node_rank, hostname, runtime_pid,
        #  sample_ts_s, seq, step, bytes_total, n_transfers)
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
        payload = {
            "sampler": "StepTimeSampler",
            "tables": {"x": [{"seq": 1}]},
        }
        out = bs_writer.build_rows(payload, recv_ts_ns=0)
        assert out == {"batch_size_samples": []}

    def test_insert_rows_writes_to_table(self):
        conn = sqlite3.connect(":memory:")
        try:
            bs_writer.init_schema(conn)
            payload = {
                "rank": 0,
                "global_rank": 0,
                "sampler": "BatchSizeSampler",
                "tables": {
                    "BatchSizeTable": [
                        {
                            "seq": 1,
                            "timestamp": 100.0,
                            "step": 1,
                            "bytes_total": 64,
                            "n_transfers": 1,
                        },
                        {
                            "seq": 2,
                            "timestamp": 101.0,
                            "step": 2,
                            "bytes_total": 128,
                            "n_transfers": 4,
                        },
                    ]
                },
            }
            rows = bs_writer.build_rows(payload, recv_ts_ns=1)
            bs_writer.insert_rows(conn, rows)
            persisted = conn.execute(
                "SELECT step, bytes_total, n_transfers FROM batch_size_samples "
                "ORDER BY step;"
            ).fetchall()
            assert persisted == [(1, 64, 1), (2, 128, 4)]
        finally:
            conn.close()


# H2D auto-patch -> bytes recording
#
# The patch internally calls _ORIG_TENSOR_TO which would actually move the
# tensor to CUDA. To stay GPU-free we monkey-patch _ORIG_TENSOR_TO to a
# pass-through stub. The patch's bytes-recording branch runs regardless of
# what _ORIG_TENSOR_TO returns.


class TestH2DAutoPatchRecordsBytes:
    def test_records_when_target_is_cuda(self, monkeypatch):
        import traceml_ai.instrumentation.patches.h2d_auto_timer_patch as h2d_patch

        def _fake_to(_self, *args, **kwargs):
            return _self

        monkeypatch.setattr(h2d_patch, "_ORIG_TENSOR_TO", _fake_to)

        t = torch.zeros(4, 8, dtype=torch.float32)  # 128 bytes
        h2d_patch._H2D_TLS._traceml_h2d_enabled = True
        try:
            h2d_patch._traceml_tensor_to(t, "cuda")
        finally:
            h2d_patch._H2D_TLS._traceml_h2d_enabled = False

        assert len(bs_module._BATCH_SIZE_BUFFER) == 1
        assert bs_module._BATCH_SIZE_BUFFER[0].bytes_count == 128

    def test_does_not_record_cpu_target(self):
        import traceml_ai.instrumentation.patches.h2d_auto_timer_patch as h2d_patch

        t = torch.zeros(4, 8, dtype=torch.float32)
        h2d_patch._H2D_TLS._traceml_h2d_enabled = True
        try:
            h2d_patch._traceml_tensor_to(t, "cpu")
        finally:
            h2d_patch._H2D_TLS._traceml_h2d_enabled = False

        assert len(bs_module._BATCH_SIZE_BUFFER) == 0

    def test_does_not_record_when_disabled(self, monkeypatch):
        import traceml_ai.instrumentation.patches.h2d_auto_timer_patch as h2d_patch

        def _fake_to(_self, *args, **kwargs):
            return _self

        monkeypatch.setattr(h2d_patch, "_ORIG_TENSOR_TO", _fake_to)

        t = torch.zeros(4, 8, dtype=torch.float32)
        # TLS off (default)
        h2d_patch._traceml_tensor_to(t, "cuda")
        assert len(bs_module._BATCH_SIZE_BUFFER) == 0


# wrap_h2d container traversal


class TestWrapH2DRecordsContainerBytes:
    def test_dict_batch_records_total_tensor_bytes(self, monkeypatch):
        # Avoid the global Tensor.to() patch path; this ensures the wrapper
        # takes its own bytes recording branch.
        monkeypatch.setattr(
            torch.Tensor, "_traceml_h2d_patched", False, raising=False
        )

        from traceml_ai.sdk.wrappers import wrap_h2d

        # Subclass of dict: tensor_bytes() recognizes it as a dict container.
        class DictBatch(dict):
            def to(self, device):
                return self

        t1 = torch.zeros(3, dtype=torch.float32)  # 12
        t2 = torch.zeros(2, dtype=torch.int64)  # 16
        batch = DictBatch({"x": t1, "y": t2})
        wrapped = wrap_h2d(batch)
        wrapped.to("cuda")

        # 28 bytes total (12 + 16)
        assert len(bs_module._BATCH_SIZE_BUFFER) == 1
        assert bs_module._BATCH_SIZE_BUFFER[0].bytes_count == 28

    def test_opaque_container_records_zero(self, monkeypatch):
        # When a custom container doesn't subclass dict/list/tuple,
        # tensor_bytes cannot inspect its contents. No event is recorded
        # (no spurious zero rows), which is the safe behavior.
        monkeypatch.setattr(
            torch.Tensor, "_traceml_h2d_patched", False, raising=False
        )

        from traceml_ai.sdk.wrappers import wrap_h2d

        class OpaqueBatch:
            def to(self, device):
                return self

        batch = OpaqueBatch()
        wrapped = wrap_h2d(batch)
        wrapped.to("cuda")

        assert len(bs_module._BATCH_SIZE_BUFFER) == 0
