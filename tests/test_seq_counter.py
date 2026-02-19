"""
Tests for the seq-counter optimisation in Database, DBIncrementalSender,
and DatabaseWriter.

Validates that the O(1) append-counter based new-row detection behaves
identically to the old identity/equality scan approach.
"""

import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import msgspec
import pytest

from traceml.database.database import Database

# Database.get_append_count() tests


class TestDatabaseAppendCount:
    """Verify the monotonic append counter on Database."""

    def test_new_table_count_is_zero(self):
        db = Database(sampler_name="test")
        db.create_table("t1")
        assert db.get_append_count("t1") == 0

    def test_unknown_table_returns_zero(self):
        db = Database(sampler_name="test")
        assert db.get_append_count("nonexistent") == 0

    def test_count_increments_on_add(self):
        db = Database(sampler_name="test")
        for i in range(5):
            db.add_record("t1", {"i": i})
        assert db.get_append_count("t1") == 5

    def test_count_survives_eviction(self):
        """Counter keeps incrementing even after deque evicts old rows."""
        db = Database(sampler_name="test", max_rows=3)
        for i in range(10):
            db.add_record("t1", {"i": i})
        # Only 3 rows remain, but 10 were appended
        assert len(db.get_table("t1")) == 3
        assert db.get_append_count("t1") == 10

    def test_clear_resets_counters(self):
        db = Database(sampler_name="test")
        db.add_record("t1", {"a": 1})
        db.clear()
        assert db.get_append_count("t1") == 0

    def test_multiple_tables_independent(self):
        db = Database(sampler_name="test")
        db.add_record("t1", {"x": 1})
        db.add_record("t1", {"x": 2})
        db.add_record("t2", {"y": 1})
        assert db.get_append_count("t1") == 2
        assert db.get_append_count("t2") == 1


# DBIncrementalSender tests


class TestDBIncrementalSender:
    """Verify the sender's seq-counter flush logic."""

    def _make_sender(self, db, max_rows_per_flush=-1):
        from traceml.database.database_sender import DBIncrementalSender

        mock_transport = MagicMock()
        sender = DBIncrementalSender(
            db=db,
            sampler_name="test_sampler",
            sender=mock_transport,
            rank=0,
            max_rows_per_flush=max_rows_per_flush,
        )
        return sender, mock_transport

    def test_skip_when_no_new_rows(self):
        db = Database(sampler_name="test")
        db.add_record("t1", {"v": 1})
        sender, transport = self._make_sender(db)

        # First flush: sends the row
        sender.flush()
        assert transport.send.call_count == 1

        # Second flush: nothing new -> no send
        transport.send.reset_mock()
        sender.flush()
        transport.send.assert_not_called()

    def test_sends_only_new_rows(self):
        db = Database(sampler_name="test")
        sender, transport = self._make_sender(db)

        # Add 3 rows, flush
        for i in range(3):
            db.add_record("t1", {"step": i})
        sender.flush()

        payload = transport.send.call_args[0][0]
        rows_sent = payload["tables"]["t1"]
        assert len(rows_sent) == 3

        # Add 2 more, flush again
        transport.send.reset_mock()
        db.add_record("t1", {"step": 3})
        db.add_record("t1", {"step": 4})
        sender.flush()

        payload = transport.send.call_args[0][0]
        rows_sent = payload["tables"]["t1"]
        assert len(rows_sent) == 2
        assert rows_sent[0]["step"] == 3
        assert rows_sent[1]["step"] == 4

    def test_handles_deque_eviction(self):
        """When old rows are evicted, sender sends entire deque."""
        db = Database(sampler_name="test", max_rows=3)
        sender, transport = self._make_sender(db)

        # Add 3 rows, flush (cursor at count=3)
        for i in range(3):
            db.add_record("t1", {"i": i})
        sender.flush()

        # Add 5 more rows (3 evicted, so cursor is now behind deque)
        transport.send.reset_mock()
        for i in range(3, 8):
            db.add_record("t1", {"i": i})
        sender.flush()

        payload = transport.send.call_args[0][0]
        rows_sent = payload["tables"]["t1"]
        # new_count=5, len(deque)=3 → sends all 3 available
        assert len(rows_sent) == 3
        assert rows_sent[0]["i"] == 5
        assert rows_sent[-1]["i"] == 7

    def test_max_rows_per_flush(self):
        """With max_rows_per_flush=1, only the latest row is sent."""
        db = Database(sampler_name="test")
        sender, transport = self._make_sender(db, max_rows_per_flush=1)

        for i in range(5):
            db.add_record("t1", {"step": i})
        sender.flush()

        payload = transport.send.call_args[0][0]
        rows_sent = payload["tables"]["t1"]
        assert len(rows_sent) == 1
        assert rows_sent[0]["step"] == 4  # latest

    def test_multiple_tables(self):
        db = Database(sampler_name="test")
        sender, transport = self._make_sender(db)

        db.add_record("t1", {"a": 1})
        db.add_record("t2", {"b": 2})
        sender.flush()

        payload = transport.send.call_args[0][0]
        assert "t1" in payload["tables"]
        assert "t2" in payload["tables"]

        # Add only to t2
        transport.send.reset_mock()
        db.add_record("t2", {"b": 3})
        sender.flush()

        payload = transport.send.call_args[0][0]
        assert "t1" not in payload["tables"]
        assert "t2" in payload["tables"]
        assert len(payload["tables"]["t2"]) == 1


# DatabaseWriter tests


def _read_framed_records(path: Path) -> list:
    """Read length-prefixed msgpack records (same as test_msgpack_roundtrip)."""
    decoder = msgspec.msgpack.Decoder()
    records = []
    with open(path, "rb") as f:
        while True:
            header = f.read(4)
            if not header:
                break
            length = struct.unpack("!I", header)[0]
            payload = f.read(length)
            records.append(decoder.decode(payload))
    return records


class TestDatabaseWriter:
    """Verify the writer's seq-counter flush logic."""

    def test_incremental_write(self, tmp_path):
        """Writer only writes new rows on subsequent flushes."""
        db = Database(sampler_name="test")

        with patch("traceml.database.database_writer.config") as mock_cfg:
            mock_cfg.enable_logging = True
            mock_cfg.logs_dir = str(tmp_path)

            # flush_every=1 to trigger on every call
            writer = db.writer
            writer.flush_every = 1
            writer.logs_dir = tmp_path / "data" / "test"
            writer._flush_counter = 0

            # Add 3 rows & flush
            for i in range(3):
                db.add_record("t1", {"step": i})
            writer.flush()

            # Add 2 more & flush
            db.add_record("t1", {"step": 3})
            db.add_record("t1", {"step": 4})
            writer.flush()

            # Read back all flushed records
            path = writer.logs_dir / "t1.msgpack"
            records = _read_framed_records(path)
            assert len(records) == 5
            assert [r["step"] for r in records] == [0, 1, 2, 3, 4]

    def test_no_write_when_no_new_rows(self, tmp_path):
        """No disk I/O when nothing is new."""
        db = Database(sampler_name="test")

        with patch("traceml.database.database_writer.config") as mock_cfg:
            mock_cfg.enable_logging = True

            writer = db.writer
            writer.flush_every = 1
            writer.logs_dir = tmp_path / "data" / "test"
            writer._flush_counter = 0

            db.add_record("t1", {"x": 1})
            writer.flush()

            # No new data -> file should not grow
            path = writer.logs_dir / "t1.msgpack"
            size_after_first = path.stat().st_size

            writer.flush()
            assert path.stat().st_size == size_after_first

    def test_writer_handles_eviction(self, tmp_path):
        """Writer handles eviction gracefully, writing available rows."""
        db = Database(sampler_name="test", max_rows=3)

        with patch("traceml.database.database_writer.config") as mock_cfg:
            mock_cfg.enable_logging = True

            writer = db.writer
            writer.flush_every = 1
            writer.logs_dir = tmp_path / "data" / "test"
            writer._flush_counter = 0

            # Add 3 rows & flush
            for i in range(3):
                db.add_record("t1", {"i": i})
            writer.flush()

            # Add 5 more (evicts all originals) & flush
            for i in range(3, 8):
                db.add_record("t1", {"i": i})
            writer.flush()

            path = writer.logs_dir / "t1.msgpack"
            records = _read_framed_records(path)
            # First flush wrote 3, second writes the 3 remaining in deque
            assert len(records) == 6
            assert records[0]["i"] == 0
            assert records[-1]["i"] == 7
