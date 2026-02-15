"""
Test msgpack roundtrip: DatabaseWriter → traceml inspect reader.

Verifies that records written with length-prefix framing can be
read back correctly using the same framing protocol.
"""

import json
import struct
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import msgspec
import pytest

# helpers


def write_framed_records(path: Path, records: list[dict]) -> None:
    """Write records in the same format as DatabaseWriter.flush()."""
    encoder = msgspec.msgpack.Encoder()
    with open(path, "ab") as f:
        for r in records:
            payload = encoder.encode(r)
            f.write(struct.pack("!I", len(payload)))
            f.write(payload)


def read_framed_records(path: Path) -> list[dict]:
    """Read records using the same protocol as cli.run_inspect()."""
    decoder = msgspec.msgpack.Decoder()
    records = []
    with open(path, "rb") as f:
        while True:
            header = f.read(4)
            if not header:
                break
            assert len(header) == 4, "Truncated header"
            length = struct.unpack("!I", header)[0]
            payload = f.read(length)
            assert len(payload) == length, "Truncated payload"
            records.append(decoder.decode(payload))
    return records


# tests


class TestMsgpackRoundtrip:

    def test_single_record(self, tmp_path):
        path = tmp_path / "test.msgpack"
        original = {"step": 1, "loss": 0.5, "lr": 1e-4}

        write_framed_records(path, [original])
        result = read_framed_records(path)

        assert len(result) == 1
        assert result[0] == original

    def test_multiple_records(self, tmp_path):
        path = tmp_path / "test.msgpack"
        originals = [
            {"step": i, "loss": 1.0 / (i + 1), "name": f"layer_{i}"}
            for i in range(100)
        ]

        write_framed_records(path, originals)
        result = read_framed_records(path)

        assert len(result) == 100
        assert result == originals

    def test_incremental_append(self, tmp_path):
        """Simulate multiple flush() calls appending to the same file."""
        path = tmp_path / "test.msgpack"

        batch1 = [{"step": 1, "val": "a"}, {"step": 2, "val": "b"}]
        batch2 = [{"step": 3, "val": "c"}]

        write_framed_records(path, batch1)
        write_framed_records(path, batch2)

        result = read_framed_records(path)
        assert len(result) == 3
        assert result == batch1 + batch2

    def test_empty_file(self, tmp_path):
        path = tmp_path / "test.msgpack"
        path.write_bytes(b"")

        result = read_framed_records(path)
        assert result == []

    def test_nested_data(self, tmp_path):
        """Ensure complex nested dicts survive the roundtrip."""
        path = tmp_path / "test.msgpack"
        original = {
            "step": 42,
            "gpus": [
                {"id": 0, "mem_used_mb": 1024.5, "util_pct": 85.3},
                {"id": 1, "mem_used_mb": 512.0, "util_pct": 22.1},
            ],
            "tags": ["training", "ddp"],
        }

        write_framed_records(path, [original])
        result = read_framed_records(path)

        assert result[0] == original

    def test_truncated_header_is_detected(self, tmp_path):
        """A file with only 2 bytes should be detected as truncated."""
        path = tmp_path / "test.msgpack"
        path.write_bytes(b"\x00\x01")

        with pytest.raises(AssertionError, match="Truncated header"):
            read_framed_records(path)

    def test_truncated_payload_is_detected(self, tmp_path):
        """Header claims 100 bytes but only 5 are available."""
        path = tmp_path / "test.msgpack"
        # Write a header that says 100 bytes, plus only 5 bytes of data
        path.write_bytes(struct.pack("!I", 100) + b"\x00" * 5)

        with pytest.raises(AssertionError, match="Truncated payload"):
            read_framed_records(path)


class TestCliInspect:
    """Integration test for the CLI `inspect` subcommand."""

    def test_inspect_prints_json(self, tmp_path, capsys):
        """traceml inspect should print each record as pretty JSON."""
        from traceml.cli import run_inspect

        path = tmp_path / "data.msgpack"
        records = [{"step": 1, "loss": 0.9}, {"step": 2, "loss": 0.7}]
        write_framed_records(path, records)

        args = MagicMock()
        args.file = str(path)

        run_inspect(args)

        captured = capsys.readouterr()
        lines = captured.out.strip()

        # Each record should be valid JSON
        # Split on "}\n{" boundary — the output has two JSON objects
        parsed = []
        for block in lines.split("\n"):
            block = block.strip()
            if block:
                try:
                    parsed.append(json.loads(block))
                except json.JSONDecodeError:
                    pass  # partial line from indent=2 formatting

        # Alternatively, just check the full output is parseable as
        # two concatenated JSON docs with indent=2
        full_docs = []
        decoder = json.JSONDecoder()
        idx = 0
        text = captured.out
        while idx < len(text):
            text_remaining = text[idx:].lstrip()
            if not text_remaining:
                break
            obj, end = decoder.raw_decode(text_remaining)
            full_docs.append(obj)
            idx += len(text) - len(text_remaining) + end

        assert len(full_docs) == 2
        assert full_docs[0]["step"] == 1
        assert full_docs[1]["step"] == 2
