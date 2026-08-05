# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Real client-to-writer TCP wire round-trip (issue #318).

No production test crossed the real wire: exporter/publisher tests fake the
TCP client, and TCPServer tests only cover bind, never accepting and
decoding a framed envelope. These tests use real sockets end to end
(``TCPServer``, ``TCPClient``, ``SQLiteWriterSimple``, and
``TraceMLAggregator._drain_tcp``) on port 0, in one process, no
subprocesses.
"""

from __future__ import annotations

import socket
import sqlite3
import struct
import threading
import time
from pathlib import Path

import pytest

from traceml_ai.aggregator.sqlite_writer import (
    SQLiteWriterConfig,
    SQLiteWriterSimple,
)
from traceml_ai.aggregator.trace_aggregator import TraceMLAggregator
from traceml_ai.runtime.settings import TraceMLSettings
from traceml_ai.samplers.schema.system import SystemSample
from traceml_ai.transport.tcp_transport import TCPClient, TCPConfig, TCPServer
from traceml_ai.utils.msgpack_codec import encode as encode_msgpack


class _Logger:
    def error(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


def _make_aggregator(
    tmp_path: Path, *, tcp_server: TCPServer, writer: SQLiteWriterSimple
) -> TraceMLAggregator:
    """Build a real ``TraceMLAggregator`` around real transport + writer.

    Mirrors the ``TraceMLAggregator.__new__`` + manual-attribute pattern
    already used in ``tests/aggregator/test_finalization.py``, but wires
    real ``TCPServer``/``SQLiteWriterSimple`` instances instead of fakes so
    ``_drain_tcp`` runs unmodified against the real wire and writer.
    """
    agg = TraceMLAggregator.__new__(TraceMLAggregator)
    agg._logger = _Logger()
    agg._stop_event = threading.Event()
    agg._settings = TraceMLSettings(
        mode="summary",
        logs_dir=str(tmp_path),
        session_id="run",
        history_enabled=True,
        db_path=str(tmp_path / "telemetry"),
        expected_world_size=1,
    )
    agg._started = True
    agg._expected_world_size = 1
    agg._finished_ranks = {}
    agg._drain_lock = threading.Lock()
    agg._tcp_server = tcp_server
    agg._sqlite_writer = writer
    return agg


def _system_envelope(*, global_rank: int = 0) -> dict:
    """One canonical SystemSampler envelope built from real wire types."""
    sample = SystemSample(
        sample_idx=1,
        timestamp=time.time(),
        cpu_percent=12.5,
        ram_used=1_000_000.0,
        ram_total=8_000_000.0,
        gpu_available=False,
        gpu_count=0,
        gpus=[],
    )
    return {
        "meta": {
            "global_rank": global_rank,
            "local_rank": 0,
            "world_size": 1,
            "local_world_size": 1,
            "node_rank": 0,
            "hostname": "test-host",
            "pid": 4242,
            "sampler": "SystemSampler",
            "timestamp": time.time(),
        },
        "body": {"tables": {"SystemTable": [sample.to_wire()]}},
    }


@pytest.fixture
def server(tmp_path):
    srv = TCPServer(TCPConfig(host="127.0.0.1", port=0))
    srv.start()
    yield srv
    srv.stop()


@pytest.fixture
def writer(tmp_path):
    w = SQLiteWriterSimple(
        SQLiteWriterConfig(
            path=str(tmp_path / "telemetry"),
            enabled=True,
            flush_interval_sec=0.05,
        )
    )
    w.start()
    yield w
    w.finalize(timeout_sec=5.0)


def _query_system_samples(db_path: Path) -> list[tuple]:
    with sqlite3.connect(str(db_path)) as conn:
        return conn.execute(
            "SELECT global_rank, hostname, cpu_percent FROM system_samples"
        ).fetchall()


def test_client_to_writer_round_trip_projects_sqlite_rows(
    tmp_path, server, writer
):
    client = TCPClient(TCPConfig(host="127.0.0.1", port=server.port))
    agg = _make_aggregator(tmp_path, tcp_server=server, writer=writer)

    client.send_batch([_system_envelope(global_rank=0)])

    assert server.wait_for_data(timeout=2.0), "server never received data"
    agg._drain_tcp()
    assert writer.force_flush(timeout_sec=2.0), "writer never flushed"

    rows = _query_system_samples(tmp_path / "telemetry")
    assert rows == [(0, "test-host", 12.5)]

    client.close()


def test_malformed_frame_is_dropped_and_later_valid_frame_still_lands(
    tmp_path, server, writer
):
    """A corrupted frame on the wire must not wedge the connection."""
    agg = _make_aggregator(tmp_path, tcp_server=server, writer=writer)

    raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    raw.connect(("127.0.0.1", server.port))

    # A well-framed length prefix around bytes that are not valid msgpack.
    garbage = b"\xff\xff\xff\xff\xff\xff\xff\xff"
    raw.sendall(struct.pack("!I", len(garbage)) + garbage)

    # The same connection keeps working: a real envelope right after the
    # corrupt frame must still decode and queue.
    good = encode_msgpack([_system_envelope(global_rank=1)])
    raw.sendall(struct.pack("!I", len(good)) + good)

    assert server.wait_for_data(timeout=2.0), "server never received data"
    agg._drain_tcp()
    assert writer.force_flush(timeout_sec=2.0)

    rows = _query_system_samples(tmp_path / "telemetry")
    assert rows == [(1, "test-host", 12.5)]

    raw.close()


def test_well_shaped_but_unroutable_envelope_yields_no_rows_without_crashing(
    tmp_path, server, writer
):
    """A payload with no recognized sampler must be dropped, not raise."""
    client = TCPClient(TCPConfig(host="127.0.0.1", port=server.port))
    agg = _make_aggregator(tmp_path, tcp_server=server, writer=writer)

    unroutable = {
        "meta": {"sampler": "NotARealSampler"},
        "body": {"tables": {"SystemTable": [{"seq": 1}]}},
    }
    client.send_batch([unroutable])

    assert server.wait_for_data(timeout=2.0)
    agg._drain_tcp()  # must not raise
    assert writer.force_flush(timeout_sec=2.0)

    assert _query_system_samples(tmp_path / "telemetry") == []

    client.close()


def test_client_reconnects_lazily_after_server_restart(tmp_path, writer):
    """TCPClient has no active reconnect loop; the next send() must retry."""
    first = TCPServer(TCPConfig(host="127.0.0.1", port=0))
    first.start()
    bound_port = first.port

    client = TCPClient(TCPConfig(host="127.0.0.1", port=bound_port))
    client.send_batch([_system_envelope(global_rank=0)])
    assert first.wait_for_data(timeout=2.0)
    first.stop()

    # first.stop() only closes the listening socket; the per-client handler
    # thread (_handle_client) notices the stop event on its own next loop
    # turn, bounded by its recv() timeout (tcp_transport.py: conn.settimeout
    # (1.0)). Until that thread closes its side, a write from the client can
    # still succeed at the OS level: measured directly, whether a single
    # post-stop send() sees the break is a genuine ~1s-scale race (2 of 3
    # manual runs at a 1.2s fixed wait detected it, 1 did not). A fixed
    # sleep is the wrong tool for an OS-timing-dependent condition; poll
    # instead, sending repeatedly until the client notices the connection
    # is gone or the deadline passes.
    deadline = time.monotonic() + 5.0
    while client._connected and time.monotonic() < deadline:
        client.send_batch([_system_envelope(global_rank=1)])
        time.sleep(0.2)
    assert (
        client._connected is False
    ), "client did not detect the closed connection within 5s"

    second = TCPServer(TCPConfig(host="127.0.0.1", port=bound_port))
    second.start()
    try:
        agg = _make_aggregator(tmp_path, tcp_server=second, writer=writer)

        client.send_batch([_system_envelope(global_rank=2)])
        assert second.wait_for_data(
            timeout=2.0
        ), "client did not reconnect after the server came back"
        agg._drain_tcp()
        assert writer.force_flush(timeout_sec=2.0)

        rows = _query_system_samples(tmp_path / "telemetry")
        assert rows == [(2, "test-host", 12.5)]
    finally:
        second.stop()
        client.close()
