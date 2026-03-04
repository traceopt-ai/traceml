"""
test_tcp_buffer_perf.py
=======================
Benchmark and correctness tests for Optimization #1:
  TCPServer._handle_client recv-buffer strategy (bytes vs bytearray).

Usage
-----
Run BEFORE applying the fix to get a baseline:

    pytest tests/test_tcp_buffer_perf.py -v -s 2>&1 | tee baseline_results.txt

Apply the fix in tcp_transport.py, then run again:

    pytest tests/test_tcp_buffer_perf.py -v -s 2>&1 | tee optimized_results.txt

Compare throughput lines:

    grep "throughput\\|MB/s\\|elapsed" baseline_results.txt optimized_results.txt

Structure
---------
1. test_drain_frames_correctness  - bytes vs bytearray give identical frame output
2. test_bench_bytes_concat        - simulates OLD approach (bytes +=)
3. test_bench_bytearray_extend    - simulates NEW approach (bytearray.extend)
4. test_tcp_roundtrip             - full end-to-end: TCPClient -> TCPServer -> poll()
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Optional

import msgspec
import pytest

# ---------------------------------------------------------------------------
# Import the module under test.
# _drain_frames is a method on TCPServer; we test it via an instance.
# ---------------------------------------------------------------------------
from traceml.transport.tcp_transport import TCPClient, TCPConfig, TCPServer

# ===========================================================================
# Helpers — frame building (mirrors TCPClient.send logic)
# ===========================================================================


def _encode_frame(payload: dict) -> bytes:
    """Encode a single dict as a length-prefixed msgpack frame."""
    data = msgspec.msgpack.encode(payload)
    return struct.pack("!I", len(data)) + data


def _build_stream(messages: list[dict]) -> bytes:
    """
    Concatenate N framed messages into a single byte stream,
    as if they arrived over the wire in one big chunk.
    """
    return b"".join(_encode_frame(m) for m in messages)


# ===========================================================================
# 1. Correctness: _drain_frames works identically for bytes and bytearray
# ===========================================================================


class TestDrainFramesCorrectness:
    """
    Verifies that _drain_frames produces identical output regardless of
    whether the buffer is `bytes` or `bytearray`.

    This is the key safety check before / after the optimization.
    """

    def _make_server(self) -> TCPServer:
        """Return a bare TCPServer instance (not started)."""
        return TCPServer(TCPConfig())

    def test_single_frame_bytes(self):
        server = self._make_server()
        msg = {"rank": 0, "loss": 0.5}
        stream = _build_stream([msg])

        frames, remaining, expected = server._drain_frames(stream, None)

        assert len(frames) == 1
        assert remaining == b""
        assert expected is None
        decoded = msgspec.msgpack.decode(frames[0])
        assert decoded == msg

    def test_single_frame_bytearray(self):
        """Same as above but with the buffer as bytearray (post-opt type)."""
        server = self._make_server()
        msg = {"rank": 0, "loss": 0.5}
        stream = bytearray(_build_stream([msg]))

        frames, remaining, expected = server._drain_frames(stream, None)

        assert len(frames) == 1
        decoded = msgspec.msgpack.decode(frames[0])
        assert decoded == msg

    def test_multiple_frames_identical_output(self):
        """bytes and bytearray inputs must yield frame-for-frame identical output."""
        server = self._make_server()
        messages = [{"step": i, "val": float(i) * 0.1} for i in range(50)]
        stream_bytes = _build_stream(messages)
        stream_bytearray = bytearray(stream_bytes)

        frames_b, rem_b, _e_b = server._drain_frames(stream_bytes, None)
        frames_ba, rem_ba, _e_ba = server._drain_frames(stream_bytearray, None)

        assert len(frames_b) == len(frames_ba) == 50
        assert rem_b == b"" or rem_b == bytearray()
        for fb, fba in zip(frames_b, frames_ba):
            assert msgspec.msgpack.decode(fb) == msgspec.msgpack.decode(fba)

    def test_partial_frame_held(self):
        """
        If not all bytes of the last frame have arrived yet,
        the remainder is held in the buffer (not decoded).
        """
        server = self._make_server()
        msg = {"key": "x" * 200}
        full_frame = _encode_frame(msg)
        # Feed only the first half
        partial_stream = bytearray(full_frame[: len(full_frame) // 2])

        frames, remaining, expected = server._drain_frames(
            partial_stream, None
        )

        assert frames == []
        assert len(remaining) > 0

    def test_fragmented_delivery(self):
        """
        Simulates arriving byte-by-byte (worst case fragmentation).
        All chunks fed sequentially must eventually decode the full message.
        """
        server = self._make_server()
        msg = {"data": list(range(20))}
        full_stream = bytearray(_build_stream([msg]))

        buffer = bytearray()
        expected = None
        all_frames = []

        for byte in full_stream:
            buffer.extend(bytes([byte]))
            frames, buffer, expected = server._drain_frames(buffer, expected)
            all_frames.extend(frames)

        assert len(all_frames) == 1
        assert msgspec.msgpack.decode(all_frames[0]) == msg


# ===========================================================================
# 2. Benchmark — OLD pattern: bytes +=
# ===========================================================================


class TestBenchBytesConcat:
    """
    Simulates the CURRENT (pre-optimization) recv-buffer strategy.

    This does NOT use TCPServer directly; it isolates the exact pattern:
        buffer: bytes = b""
        buffer += chunk
    to measure the raw allocation cost.
    """

    @pytest.mark.parametrize(
        "msg_size_bytes,n_msgs",
        [
            (256, 5_000),  # small messages, many iterations
            (4_096, 2_000),  # medium messages
            (65_536, 500),  # large messages (== recv_buf default)
        ],
        ids=["256B×5K", "4KB×2K", "64KB×500"],
    )
    def test_bench_bytes_concat(self, msg_size_bytes: int, n_msgs: int):
        """
        Measures throughput of the OLD ``bytes +=`` pattern.

        The buffer simulates a connection that receives `n_msgs` chunks of
        `msg_size_bytes` bytes. All data ends up in a single ever-growing
        `bytes` object — matching the real recv loop.
        """
        chunk = bytes(msg_size_bytes)  # zero-filled chunk
        total_bytes = msg_size_bytes * n_msgs

        buffer: bytes = b""
        start = time.perf_counter()
        for _ in range(n_msgs):
            buffer += chunk
        elapsed = time.perf_counter() - start

        throughput_mb = (total_bytes / 1024 / 1024) / elapsed
        print(
            f"\n[BYTES +=] size={msg_size_bytes}B n={n_msgs} | "
            f"elapsed={elapsed*1000:.1f}ms  throughput={throughput_mb:.1f} MB/s"
        )
        assert len(buffer) == total_bytes  # sanity


# ===========================================================================
# 3. Benchmark — NEW pattern: bytearray.extend
# ===========================================================================


class TestBenchBytearrayExtend:
    """
    Simulates the OPTIMIZED recv-buffer strategy.

    Isolates:
        buffer: bytearray = bytearray()
        buffer.extend(chunk)
    """

    @pytest.mark.parametrize(
        "msg_size_bytes,n_msgs",
        [
            (256, 5_000),
            (4_096, 2_000),
            (65_536, 500),
        ],
        ids=["256B×5K", "4KB×2K", "64KB×500"],
    )
    def test_bench_bytearray_extend(self, msg_size_bytes: int, n_msgs: int):
        """
        Measures throughput of the NEW ``bytearray.extend()`` pattern.

        Should be significantly faster than ``bytes +=`` for large inputs,
        and show zero degradation as the buffer grows.
        """
        chunk = bytes(msg_size_bytes)
        total_bytes = msg_size_bytes * n_msgs

        buffer: bytearray = bytearray()
        start = time.perf_counter()
        for _ in range(n_msgs):
            buffer.extend(chunk)
        elapsed = time.perf_counter() - start

        throughput_mb = (total_bytes / 1024 / 1024) / elapsed
        print(
            f"\n[BYTEARRAY] size={msg_size_bytes}B n={n_msgs} | "
            f"elapsed={elapsed*1000:.1f}ms  throughput={throughput_mb:.1f} MB/s"
        )
        assert len(buffer) == total_bytes  # sanity


# ===========================================================================
# 4. End-to-end roundtrip: TCPClient → TCPServer → poll()
# ===========================================================================


class TestTCPRoundtrip:
    """
    Full integration test: brings up a real TCPServer + TCPClient on localhost,
    sends N msgpack payloads, and verifies they all arrive via poll().

    Run this BOTH before and after the optimization to prove correctness
    is preserved.
    """

    # Use an unlikely port to avoid clashing with the real aggregator
    TEST_PORT = 29788

    def _free_port(self) -> int:
        """Find a free TCP port dynamically (avoids hard-coded conflicts)."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def test_roundtrip_small_payloads(self):
        """Send 100 small messages and assert all are received."""
        port = self._free_port()
        cfg = TCPConfig(host="127.0.0.1", port=port)
        server = TCPServer(cfg)
        server.start()
        time.sleep(0.05)  # give the server thread time to bind

        client = TCPClient(cfg)
        messages = [
            {"rank": 0, "step": i, "loss": 1.0 / (i + 1)} for i in range(100)
        ]
        for msg in messages:
            client.send(msg)
        client.close()

        # Give server thread time to process
        time.sleep(0.3)
        server.stop()

        received = list(server.poll())
        assert len(received) == 100

        steps_received = sorted(r["step"] for r in received)
        assert steps_received == list(range(100))

    def test_roundtrip_large_payloads(self):
        """
        Send payloads larger than recv_buf (65536 bytes) to exercise
        multi-chunk reassembly in _drain_frames.
        """
        port = self._free_port()
        cfg = TCPConfig(host="127.0.0.1", port=port, recv_buf=4096)
        server = TCPServer(cfg)
        server.start()
        time.sleep(0.05)

        client = TCPClient(cfg)
        # Each payload is ~16 KB — forces multiple recv() calls per message
        big_payload = {"rank": 0, "data": "x" * 16_000}
        n = 20
        for _ in range(n):
            client.send(big_payload)
        client.close()

        time.sleep(0.5)
        server.stop()

        received = list(server.poll())
        assert len(received) == n
        for r in received:
            assert r["data"] == "x" * 16_000

    def test_roundtrip_batch_send(self):
        """
        Exercises TCPClient.send_batch() — N payloads in one syscall,
        decoded as a list envelope on the server side.
        """
        port = self._free_port()
        cfg = TCPConfig(host="127.0.0.1", port=port)
        server = TCPServer(cfg)
        server.start()
        time.sleep(0.05)

        client = TCPClient(cfg)
        batch = [{"rank": 0, "step": i} for i in range(50)]
        client.send_batch(batch)
        client.close()

        time.sleep(0.3)
        server.stop()

        # send_batch() sends ONE framed message that is a list
        received = list(server.poll())
        # The server enqueues the raw decoded object (which is the list)
        assert len(received) == 1
        assert isinstance(received[0], list)
        assert len(received[0]) == 50

    def test_roundtrip_throughput(self):
        """
        Stress test: send 1000 messages as fast as possible and
        measure end-to-end throughput.  Prints MB/s for before/after comparison.
        """
        port = self._free_port()
        cfg = TCPConfig(host="127.0.0.1", port=port)
        server = TCPServer(cfg)
        server.start()
        time.sleep(0.05)

        client = TCPClient(cfg)
        n = 1_000
        payload = {"rank": 0, "step": 0, "data": "y" * 512}
        encoded_size = (
            len(msgspec.msgpack.encode(payload)) + 4
        )  # +4 for header

        start = time.perf_counter()
        for i in range(n):
            payload["step"] = i
            client.send(payload)
        client.close()

        time.sleep(1.0)  # allow server to drain
        server.stop()
        elapsed = time.perf_counter() - start

        received = list(server.poll())
        total_mb = (encoded_size * n) / 1024 / 1024
        throughput = total_mb / elapsed

        print(
            f"\n[E2E THROUGHPUT] n={n} payload={encoded_size}B | "
            f"received={len(received)} elapsed={elapsed*1000:.0f}ms "
            f"throughput={throughput:.1f} MB/s"
        )

        # At minimum, we must receive all messages (no drops on loopback)
        assert len(received) == n
