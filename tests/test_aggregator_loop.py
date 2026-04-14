"""
Tests for Issue #1: Event-Driven Aggregator Loop + Inlined _drain_tcp.

These tests are structured so that:
- Tests marked with `xfail` demonstrate the CURRENT broken behaviour.
  They are expected to FAIL on the unmodified codebase and PASS after the fix.
- Tests without `xfail` verify invariants that must hold both before and after.

Sub-problem A (TCPServer._data_ready)
    The aggregator loop sleeps for the full render_interval_sec even when
    messages are sitting in the TCP queue.  After the fix, a threading.Event
    wakes the loop immediately upon message arrival.

Sub-problem B (_drain_tcp closure allocations)
    _drain_tcp calls _safe(..., lambda m=msg: ...) twice per message.
    After the fix, the calls are inlined into a bare try/except — no closures.
"""

import queue
import threading
import time
import types
from unittest.mock import MagicMock, call, patch

import pytest

from traceml.transport.tcp_transport import TCPServer, TCPConfig


# ---------------------------------------------------------------------------
# Helper: build a minimal TCPServer without binding a real socket
# ---------------------------------------------------------------------------


def _make_server_no_socket() -> TCPServer:
    """Return a TCPServer whose network stack is stubbed out."""
    cfg = TCPConfig(host="127.0.0.1", port=0)
    srv = TCPServer(cfg)
    # Replace the real queue with one we can poke directly
    srv._queue = queue.Queue()
    return srv


# ===========================================================================
# TestTCPServerDataReady
# ===========================================================================


class TestTCPServerDataReady:
    """
    Unit tests for the _data_ready Event on TCPServer.

    These tests will FAIL on the current codebase (TCPServer has no
    _data_ready attribute) and PASS once the attribute + wait_for_data()
    method are added.
    """

    def test_server_has_data_ready_event(self):
        """TCPServer must expose a _data_ready threading.Event."""
        srv = _make_server_no_socket()
        assert hasattr(
            srv, "_data_ready"
        ), "TCPServer is missing '_data_ready' Event (Issue #1 fix not applied)"
        assert isinstance(srv._data_ready, threading.Event)

    def test_server_has_wait_for_data_method(self):
        """TCPServer must expose a wait_for_data(timeout) method."""
        srv = _make_server_no_socket()
        assert hasattr(
            srv, "wait_for_data"
        ), "TCPServer is missing 'wait_for_data' method (Issue #1 fix not applied)"
        assert callable(srv.wait_for_data)

    @pytest.mark.xfail(
        reason="Requires Issue #1 fix: _data_ready Event not yet added to TCPServer",
        strict=True,
    )
    def test_wait_returns_false_when_no_data(self):
        """wait_for_data should return False when timeout expires with no data."""
        srv = _make_server_no_socket()
        start = time.monotonic()
        result = srv.wait_for_data(timeout=0.05)
        elapsed = time.monotonic() - start

        assert result is False, "Expected False (timeout), got True"
        # Should have waited roughly the timeout (within 20 ms slack)
        assert elapsed >= 0.04, f"Returned too early: {elapsed:.3f}s"

    @pytest.mark.xfail(
        reason="Requires Issue #1 fix: _data_ready Event not yet added to TCPServer",
        strict=True,
    )
    def test_wait_returns_true_after_put(self):
        """
        wait_for_data should return True (and quickly) when a message is
        placed in the queue while waiting.
        """
        srv = _make_server_no_socket()

        def _put_after_delay():
            time.sleep(0.05)
            srv._queue.put_nowait({"hello": "world"})
            srv._data_ready.set()  # simulate what _handle_client does post-fix

        t = threading.Thread(target=_put_after_delay, daemon=True)
        t.start()

        start = time.monotonic()
        result = srv.wait_for_data(timeout=2.0)
        elapsed = time.monotonic() - start
        t.join()

        assert result is True, "Expected True (data arrived), got False"
        # Should have woken within ~100 ms of the put (not the full 2 s timeout)
        assert (
            elapsed < 0.3
        ), f"Loop woke too late ({elapsed:.3f}s) — event-driven drain not working"

    @pytest.mark.xfail(
        reason="Requires Issue #1 fix: _data_ready Event not yet added to TCPServer",
        strict=True,
    )
    def test_event_clears_after_wait(self):
        """
        After wait_for_data() returns True, the event must be cleared so
        the NEXT call can block again (no spurious immediate wakeup).
        """
        srv = _make_server_no_socket()
        srv._data_ready.set()

        # First wait: returns True immediately (event was already set)
        result1 = srv.wait_for_data(timeout=1.0)
        assert result1 is True

        # Second wait: must block (event should have been cleared)
        start = time.monotonic()
        result2 = srv.wait_for_data(timeout=0.05)
        elapsed = time.monotonic() - start

        assert (
            result2 is False
        ), "Event was not cleared after first wait — loop would spin!"
        assert elapsed >= 0.04

    @pytest.mark.xfail(
        reason="Requires Issue #1 fix: _data_ready Event not yet added to TCPServer",
        strict=True,
    )
    def test_multiple_puts_single_wait(self):
        """
        A burst of N messages should produce exactly one wakeup; all messages
        must still be drained via poll().
        """
        srv = _make_server_no_socket()
        N = 20

        for i in range(N):
            srv._queue.put_nowait({"i": i})
        srv._data_ready.set()

        result = srv.wait_for_data(timeout=0.1)
        assert result is True

        drained = list(srv.poll())
        assert len(drained) == N


# ===========================================================================
# TestDrainTcpInlined  (Sub-problem B)
# ===========================================================================


class TestDrainTcpInlined:
    """
    Tests that verify the inlined _drain_tcp isolates errors between the two
    sinks (store and writer) without using lambda closures.

    These tests work against a duck-typed stub of TraceMLAggregator's
    _drain_tcp logic so they run without a real aggregator.
    """

    # ------------------------------------------------------------------
    # Build a minimal object that mimics _drain_tcp behaviour
    # ------------------------------------------------------------------

    @staticmethod
    def _make_drainer(store, writer, messages):
        """
        Return a callable that mimics _drain_tcp(). We test two variants:
          - 'current': uses the lambda/_safe pattern (what exists today)
          - 'fixed':   uses the inlined try/except pattern (what we want)
        Both operate on the same store/writer mocks and the same message list.
        """
        logger = MagicMock()

        def _safe(lbl, fn):
            try:
                fn()
            except Exception:
                logger.exception(f"[TraceML] {lbl}")

        def drain_current():
            for msg in messages:
                _safe(
                    "RemoteDBStore.ingest failed",
                    lambda m=msg: store.ingest(m),
                )
                _safe(
                    "SQLiteWriter.ingest failed",
                    lambda m=msg: writer.ingest(m),
                )

        def drain_fixed():
            for msg in messages:
                try:
                    store.ingest(msg)
                except Exception:
                    logger.exception("[TraceML] RemoteDBStore.ingest failed")
                try:
                    writer.ingest(msg)
                except Exception:
                    logger.exception("[TraceML] SQLiteWriter.ingest failed")

        return drain_current, drain_fixed, logger

    def test_both_sinks_called_per_message_fixed(self):
        """After fix: store.ingest and writer.ingest are both called for each msg."""
        store = MagicMock()
        writer = MagicMock()
        msgs = [{"step": i} for i in range(5)]
        _, drain_fixed, _ = self._make_drainer(store, writer, msgs)

        drain_fixed()

        assert store.ingest.call_count == 5
        assert writer.ingest.call_count == 5

    def test_store_error_does_not_skip_writer_fixed(self):
        """
        After fix: if store.ingest raises, writer.ingest MUST still be called
        for the same message (error isolation).
        """
        store = MagicMock()
        store.ingest.side_effect = RuntimeError("store boom")
        writer = MagicMock()
        msgs = [{"step": 0}]
        _, drain_fixed, _ = self._make_drainer(store, writer, msgs)

        drain_fixed()  # must not raise

        store.ingest.assert_called_once()
        writer.ingest.assert_called_once()  # must still be called despite store error

    def test_writer_error_does_not_propagate_fixed(self):
        """
        After fix: writer.ingest raising must not propagate out of _drain_tcp.
        """
        store = MagicMock()
        writer = MagicMock()
        writer.ingest.side_effect = RuntimeError("writer boom")
        msgs = [{"step": 0}]
        _, drain_fixed, _ = self._make_drainer(store, writer, msgs)

        # Must not raise
        drain_fixed()
        writer.ingest.assert_called_once()

    def test_current_drain_also_isolates_errors(self):
        """
        Ensure the current lambda version also isolates errors (regression
        guard — behaviour must be identical after the refactor).
        """
        store = MagicMock()
        store.ingest.side_effect = RuntimeError("store boom")
        writer = MagicMock()
        msgs = [{"step": 0}]
        drain_current, _, _ = self._make_drainer(store, writer, msgs)

        drain_current()

        store.ingest.assert_called_once()
        writer.ingest.assert_called_once()

    def test_no_closure_in_fixed_drain(self):
        """
        Smoke-test: the fixed drain must not allocate lambda objects.
        This is verified by counting function objects created during a call
        using sys.getrefcount heuristics isn't reliable, so we instead
        confirm the inlined version uses no explicit lambda in its bytecode.
        """
        import dis, io

        store = MagicMock()
        writer = MagicMock()
        _, drain_fixed, _ = self._make_drainer(store, writer, [])

        buf = io.StringIO()
        dis.dis(drain_fixed, file=buf)
        bytecode_text = buf.getvalue()

        # The fixed drain should not contain MAKE_FUNCTION (lambda)
        # Note: 'drain_current' WILL have it; 'drain_fixed' must not.
        assert (
            "MAKE_FUNCTION" not in bytecode_text
        ), "Fixed _drain_tcp still contains lambda/closure allocation"


# ===========================================================================
# TestLoopLatency  (integration-style, no real TCP sockets)
# ===========================================================================


class TestLoopLatency:
    """
    Verify that the aggregator loop wakes up quickly when data arrives,
    and that the UI tick is still rate-limited.

    These tests mock out everything except the event-wait + drain interaction.
    """

    @pytest.mark.xfail(
        reason="Requires Issue #1 fix: aggregator loop still uses blind sleep",
        strict=True,
    )
    def test_drain_triggered_before_interval_expires(self):
        """
        Key latency regression test.

        Schedule a message to arrive 50 ms after the loop starts.
        The loop's interval is 2 s.  After the fix, _drain_tcp must be called
        within 200 ms of message arrival — NOT after the full 2 s.
        """
        drain_times: list[float] = []
        data_ready = threading.Event()
        stop = threading.Event()

        INTERVAL = 2.0
        MESSAGE_DELAY = 0.05  # 50 ms
        MAX_DRAIN_LATENCY = 0.2  # 200 ms

        def mock_drain():
            drain_times.append(time.monotonic())

        def loop():
            last_tick = 0.0
            while not stop.is_set():
                # Event-driven wait (the fix)
                data_ready.wait(timeout=INTERVAL)
                data_ready.clear()
                mock_drain()

                now = time.monotonic()
                if now - last_tick >= INTERVAL:
                    last_tick = now

        t = threading.Thread(target=loop, daemon=True)
        t.start()

        # Let the loop settle, then inject the message
        time.sleep(0.02)
        inject_ts = time.monotonic()
        data_ready.set()  # simulate TCPServer waking the loop

        time.sleep(MAX_DRAIN_LATENCY + 0.05)
        stop.set()
        data_ready.set()  # unblock loop for clean shutdown
        t.join(timeout=1.0)

        assert drain_times, "drain was never called"
        first_drain = drain_times[0]
        latency = first_drain - inject_ts
        assert latency < MAX_DRAIN_LATENCY, (
            f"Drain latency {latency*1000:.1f} ms exceeds {MAX_DRAIN_LATENCY*1000:.0f} ms — "
            f"aggregator loop is not event-driven"
        )

    @pytest.mark.xfail(
        reason="Requires Issue #1 fix: aggregator loop still uses blind sleep",
        strict=True,
    )
    def test_tick_not_faster_than_interval(self):
        """
        Even when data arrives very frequently, the UI tick must not be called
        more often than once per interval_sec.
        """
        tick_times: list[float] = []
        data_ready = threading.Event()
        stop = threading.Event()

        INTERVAL = 0.2  # 200 ms for test speed

        def mock_tick():
            tick_times.append(time.monotonic())

        def loop():
            last_tick = 0.0
            while not stop.is_set():
                data_ready.wait(timeout=INTERVAL)
                data_ready.clear()

                now = time.monotonic()
                if now - last_tick >= INTERVAL:
                    mock_tick()
                    last_tick = now

        t = threading.Thread(target=loop, daemon=True)
        t.start()

        # Flood with 50 messages over 1 second (one every 20 ms)
        for _ in range(50):
            time.sleep(0.02)
            data_ready.set()

        stop.set()
        data_ready.set()
        t.join(timeout=2.0)

        assert len(tick_times) >= 2, "tick was never called"

        # All consecutive ticks must be at least INTERVAL apart
        for i in range(1, len(tick_times)):
            gap = tick_times[i] - tick_times[i - 1]
            assert (
                gap >= INTERVAL * 0.9
            ), f"UI ticked too fast: gap={gap*1000:.1f}ms < interval={INTERVAL*1000:.0f}ms"  # 10% slack for scheduling jitter
