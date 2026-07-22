# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

import threading
import time

from traceml_ai.runtime.exporter import TelemetryExporter
from traceml_ai.runtime.sender import SenderIdentity, TelemetryPublisher


class _FakeLogger:
    def __init__(self) -> None:
        self.exceptions: list[str] = []
        self.errors: list[str] = []

    def exception(self, message: str, *args: object) -> None:
        self.exceptions.append(message % args if args else message)

    def error(self, message: str, *args: object) -> None:
        self.errors.append(message % args if args else message)


class _FakeClient:
    """Records send_batch calls; can be told to raise or sleep."""

    def __init__(
        self,
        *,
        raise_on_send: bool = False,
        send_delay: float = 0.0,
    ) -> None:
        self._lock = threading.Lock()
        self.batches: list[list] = []
        self.send_calls = 0
        self.closed = False
        self._raise_on_send = raise_on_send
        self._send_delay = send_delay

    def send_batch(self, batch: list) -> None:
        with self._lock:
            self.send_calls += 1
        if self._send_delay:
            time.sleep(self._send_delay)
        if self._raise_on_send:
            raise RuntimeError("send failed")
        with self._lock:
            self.batches.append(batch)

    def send(self, payload: object) -> None:
        self.send_batch([payload])

    def close(self) -> None:
        with self._lock:
            self.closed = True

    def snapshot(self) -> list[list]:
        with self._lock:
            return list(self.batches)

    @property
    def call_count(self) -> int:
        with self._lock:
            return self.send_calls


class _FakeSender:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.sender = None
        self.identity = None

    def collect_payload(self) -> object:
        return self.payload


class _FakeSampler:
    def __init__(self, name: str, *, sender: _FakeSender) -> None:
        self.sampler_name = name
        self.sender = sender
        self.db = None


def _wait_until(pred, timeout: float = 2.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


def test_normal_enqueue_and_export() -> None:
    client = _FakeClient()
    exporter = TelemetryExporter(tcp_client=client, logger=_FakeLogger())
    exporter.start()
    try:
        exporter.send_batch([{"id": 1}])
        assert _wait_until(lambda: client.snapshot() == [[{"id": 1}]])
    finally:
        exporter.stop()
    assert client.closed is True


def test_drop_oldest_when_full() -> None:
    client = _FakeClient()
    exporter = TelemetryExporter(
        tcp_client=client,
        max_queue_size=2,
        logger=_FakeLogger(),
    )
    # Thread not started, so nothing drains and the queue fills.
    exporter.send_batch([{"id": 1}])
    exporter.send_batch([{"id": 2}])
    exporter.send_batch([{"id": 3}])  # drops the oldest ([{"id": 1}])

    assert exporter.queue_size == 2
    assert exporter.dropped_count == 1

    # The newest two survive, in order.
    exporter.start()
    try:
        assert _wait_until(
            lambda: client.snapshot() == [[{"id": 2}], [{"id": 3}]]
        )
    finally:
        exporter.stop()
    assert exporter.dropped_count == 1


def test_aggregator_unavailable_does_not_block_or_crash() -> None:
    logger = _FakeLogger()
    client = _FakeClient(raise_on_send=True)
    exporter = TelemetryExporter(tcp_client=client, logger=logger)
    exporter.start()
    try:
        # Enqueue must return immediately and never raise.
        for i in range(5):
            exporter.send_batch([{"id": i}])
        # The exporter thread keeps attempting sends and swallows failures.
        assert _wait_until(lambda: client.call_count >= 5)
        assert exporter.dropped_count == 0
        assert len(logger.exceptions) >= 5
    finally:
        exporter.stop()


def test_enqueue_after_stop_is_dropped() -> None:
    client = _FakeClient()
    exporter = TelemetryExporter(tcp_client=client, logger=_FakeLogger())
    exporter.start()
    exporter.stop()

    exporter.send_batch([{"id": 1}])

    assert exporter.queue_size == 0
    assert exporter.dropped_count == 1
    assert client.call_count == 0


def test_shutdown_drain_timeout_is_bounded() -> None:
    client = _FakeClient(send_delay=2.0)
    exporter = TelemetryExporter(
        tcp_client=client,
        drain_timeout_sec=0.2,
        poll_interval_sec=0.1,
        logger=_FakeLogger(),
    )
    exporter.start()
    for i in range(10):
        exporter.send_batch([{"id": i}])

    start = time.monotonic()
    exporter.stop()
    elapsed = time.monotonic() - start

    # Must return before a single slow send completes, not after draining all.
    assert elapsed < 2.0


def test_shutdown_drain_timeout_counts_and_logs_remaining_batches() -> None:
    logger = _FakeLogger()
    client = _FakeClient(send_delay=0.2)
    exporter = TelemetryExporter(
        tcp_client=client,
        drain_timeout_sec=0.0,
        poll_interval_sec=0.01,
        logger=logger,
    )
    exporter.start()
    exporter.send_batch([{"id": 1}])
    assert _wait_until(lambda: client.call_count == 1)
    exporter.send_batch([{"id": 2}])
    exporter.send_batch([{"id": 3}])

    exporter.stop()

    assert exporter.queue_size == 0
    assert exporter.dropped_count == 2
    assert any(
        "shutdown dropped 2 queued payload batches" in e for e in logger.errors
    )


def test_tick_enqueues_and_does_not_send_inline() -> None:
    client = _FakeClient()
    exporter = TelemetryExporter(tcp_client=client, logger=_FakeLogger())
    publisher = TelemetryPublisher(
        tcp_client=exporter,
        identity=SenderIdentity(global_rank=0, local_rank=0),
        logger=_FakeLogger(),
    )
    sampler = _FakeSampler("A", sender=_FakeSender(payload={"rows": [1]}))

    # Exporter thread not started: publish must only enqueue, never send.
    publisher.publish([sampler])
    assert client.call_count == 0
    assert exporter.queue_size == 1

    # Once the exporter runs, the batch is sent.
    exporter.start()
    try:
        assert _wait_until(lambda: client.call_count == 1)
        assert client.snapshot() == [[{"rows": [1]}]]
    finally:
        exporter.stop()
