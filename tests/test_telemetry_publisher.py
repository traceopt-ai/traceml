from traceml.runtime.sender import TelemetryPublisher


class _FakeLogger:
    def __init__(self) -> None:
        self.exceptions: list[tuple[str, tuple[object, ...]]] = []
        self.errors: list[str] = []

    def exception(self, message: str, *args: object) -> None:
        formatted = message % args if args else message
        self.exceptions.append((formatted, args))

    def error(self, message: str) -> None:
        self.errors.append(message)


class _FakeTCPClient:
    def __init__(self, *, fail_send: bool = False) -> None:
        self.fail_send = fail_send
        self.sent_batches: list[list[object]] = []
        self.closed = False

    def send_batch(self, payloads: list[object]) -> None:
        if self.fail_send:
            raise RuntimeError("send failed")
        self.sent_batches.append(payloads)

    def close(self) -> None:
        self.closed = True


class _FakeWriter:
    def __init__(self, *, fail_flush: bool = False) -> None:
        self.fail_flush = fail_flush
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        if self.fail_flush:
            raise RuntimeError("flush failed")


class _FakeDB:
    def __init__(self, writer: _FakeWriter) -> None:
        self.writer = writer


class _FakeSender:
    def __init__(
        self,
        payload: object | None,
        *,
        fail_collect: bool = False,
    ) -> None:
        self.payload = payload
        self.fail_collect = fail_collect
        self.sender = None
        self.rank = None
        self.collect_count = 0

    def collect_payload(self) -> object | None:
        self.collect_count += 1
        if self.fail_collect:
            raise RuntimeError("collect failed")
        return self.payload


class _FakeSampler:
    def __init__(
        self,
        name: str,
        *,
        sender: _FakeSender | None = None,
        writer: _FakeWriter | None = None,
    ) -> None:
        self.sampler_name = name
        self.sender = sender
        self.db = _FakeDB(writer) if writer is not None else None


def test_publisher_attaches_senders_to_tcp_client_and_rank() -> None:
    tcp_client = _FakeTCPClient()
    sender = _FakeSender(payload={"rows": [1]})
    sampler = _FakeSampler("SamplerA", sender=sender)
    publisher = TelemetryPublisher(
        tcp_client=tcp_client,
        rank=3,
        logger=_FakeLogger(),
    )

    publisher.attach_senders([sampler])

    assert sender.sender is tcp_client
    assert sender.rank == 3


def test_publisher_flushes_collects_and_sends_one_batch() -> None:
    tcp_client = _FakeTCPClient()
    writer = _FakeWriter()
    sampler_a = _FakeSampler(
        "SamplerA",
        sender=_FakeSender(payload={"sampler": "a"}),
        writer=writer,
    )
    sampler_b = _FakeSampler(
        "SamplerB",
        sender=_FakeSender(payload=None),
    )
    publisher = TelemetryPublisher(
        tcp_client=tcp_client,
        rank=0,
        logger=_FakeLogger(),
    )

    publisher.publish([sampler_a, sampler_b])

    assert writer.flush_count == 1
    assert tcp_client.sent_batches == [[{"sampler": "a"}]]


def test_publisher_does_not_send_empty_batch() -> None:
    tcp_client = _FakeTCPClient()
    sampler = _FakeSampler(
        "SamplerA",
        sender=_FakeSender(payload=None),
    )
    publisher = TelemetryPublisher(
        tcp_client=tcp_client,
        rank=0,
        logger=_FakeLogger(),
    )

    publisher.publish([sampler])

    assert tcp_client.sent_batches == []


def test_publisher_logs_failures_and_continues() -> None:
    tcp_client = _FakeTCPClient(fail_send=True)
    logger = _FakeLogger()
    good_sampler = _FakeSampler(
        "GoodSampler",
        sender=_FakeSender(payload={"sampler": "good"}),
        writer=_FakeWriter(),
    )
    bad_flush_sampler = _FakeSampler(
        "BadFlushSampler",
        sender=_FakeSender(payload={"sampler": "bad_flush"}),
        writer=_FakeWriter(fail_flush=True),
    )
    bad_collect_sampler = _FakeSampler(
        "BadCollectSampler",
        sender=_FakeSender(payload=None, fail_collect=True),
    )
    publisher = TelemetryPublisher(
        tcp_client=tcp_client,
        rank=0,
        logger=logger,
    )

    publisher.publish([good_sampler, bad_flush_sampler, bad_collect_sampler])

    messages = [entry[0] for entry in logger.exceptions]
    assert len(messages) == 3
    assert any("writer.flush failed" in message for message in messages)
    assert any("collect_payload failed" in message for message in messages)
    assert any("send_batch failed" in message for message in messages)


def test_publisher_close_delegates_to_tcp_client() -> None:
    tcp_client = _FakeTCPClient()
    publisher = TelemetryPublisher(
        tcp_client=tcp_client,
        rank=0,
        logger=_FakeLogger(),
    )

    publisher.close()

    assert tcp_client.closed is True
