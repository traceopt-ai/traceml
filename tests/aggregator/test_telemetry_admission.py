# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Aggregator admission of telemetry and control payloads.

A rank that outlives its ``traceml run`` keeps reconnecting to whatever
aggregator listens on the port, so a later run can ingest it. The first block
freezes what the aggregator accepts today, including payloads from older
ranks that carry no run stamp; those must keep landing after admission
filtering exists.
"""

from __future__ import annotations

import threading
from pathlib import Path

from traceml_ai.aggregator.trace_aggregator import TraceMLAggregator
from traceml_ai.runtime.sender import SenderIdentity
from traceml_ai.runtime.settings import TraceMLSettings
from traceml_ai.telemetry.control import (
    build_rank_finished_payload,
    parse_rank_finished,
)
from traceml_ai.telemetry.envelope import (
    build_telemetry_envelope,
    normalize_telemetry_envelope,
)


class _Logger:
    def error(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


def _make_aggregator(tmp_path: Path, settings: TraceMLSettings):
    agg = TraceMLAggregator.__new__(TraceMLAggregator)
    agg._logger = _Logger()
    agg._stop_event = threading.Event()
    agg._settings = settings
    agg._started = True
    agg._expected_world_size = int(settings.expected_world_size)
    agg._finished_ranks = {}
    agg._foreign_senders = {}
    agg._drain_lock = threading.Lock()
    return agg


def _settings(tmp_path: Path, **overrides) -> TraceMLSettings:
    values = {
        "mode": "summary",
        "logs_dir": str(tmp_path),
        "session_id": "run-b",
        "db_path": str(tmp_path / "telemetry"),
        "expected_world_size": 1,
    }
    values.update(overrides)
    return TraceMLSettings(**values)


def _envelope(**stamp) -> dict:
    """A canonical envelope as a rank builds it, plus optional stamp keys."""
    payload = build_telemetry_envelope(
        identity=SenderIdentity(
            global_rank=0, local_rank=0, hostname="host-a", pid=4242
        ),
        sampler_name="SystemSampler",
        tables={"SystemTable": [{"seq": 1}]},
        timestamp=1.0,
    )
    payload["meta"].update(stamp)
    return payload


def _rank_finished(**stamp) -> dict:
    payload = build_rank_finished_payload(
        global_rank=0, world_size=1, node_rank=0, hostname="host-a"
    )
    payload.update(stamp)
    return payload


# Characterization: current accept behaviour that must not change.


def test_unstamped_envelope_batch_is_forwarded(tmp_path):
    agg = _make_aggregator(tmp_path, _settings(tmp_path))
    envelope = _envelope()
    envelope["meta"].pop("session_id", None)
    envelope["meta"].pop("run_nonce", None)

    assert agg._split_telemetry_payloads([envelope]) == [[envelope]]
    assert agg._split_telemetry_payloads(envelope) == [envelope]


def test_unstamped_rank_finished_counts_as_finished(tmp_path):
    agg = _make_aggregator(tmp_path, _settings(tmp_path))
    control = _rank_finished()
    control.pop("session_id", None)
    control.pop("run_nonce", None)

    assert agg._split_telemetry_payloads([control]) == []
    assert sorted(agg._finished_ranks) == [0]


def test_parsers_ignore_stamp_keys(tmp_path):
    """An aggregator that predates the stamp must still parse stamped data."""
    stamp = {"session_id": "run-a", "run_nonce": "abc"}

    envelope = normalize_telemetry_envelope(_envelope(**stamp))
    control = parse_rank_finished(_rank_finished(**stamp))

    assert envelope is not None
    assert envelope.meta.sampler == "SystemSampler"
    assert envelope.meta.pid == 4242
    assert dict(envelope.tables) == {"SystemTable": [{"seq": 1}]}
    assert control is not None
    assert control.global_rank == 0


def test_foreign_stamp_is_forwarded_when_nothing_is_enforced(tmp_path):
    """`traceml serve` with a generated id admits any run's ranks."""
    agg = _make_aggregator(tmp_path, _settings(tmp_path))
    envelope = _envelope(session_id="run-a", run_nonce="abc")
    control = _rank_finished(session_id="run-a", run_nonce="abc")

    assert agg._split_telemetry_payloads([envelope, control]) == [[envelope]]
    assert sorted(agg._finished_ranks) == [0]
