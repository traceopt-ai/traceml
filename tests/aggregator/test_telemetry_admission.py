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
    stamp = {
        "session_id": "run-a",
        "run_nonce": "abc",
        "session_source": "explicit",
    }

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


# Admission: drop payloads stamped for another run before SQLite.


def _enforcing(tmp_path: Path, **overrides) -> TraceMLSettings:
    return _settings(tmp_path, enforce_session_id=True, **overrides)


def test_foreign_session_is_dropped_and_own_session_kept(tmp_path):
    agg = _make_aggregator(tmp_path, _enforcing(tmp_path))
    own = _envelope(session_id="run-b")
    foreign = _envelope(session_id="run-a")
    unstamped = _envelope()

    assert agg._split_telemetry_payloads([foreign, own, unstamped]) == [
        [own, unstamped]
    ]
    assert agg._split_telemetry_payloads(foreign) == []
    assert agg._split_telemetry_payloads([foreign]) == []
    assert agg._foreign_senders == {("host-a", "4242", "run-a"): 3}


def test_foreign_nonce_is_dropped_for_a_reused_run_name(tmp_path):
    agg = _make_aggregator(tmp_path, _enforcing(tmp_path, run_nonce="nb"))
    own = _envelope(session_id="run-b", run_nonce="nb")
    rerun = _envelope(session_id="run-b", run_nonce="na")
    no_nonce = _envelope(session_id="run-b")

    assert agg._split_telemetry_payloads([rerun, own, no_nonce]) == [
        [own, no_nonce]
    ]
    assert agg._foreign_senders == {("host-a", "4242", "run-b"): 1}


def test_foreign_rank_finished_does_not_finish_a_rank(tmp_path):
    agg = _make_aggregator(tmp_path, _enforcing(tmp_path, run_nonce="nb"))

    agg._split_telemetry_payloads(_rank_finished(session_id="run-a"))
    agg._split_telemetry_payloads(
        [_rank_finished(session_id="run-b", run_nonce="na")]
    )
    assert agg._finished_ranks == {}

    agg._split_telemetry_payloads(
        [_rank_finished(session_id="run-b", run_nonce="nb")]
    )
    assert sorted(agg._finished_ranks) == [0]
    # Control payloads carry no pid.
    assert agg._foreign_senders == {
        ("host-a", None, "run-a"): 1,
        ("host-a", None, "run-b"): 1,
    }


def test_malformed_foreign_stamp_is_dropped_without_raising(tmp_path):
    agg = _make_aggregator(tmp_path, _enforcing(tmp_path))
    foreign = _envelope(session_id="run-a")
    foreign["meta"]["pid"] = [1, 2]

    assert agg._split_telemetry_payloads([foreign]) == []
    assert agg._foreign_senders == {("host-a", "[1, 2]", "run-a"): 1}


def test_session_is_not_enforced_without_the_flag(tmp_path):
    agg = _make_aggregator(tmp_path, _settings(tmp_path, run_nonce="nb"))
    other_session = _envelope(session_id="run-a", run_nonce="nb")
    other_nonce = _envelope(session_id="run-b", run_nonce="na")

    assert agg._split_telemetry_payloads([other_session, other_nonce]) == [
        [other_session]
    ]


class _TCP:
    def __init__(self, messages):
        self._messages = list(messages)

    def poll(self):
        while self._messages:
            yield self._messages.pop(0)

    def wait_for_data(self, timeout):
        return False

    def stop(self):
        return None


class _Writer:
    def __init__(self):
        self.ingested = []

    def ingest(self, payload):
        self.ingested.append(payload)

    def finalize(self, timeout_sec):
        from traceml_ai.aggregator.sqlite_writer import SQLiteFinalizeResult

        return SQLiteFinalizeResult(
            ok=True,
            elapsed_sec=0.0,
            enqueued=0,
            written=0,
            dropped=0,
            queue_size=0,
            checkpoint_ok=True,
            error=None,
        )

    def stats(self):
        return {}


class _Stopped:
    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None

    def stop(self):
        return None


def test_stop_prints_one_warning_naming_foreign_senders(tmp_path, capsys):
    settings = _enforcing(tmp_path, history_enabled=False)
    agg = _make_aggregator(tmp_path, settings)
    agg._thread = _Stopped()
    agg._display_driver = _Stopped()
    agg._sqlite_writer = _Writer()
    other = dict(_envelope(session_id="run-c"))
    other["meta"] = dict(other["meta"], hostname="host-c", pid=7)
    agg._tcp_server = _TCP(
        [
            [_envelope(session_id="run-a"), _envelope(session_id="run-b")],
            [_envelope(session_id="run-a"), other],
            _rank_finished(session_id="run-a"),
        ]
    )

    agg._drain_tcp()
    agg.stop(timeout_sec=1.0)

    assert len(agg._sqlite_writer.ingested) == 1
    warnings = [
        line
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("[TraceML]")
    ]
    assert len(warnings) == 1
    assert "ignored 4 payload(s) from another TraceML run" in warnings[0]
    assert "host=host-a pid=4242 session=run-a (2)" in warnings[0]
    assert "host=host-a pid=? session=run-a (1)" in warnings[0]
    assert "host=host-c pid=7 session=run-c (1)" in warnings[0]


def test_stop_prints_nothing_without_foreign_payloads(tmp_path, capsys):
    agg = _make_aggregator(
        tmp_path, _enforcing(tmp_path, history_enabled=False)
    )
    agg._thread = _Stopped()
    agg._display_driver = _Stopped()
    agg._sqlite_writer = _Writer()
    agg._tcp_server = _TCP([[_envelope(session_id="run-b")]])

    agg._drain_tcp()
    agg.stop(timeout_sec=1.0)

    assert "[TraceML]" not in capsys.readouterr().err


def test_foreign_rows_never_reach_sqlite_over_the_real_wire(tmp_path):
    """Own rows land intact; foreign rows are absent, not zeroed."""
    import sqlite3
    import time

    from traceml_ai.aggregator.sqlite_writer import (
        SQLiteWriterConfig,
        SQLiteWriterSimple,
    )
    from traceml_ai.samplers.schema.system import SystemSample
    from traceml_ai.transport.tcp_transport import (
        TCPClient,
        TCPConfig,
        TCPServer,
    )

    def system_envelope(rank: int, cpu: float, session: str) -> dict:
        sample = SystemSample(
            sample_idx=1,
            timestamp=time.time(),
            cpu_percent=cpu,
            ram_used=1.0,
            ram_total=8.0,
            gpu_available=False,
            gpu_count=0,
            gpus=[],
        )
        return build_telemetry_envelope(
            identity=SenderIdentity(
                global_rank=rank,
                local_rank=rank,
                hostname="test-host",
                pid=100 + rank,
                session_id=session,
                run_nonce="nb",
            ),
            sampler_name="SystemSampler",
            tables={"SystemTable": [sample.to_wire()]},
        )

    server = TCPServer(TCPConfig(host="127.0.0.1", port=0))
    server.start()
    writer = SQLiteWriterSimple(
        SQLiteWriterConfig(
            path=str(tmp_path / "telemetry"), flush_interval_sec=0.05
        )
    )
    writer.start()
    client = TCPClient(TCPConfig(host="127.0.0.1", port=server.port))
    try:
        agg = _make_aggregator(tmp_path, _enforcing(tmp_path, run_nonce="nb"))
        agg._tcp_server = server
        agg._sqlite_writer = writer

        client.send_batch(
            [
                system_envelope(0, 11.0, "run-b"),
                system_envelope(1, 99.0, "run-a"),
            ]
        )
        assert server.wait_for_data(timeout=2.0)
        agg._drain_tcp()
        assert writer.force_flush(timeout_sec=2.0)

        with sqlite3.connect(str(tmp_path / "telemetry")) as conn:
            rows = conn.execute(
                "SELECT global_rank, cpu_percent FROM system_samples"
            ).fetchall()
        assert rows == [(0, 11.0)]
    finally:
        client.close()
        server.stop()
        writer.finalize(timeout_sec=5.0)


# `traceml serve --run-name X` enforces explicit session ids only.


def _serve_settings(tmp_path: Path, monkeypatch, *extra: str):
    from traceml_ai.launcher.cli import build_parser
    from traceml_ai.launcher.commands import _resolve_serve_settings

    monkeypatch.chdir(tmp_path)
    for var in ("TRACEML_EXPECTED_WORLD_SIZE", "TRACEML_UI_MODE"):
        monkeypatch.delenv(var, raising=False)
    return _resolve_serve_settings(
        build_parser().parse_args(["serve", *extra])
    )


def test_serve_admits_generated_worker_and_does_not_wait_on_it(
    tmp_path, monkeypatch
):
    """A plain `python train.py` creates its own id and must be traced."""
    import time

    settings = _serve_settings(tmp_path, monkeypatch, "--run-name", "X")
    agg = _make_aggregator(tmp_path, settings)
    worker = _envelope(session_id="session_1", session_source="generated")
    done = _rank_finished(session_id="session_1", session_source="generated")
    agg._tcp_server = _TCP([[worker, done]])
    agg._sqlite_writer = _Writer()

    started = time.monotonic()
    warning = agg._settle_end_of_run_telemetry(timeout_sec=30.0)

    assert warning is None
    assert time.monotonic() - started < 5.0
    assert agg._sqlite_writer.ingested == [[worker]]
    assert sorted(agg._finished_ranks) == [0]
    assert agg._foreign_senders == {}


def test_serve_drops_a_different_explicit_session(tmp_path, monkeypatch):
    settings = _serve_settings(tmp_path, monkeypatch, "--run-name", "X")
    agg = _make_aggregator(tmp_path, settings)
    other = _envelope(session_id="Y", session_source="explicit")
    other_done = _rank_finished(session_id="Y", session_source="explicit")
    # A stamped session without a source is treated as explicit.
    no_source = _envelope(session_id="Y")

    assert agg._split_telemetry_payloads([other, other_done, no_source]) == []
    assert agg._finished_ranks == {}
    assert agg._foreign_senders == {
        ("host-a", "4242", "Y"): 2,
        ("host-a", None, "Y"): 1,
    }


def test_serve_admits_its_own_explicit_session(tmp_path, monkeypatch):
    settings = _serve_settings(tmp_path, monkeypatch, "--run-name", "X")
    agg = _make_aggregator(tmp_path, settings)
    own = _envelope(session_id="X", session_source="explicit")

    assert agg._split_telemetry_payloads([own]) == [[own]]
    assert agg._foreign_senders == {}


def test_run_aggregator_still_drops_a_generated_foreign_session(tmp_path):
    """`traceml run` keeps strict session enforcement."""
    agg = _make_aggregator(tmp_path, _enforcing(tmp_path))
    generated = _envelope(session_id="session_1", session_source="generated")

    assert agg._split_telemetry_payloads([generated]) == []
