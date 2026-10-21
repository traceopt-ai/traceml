# Copyright 2026 OptAI UG (haftungsbeschraenkt)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import traceml_ai.samplers.stdout_stderr_sampler as sampler_module
from traceml_ai.aggregator.sqlite_writers import stdout_stderr
from traceml_ai.samplers.stdout_stderr_sampler import StdoutStderrSampler
from traceml_ai.telemetry.envelope import normalize_telemetry_envelope


class _Capture:
    def read_buffer(self) -> str:
        return "first line\n\nsecond line\n"


def test_stdout_drain_time_survives_transport_and_projection(
    monkeypatch, tmp_path
) -> None:
    timestamp_unix_ns = 1_700_000_000_000_000_001
    records = []
    sampler = StdoutStderrSampler.__new__(StdoutStderrSampler)
    sampler._ctx = SimpleNamespace(local_rank=0)
    sampler._add_record = records.append
    sampler.log_path = tmp_path / "stdout_stderr.log"
    monkeypatch.setattr(
        sampler_module.StreamCapture, "_stdout_stderr_capture", _Capture()
    )
    monkeypatch.setattr(
        sampler_module.time, "time_ns", lambda: timestamp_unix_ns
    )

    sampler.sample()

    assert records == [
        {"timestamp_unix_ns": timestamp_unix_ns, "line": "first line"},
        {"timestamp_unix_ns": timestamp_unix_ns, "line": "second line"},
    ]
    records.extend(
        [
            {"timestamp_unix_ns": 1.5, "line": "float timestamp"},
            {"timestamp_unix_ns": True, "line": "boolean timestamp"},
            {"timestamp_unix_ns": 2**63, "line": "overflowing timestamp"},
            {"line": "missing timestamp"},
        ]
    )
    envelope = normalize_telemetry_envelope(
        {
            "meta": {"sampler": "Stdout/Stderr", "global_rank": 0},
            "body": {"tables": {"stdout_stderr": records}},
        }
    )
    assert envelope is not None
    rows = stdout_stderr.build_rows(envelope, recv_ts_ns=timestamp_unix_ns + 9)
    with sqlite3.connect(":memory:") as conn:
        stdout_stderr.init_schema(conn)
        stdout_stderr.insert_rows(conn, rows)
        stored = conn.execute(
            "SELECT timestamp_unix_ns, line FROM stdout_stderr_samples "
            "ORDER BY id;"
        ).fetchall()

    assert stored == [
        (timestamp_unix_ns, "first line"),
        (timestamp_unix_ns, "second line"),
        (None, "float timestamp"),
        (None, "boolean timestamp"),
        (None, "overflowing timestamp"),
        (None, "missing timestamp"),
    ]
