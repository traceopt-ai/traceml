# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Freshness and unchanged-cursor contracts for the live Step Time session."""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from tests.step_time.factories import window_from_rank_averages
from tests.step_time.scenarios import (
    BALANCED_PROFILE,
    SQLiteSelectRecorder,
    StepTimeScenario,
    create_step_time_database,
)
from traceml_ai.diagnostics.step_time import (
    LIVE_STEP_TIME_POLICY,
    diagnose_step_time_window,
)
from traceml_ai.step_time.model import (
    StepTimeLoadRequest,
    StepTimeMetric,
    StepTimeRepositorySnapshot,
    StepTimeSourceCursor,
    StepTimeWindow,
)
from traceml_ai.step_time.pipeline import (
    LiveStepTimeSession,
    StepTimeAnalysis,
)

_REQUEST = StepTimeLoadRequest(window_size=2, lookback_factor=4)


def _analysis(*, metrics: bool, cursor: int = 1) -> StepTimeAnalysis:
    metric = StepTimeMetric(
        metric="step_time",
        series=None,
        median_total=10.0,
        worst_total=10.0,
        worst_rank=0,
        skew_pct=0.0,
    )
    window = (
        window_from_rank_averages(
            {0: {"step_time": 10.0, "total_step": 10.0}},
            expected_ranks=(0,),
            metrics=(metric,),
            steps_used=2,
        )
        if metrics
        else StepTimeWindow()
    )
    snapshot = StepTimeRepositorySnapshot(
        cursor=StepTimeSourceCursor(
            last_row_id=cursor,
            latest_step=cursor,
        )
    )
    return StepTimeAnalysis(
        request=_REQUEST,
        snapshot=snapshot,
        window=window,
        diagnosis=diagnose_step_time_window(
            window,
            policy=LIVE_STEP_TIME_POLICY,
        ),
    )


def test_cold_empty_read_does_not_claim_prior_data(monkeypatch) -> None:
    session = LiveStepTimeSession(":memory:", request=_REQUEST)
    empty = _analysis(metrics=False)
    monkeypatch.setattr(session, "_run_pipeline", lambda: empty)

    result = session.refresh()

    assert result.freshness == "cold"
    assert result.analysis is empty


def test_last_good_bridges_then_expires_with_monotonic_time(
    monkeypatch,
) -> None:
    now = [0.0]
    session = LiveStepTimeSession(
        ":memory:",
        request=_REQUEST,
        stale_ttl_s=30.0,
        monotonic=lambda: now[0],
    )
    good = _analysis(metrics=True)
    empty = _analysis(metrics=False, cursor=2)
    results = iter((good, empty, empty))
    monkeypatch.setattr(session, "_run_pipeline", lambda: next(results))

    live = session.refresh()
    now[0] = 10.0
    bridged = session.refresh()
    now[0] = 31.0
    expired = session.refresh()

    assert live.freshness == "live"
    assert live.analysis is good
    assert bridged.freshness == "bridged"
    assert bridged.analysis is good
    assert expired.freshness == "expired"
    assert expired.analysis is empty
    assert not expired.analysis.window.metrics


def test_unlimited_ttl_keeps_the_last_good_analysis(monkeypatch) -> None:
    now = [0.0]
    session = LiveStepTimeSession(
        ":memory:",
        request=_REQUEST,
        stale_ttl_s=None,
        monotonic=lambda: now[0],
    )
    good = _analysis(metrics=True)
    empty = _analysis(metrics=False, cursor=2)
    results = iter((good, empty))
    monkeypatch.setattr(session, "_run_pipeline", lambda: next(results))

    assert session.refresh().freshness == "live"
    now[0] = 10_000.0
    bridged = session.refresh()

    assert bridged.freshness == "bridged"
    assert bridged.analysis is good


def test_failed_reads_use_the_same_bridge_and_expiry_policy(
    monkeypatch,
) -> None:
    now = [0.0]
    session = LiveStepTimeSession(
        ":memory:",
        request=_REQUEST,
        stale_ttl_s=5.0,
        monotonic=lambda: now[0],
    )
    good = _analysis(metrics=True)

    monkeypatch.setattr(session, "_run_pipeline", lambda: good)
    assert session.refresh().freshness == "live"

    def fail() -> StepTimeAnalysis:
        raise sqlite3.OperationalError("temporary read failure")

    monkeypatch.setattr(session, "_run_pipeline", fail)
    now[0] = 3.0
    bridged = session.refresh()
    now[0] = 6.0
    expired = session.refresh()

    assert bridged.freshness == "bridged"
    assert bridged.analysis is good
    assert expired.freshness == "expired"
    assert not expired.analysis.window.metrics


def _database(tmp_path: Path) -> Path:
    db_path = tmp_path / "live-session.db"
    create_step_time_database(
        db_path,
        StepTimeScenario(
            name="live_session",
            profiles={0: BALANCED_PROFILE},
            steps=(1, 2),
        ),
    )
    return db_path


def test_unchanged_cursor_reuses_analysis_without_recomputing(
    tmp_path: Path,
) -> None:
    db_path = _database(tmp_path)
    session = LiveStepTimeSession(str(db_path), request=_REQUEST)
    first = session.refresh()
    recorder = SQLiteSelectRecorder()

    with (
        patch.object(sqlite3, "connect", recorder.connect),
        patch(
            "traceml_ai.step_time.sqlite.json.loads",
            wraps=json.loads,
        ) as loads,
        patch.object(
            session._analyzer,
            "analyze",
            wraps=session._analyzer.analyze,
        ) as analyze,
        patch(
            "traceml_ai.step_time.pipeline.diagnose_step_time_window"
        ) as diagnose,
    ):
        second = session.refresh()

    assert second.freshness == "live"
    assert second.analysis is first.analysis
    assert recorder.count == 2
    assert loads.call_count == 0
    assert analyze.call_count == 0
    assert diagnose.call_count == 0


def test_changed_cursor_builds_one_new_analysis(tmp_path: Path) -> None:
    db_path = _database(tmp_path)
    session = LiveStepTimeSession(str(db_path), request=_REQUEST)
    first = session.refresh()
    with sqlite3.connect(db_path) as conn:
        events_json = conn.execute(
            "SELECT events_json FROM step_time_samples LIMIT 1"
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO step_time_samples(
                recv_ts_ns, rank, global_rank, step, events_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (3, 0, 0, 3, events_json),
        )
        conn.commit()

    with (
        patch(
            "traceml_ai.step_time.sqlite.json.loads",
            wraps=json.loads,
        ) as loads,
        patch.object(
            session._analyzer,
            "analyze",
            wraps=session._analyzer.analyze,
        ) as analyze,
        patch(
            "traceml_ai.step_time.pipeline.diagnose_step_time_window",
            wraps=diagnose_step_time_window,
        ) as diagnose,
    ):
        second = session.refresh()

    assert second.analysis is not first.analysis
    assert second.analysis.snapshot.cursor.last_row_id == 3
    assert loads.call_count == len(second.analysis.snapshot.rows)
    assert analyze.call_count == 1
    assert diagnose.call_count == 1


def test_concurrent_refreshes_share_one_analysis(tmp_path: Path) -> None:
    session = LiveStepTimeSession(str(_database(tmp_path)), request=_REQUEST)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = executor.map(lambda _: session.refresh(), range(2))
        first, second = tuple(results)

    assert first.freshness == second.freshness == "live"
    assert first.analysis is second.analysis
