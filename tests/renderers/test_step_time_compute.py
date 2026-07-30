import json
import sqlite3

from traceml_ai.renderers.step_time.compute import StepCombinedComputer


def test_step_time_compute_uses_selected_gpu_diagnosis_clock(
    tmp_path,
) -> None:
    db_path = tmp_path / "telemetry.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE step_time_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rank INTEGER,
                global_rank INTEGER,
                step INTEGER,
                events_json TEXT NOT NULL
            );
            """
        )
        events_1 = {
            "_traceml_internal:dataloader_next": {
                "cuda:0": {
                    "duration_ms": 12.0,
                    "cpu_ms": 12.0,
                    "gpu_ms": 4.0,
                    "is_gpu": False,
                    "n_calls": 1,
                }
            },
            "_traceml_internal:step_time": {
                "cuda:0": {
                    "duration_ms": 60.0,
                    "cpu_ms": 60.0,
                    "gpu_ms": 20.0,
                    "is_gpu": False,
                    "n_calls": 1,
                }
            },
        }
        events_2 = {
            "_traceml_internal:dataloader_next": {
                "cuda:0": {
                    "duration_ms": 18.0,
                    "cpu_ms": 18.0,
                    "gpu_ms": 6.0,
                    "is_gpu": False,
                    "n_calls": 1,
                }
            },
            "_traceml_internal:step_time": {
                "cuda:0": {
                    "duration_ms": 90.0,
                    "cpu_ms": 90.0,
                    "gpu_ms": 40.0,
                    "is_gpu": False,
                    "n_calls": 1,
                }
            },
        }
        conn.execute(
            """
            INSERT INTO step_time_samples(rank, global_rank, step, events_json)
            VALUES (?, ?, ?, ?);
            """,
            (0, 0, 1, json.dumps(events_1)),
        )
        conn.execute(
            """
            INSERT INTO step_time_samples(rank, global_rank, step, events_json)
            VALUES (?, ?, ?, ?);
            """,
            (0, 0, 2, json.dumps(events_2)),
        )
        conn.commit()
    finally:
        conn.close()

    result = StepCombinedComputer(
        db_path=str(db_path),
        window_size=2,
    ).compute_cli()

    metrics = {metric.metric: metric for metric in result.diagnosis_metrics}
    assert "dataloader_fetch" not in metrics
    assert metrics["input_wait"].summary.worst_total == 5.0
    assert metrics["step_time"].summary.worst_total == 30.0
    assert metrics["input_wait"].series is not None
    assert metrics["input_wait"].series.worst == [4.0, 6.0]
    assert metrics["input_wait"].series.sum == [4.0, 6.0]
    assert result.diagnosis_clock == "gpu"
    assert result.training_strategy == "ddp"

    per_rank = result.per_rank_timing[0]
    assert per_rank["input_wait"] == 5.0
    assert per_rank["step_time"] == 30.0


def test_worst_rank_requires_measured_total_step() -> None:
    from traceml_ai.renderers.step_time.compute import _worst_rank_from_window

    # Ranks without a measured total step (input wait never measured)
    # cannot win the status-line worst-rank pick with a fake zero.
    assert (
        _worst_rank_from_window(
            {
                0: {"step_time": 100.0},
                1: {"step_time": 400.0},
            }
        )
        is None
    )
    assert (
        _worst_rank_from_window(
            {
                0: {"step_time": 100.0, "total_step": 105.0},
                1: {"step_time": 400.0},
            }
        )
        == 0
    )


def test_computer_bridges_transients_then_expires(monkeypatch) -> None:
    """The last-ok window is the only flicker/dead-view bridge (issue #259).

    A transient empty compute re-serves the last good metrics with a
    STALE status; once the TTL passes, the empty result comes through
    and had_ok records that data existed before.
    """
    from traceml_ai.renderers.step_time import compute as compute_module
    from traceml_ai.renderers.step_time.schema import (
        StepCombinedTimeCoverage,
        StepCombinedTimeMetric,
        StepCombinedTimeResult,
        StepCombinedTimeSummary,
    )

    metric = StepCombinedTimeMetric(
        metric="step_time",
        clock="cpu",
        series=None,
        summary=StepCombinedTimeSummary(
            window_size=1,
            steps_used=1,
            median_total=10.0,
            worst_total=10.0,
            worst_rank=0,
            skew_ratio=0.0,
            skew_pct=0.0,
        ),
        coverage=StepCombinedTimeCoverage(
            expected_steps=1,
            steps_used=1,
            completed_step=1,
            world_size=1,
            ranks_present=1,
            incomplete=False,
        ),
    )
    good = StepCombinedTimeResult(
        status_message="OK",
        per_rank_timing={0: {"step_time": 10.0, "total_step": 10.0}},
        diagnosis_clock="cpu",
        diagnosis_metrics=[metric],
    )
    empty = StepCombinedTimeResult(status_message="no rows")

    computer = StepCombinedComputer(db_path=":memory:", stale_ttl_s=30.0)
    assert computer.had_ok is False

    results = iter([good, empty, empty])
    monkeypatch.setattr(computer, "_compute_impl", lambda conn: next(results))

    first = computer.compute_cli()
    assert first.diagnosis_metrics == [metric]
    assert computer.had_ok is True
    # The payload itself carries had_ok too (dashboard subscribers only
    # ever see the payload, never the computer instance directly).
    assert first.had_ok is True

    # Within the TTL a transient gap re-serves the last good metrics.
    bridged = computer.compute_cli()
    assert bridged.diagnosis_metrics == [metric]
    assert bridged.status_message.startswith("STALE")
    assert bridged.had_ok is True

    # Past the TTL the empty result comes through unbridged.
    monkeypatch.setattr(
        compute_module.time,
        "time",
        lambda: computer._last_ok_ts + 31.0,
    )
    expired = computer.compute_cli()
    assert expired.diagnosis_metrics == []
    assert expired.status_message == "No fresh step-combined data"
    assert computer.had_ok is True
    assert expired.had_ok is True


def test_dead_run_expires_even_though_sqlite_rows_persist(
    monkeypatch,
) -> None:
    """The dead-run tripwire must actually trip.

    SQLite rows outlive the process that wrote them, so a stopped run
    keeps recomputing the SAME window successfully forever. Before source
    -freshness tracking, that reset the stale timer on every tick and the
    advertised expiry never fired in a real run, only in tests that
    forced an empty compute.
    """
    from traceml_ai.renderers.step_time import compute as compute_module
    from traceml_ai.renderers.step_time.schema import (
        StepCombinedTimeCoverage,
        StepCombinedTimeMetric,
        StepCombinedTimeResult,
        StepCombinedTimeSummary,
    )

    def _result(completed_step: int) -> StepCombinedTimeResult:
        metric = StepCombinedTimeMetric(
            metric="step_time",
            clock="cpu",
            series=None,
            summary=StepCombinedTimeSummary(
                window_size=1,
                steps_used=1,
                median_total=10.0,
                worst_total=10.0,
                worst_rank=0,
                skew_ratio=0.0,
                skew_pct=0.0,
            ),
            coverage=StepCombinedTimeCoverage(
                expected_steps=1,
                steps_used=1,
                completed_step=completed_step,
                world_size=1,
                ranks_present=1,
                incomplete=False,
            ),
        )
        return StepCombinedTimeResult(
            status_message="OK",
            per_rank_timing={0: {"step_time": 10.0, "total_step": 10.0}},
            diagnosis_clock="cpu",
            diagnosis_metrics=[metric],
        )

    computer = StepCombinedComputer(db_path=":memory:", stale_ttl_s=30.0)

    now = [1000.0]
    monkeypatch.setattr(compute_module.time, "time", lambda: now[0])

    # Live run: the window advances, so every tick stays fresh.
    step = [1]
    monkeypatch.setattr(
        computer, "_compute_impl", lambda conn: _result(step[0])
    )
    assert computer.compute_cli().diagnosis_metrics != []
    now[0] += 60.0
    step[0] = 2
    assert computer.compute_cli().diagnosis_metrics != []

    # Producer stops. Rows remain, so _compute_impl keeps succeeding with
    # the SAME completed_step. Within the TTL the view is still served.
    now[0] += 10.0
    assert computer.compute_cli().diagnosis_metrics != []

    # Past the TTL with no advance, the view must expire.
    now[0] += 31.0
    expired = computer.compute_cli()
    assert expired.diagnosis_metrics == []
    assert expired.had_ok is True
    assert expired.status_message == "No fresh step-combined data"

    # And it recovers if the producer comes back.
    step[0] = 3
    assert computer.compute_cli().diagnosis_metrics != []


def test_computer_had_ok_stays_false_on_cold_start(monkeypatch) -> None:
    """A payload that never had good data reports had_ok=False on itself,
    not just on the computer -- distinguishes cold start from a dead run
    for any consumer that only sees the payload (issue #259)."""
    from traceml_ai.renderers.step_time.schema import StepCombinedTimeResult

    computer = StepCombinedComputer(db_path=":memory:", stale_ttl_s=30.0)
    monkeypatch.setattr(
        computer,
        "_compute_impl",
        lambda conn: StepCombinedTimeResult(status_message="no rows"),
    )

    result = computer.compute_cli()
    assert result.diagnosis_metrics == []
    assert result.had_ok is False
    assert computer.had_ok is False
