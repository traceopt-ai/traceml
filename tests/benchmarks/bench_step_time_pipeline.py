# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Reproducible observational benchmark for the Step Time pipeline.

This is intentionally a standalone script rather than a CI performance gate.
It reports current query counts and median/p95 wall time for the existing
loader, diagnosis, live, dashboard, and final-summary paths.
"""

from __future__ import annotations

import argparse
import math
import platform
import sqlite3
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tests.step_time.scenarios import (  # noqa: E402
    BALANCED_PROFILE,
    SQLiteSelectRecorder,
    StepTimeScenario,
    create_step_time_database,
)
from traceml_ai.aggregator.sqlite_writers.step_time import (  # noqa: E402
    init_schema as init_step_time_schema,
)
from traceml_ai.diagnostics.step_time.api import (  # noqa: E402
    diagnose_step_time_window,
)
from traceml_ai.diagnostics.step_time.policy import (  # noqa: E402
    LIVE_STEP_TIME_POLICY,
)
from traceml_ai.reporting.sections.step_time import (  # noqa: E402
    StepTimeSummarySection,
)
from traceml_ai.step_time.analysis import StepTimeAnalyzer  # noqa: E402
from traceml_ai.step_time.model import StepTimeLoadRequest  # noqa: E402
from traceml_ai.step_time.pipeline import LiveStepTimeSession  # noqa: E402
from traceml_ai.step_time.sqlite import (  # noqa: E402
    SQLiteStepTimeRepository,
)


@dataclass(frozen=True)
class Measurement:
    """One timing and query-count result printed by the benchmark."""

    stage: str
    median_ms: float
    p95_ms: float
    selects: int


def _measure(
    stage: str,
    call: Callable[[], object],
    *,
    warmups: int,
    repetitions: int,
) -> Measurement:
    for _ in range(warmups):
        call()

    samples: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1000.0)

    ordered = sorted(samples)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    recorder = SQLiteSelectRecorder()
    with patch.object(sqlite3, "connect", recorder.connect):
        call()
    return Measurement(
        stage=stage,
        median_ms=statistics.median(samples),
        p95_ms=ordered[p95_index],
        selects=recorder.count,
    )


def _benchmark_rank_count(
    root: Path,
    *,
    rank_count: int,
    stored_steps: int,
    window_size: int,
    warmups: int,
    repetitions: int,
) -> list[Measurement]:
    scenario = StepTimeScenario(
        name=f"benchmark_{rank_count}_ranks",
        profiles={rank: dict(BALANCED_PROFILE) for rank in range(rank_count)},
        steps=tuple(range(1, stored_steps + 1)),
    )
    db_path = root / f"step-time-{rank_count}-ranks.db"
    create_step_time_database(db_path, scenario)
    with sqlite3.connect(db_path) as conn:
        # Contract fixtures intentionally use a minimal schema. Performance
        # measurements must include the indexes installed in production.
        init_step_time_schema(conn)
        conn.commit()

    def load_window():
        with sqlite3.connect(db_path) as conn:
            snapshot = SQLiteStepTimeRepository(conn).load_live(
                StepTimeLoadRequest(
                    window_size=window_size,
                    lookback_factor=4,
                )
            )
            return StepTimeAnalyzer().analyze(
                snapshot,
                window_size=window_size,
            )

    loaded = load_window()
    live = LiveStepTimeSession(
        str(db_path),
        request=StepTimeLoadRequest(
            window_size=window_size,
            lookback_factor=4,
        ),
    )
    dashboard = LiveStepTimeSession(
        str(db_path),
        request=StepTimeLoadRequest(
            window_size=window_size,
            lookback_factor=4,
        ),
    )
    summary = StepTimeSummarySection(max_rows=window_size)

    def live_cache_miss():
        live._last_observed = None
        return live.refresh()

    calls = (
        ("load + decode + analyze", load_window),
        (
            "diagnose",
            lambda: diagnose_step_time_window(
                loaded,
                policy=LIVE_STEP_TIME_POLICY,
                training_strategy=loaded.training_strategy,
            ),
        ),
        ("one live provider", live_cache_miss),
        ("unchanged live cache hit", live.refresh),
        ("dashboard shared Step Time refresh", dashboard.refresh),
        ("final summary", lambda: summary.build(str(db_path))),
    )
    return [
        _measure(
            stage,
            call,
            warmups=warmups,
            repetitions=repetitions,
        )
        for stage, call in calls
    ]


def main() -> None:
    """Run the benchmark and print a Markdown-ready baseline table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranks",
        default="1,8,32",
        help="Comma-separated rank counts (default: 1,8,32)",
    )
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()

    rank_counts = tuple(
        int(value.strip()) for value in args.ranks.split(",") if value.strip()
    )
    if not rank_counts or any(rank <= 0 for rank in rank_counts):
        parser.error("--ranks must contain positive integers")
    if min(args.steps, args.window_size, args.repetitions) <= 0:
        parser.error("--steps, --window-size, and --repetitions must be > 0")
    if args.warmups < 0:
        parser.error("--warmups must be >= 0")

    print(f"Python: {platform.python_version()}")
    print(f"SQLite: {sqlite3.sqlite_version}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(
        f"Rows: {args.steps}/rank; window: {args.window_size}; "
        f"warmups: {args.warmups}; repetitions: {args.repetitions}"
    )
    print()
    print("| Ranks | Stage | SELECTs | Median (ms) | p95 (ms) |")
    print("|---:|---|---:|---:|---:|")

    with tempfile.TemporaryDirectory(prefix="traceml-step-time-bench-") as td:
        root = Path(td)
        for rank_count in rank_counts:
            for result in _benchmark_rank_count(
                root,
                rank_count=rank_count,
                stored_steps=args.steps,
                window_size=args.window_size,
                warmups=args.warmups,
                repetitions=args.repetitions,
            ):
                print(
                    f"| {rank_count} | {result.stage} | {result.selects} | "
                    f"{result.median_ms:.3f} | {result.p95_ms:.3f} |"
                )


if __name__ == "__main__":
    # Keep plain ``python tests/benchmarks/bench_step_time_pipeline.py``
    # working from the repository root without requiring an editable install.
    sys.exit(main())
