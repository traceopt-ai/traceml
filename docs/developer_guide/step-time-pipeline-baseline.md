# Step Time Pipeline Baseline

This page records the pre-refactor Step Time cost and provider topology. The
numbers are observational evidence, not CI pass/fail budgets. Re-run the
benchmark on the same machine when comparing a later pipeline PR.

## Recorded environment

| Field | Value |
|---|---|
| Baseline commit | `ed130406a21a1214b985624f1b364cff0cbecd7d` |
| Recorded | 2026-08-01 |
| Python | 3.13.5 |
| SQLite | 3.50.2 |
| Platform | macOS, arm64 |
| Rows | 400 per rank |
| Live/final window | 100 aligned steps |
| Live lookback | 4× window |
| Warmups / repetitions | 2 / 7 |

## What is counted

Test-side `sqlite3.Connection.set_trace_callback` instrumentation counts only
statements beginning with `SELECT` or `WITH`. Fixture creation, inserts,
transactions, and pragmas are excluded.

The pre-refactor formulas are:

| Path | SELECTs | Reason |
|---|---:|---|
| One live provider | `R + 2` | rank discovery, one bounded read per rank, and training-strategy context |
| One dashboard refresh | `2R + 4` | two independent live providers perform the same read topology |
| Final summary | `2R + 3` | canonical load, maximum step, and one identity read per rank |

These formulas characterize current behavior. They are expected to change in
later consolidation PRs and must not be treated as desirable long-term API
contracts.

## Recorded results

| Ranks | Stage | SELECTs | Median (ms) | p95 (ms) |
|---:|---|---:|---:|---:|
| 1 | load + decode + analyze | 3 | 8.350 | 8.565 |
| 1 | diagnose | 0 | 0.111 | 0.113 |
| 1 | one live provider | 3 | 8.362 | 8.553 |
| 1 | dashboard Step Time refresh | 6 | 16.591 | 16.862 |
| 1 | final summary | 5 | 7.454 | 7.547 |
| 8 | load + decode + analyze | 10 | 36.482 | 41.452 |
| 8 | diagnose | 0 | 0.183 | 0.196 |
| 8 | one live provider | 10 | 37.128 | 42.099 |
| 8 | dashboard Step Time refresh | 20 | 73.372 | 79.595 |
| 8 | final summary | 19 | 30.479 | 31.420 |
| 32 | load + decode + analyze | 34 | 176.220 | 182.028 |
| 32 | diagnose | 0 | 0.444 | 0.706 |
| 32 | one live provider | 34 | 176.025 | 179.283 |
| 32 | dashboard Step Time refresh | 68 | 371.727 | 422.793 |
| 32 | final summary | 67 | 193.767 | 216.140 |

## PR3 repository comparison

PR3 keeps one repository and snapshot contract but uses two explicit data
profiles. Terminal and dashboard use an index-bounded per-rank tail; final
summary adds identity and progress in its source statement. Both perform one
source statement and one run-context statement inside a read snapshot. The
dashboard still invokes the live profile twice until provider consolidation.

The original PR1 fixture did not install the production SQLite indexes. PR3's
benchmark now calls the production Step Time schema initializer after fixture
creation so SQL measurements represent deployed databases. The numbers below
therefore replace, rather than compare directly with, the earlier unindexed
wall-time table. Query-count formulas remain directly comparable.

Recorded on 2026-08-02 from `abhinav/step_time/edit3`: Python 3.13.5,
SQLite 3.50.2, macOS arm64, 400 stored steps per logical rank, a 100-step
window, 4x live lookback, three warmups, and 25 measured repetitions.

| Ranks | Stage | SELECTs | Median (ms) | p95 (ms) |
|---:|---|---:|---:|---:|
| 1 | load + decode + analyze | 2 | 8.047 | 8.229 |
| 1 | diagnose | 0 | 0.111 | 0.115 |
| 1 | one live provider | 2 | 8.192 | 8.338 |
| 1 | dashboard Step Time refresh | 4 | 16.063 | 16.341 |
| 1 | final summary | 2 | 6.499 | 7.090 |
| 8 | load + decode + analyze | 2 | 31.341 | 36.755 |
| 8 | diagnose | 0 | 0.194 | 0.200 |
| 8 | one live provider | 2 | 31.464 | 36.436 |
| 8 | dashboard Step Time refresh | 4 | 63.154 | 68.694 |
| 8 | final summary | 2 | 16.077 | 16.237 |
| 32 | load + decode + analyze | 2 | 112.759 | 119.259 |
| 32 | diagnose | 0 | 0.433 | 0.466 |
| 32 | one live provider | 2 | 114.312 | 122.680 |
| 32 | dashboard Step Time refresh | 4 | 232.248 | 235.576 |
| 32 | final summary | 2 | 49.011 | 53.103 |

These are synthetic SQLite workloads: “32 ranks” means 32 logical rank streams
in one database, not a 32-GPU machine or training-throughput measurement.

The query count is independent of rank count. A same-schema PR2 reference run
recorded live medians of 8.006, 30.876, and 114.754 ms at 1, 8, and 32 ranks.
PR3 is within 2.3% at the smaller fixtures and 0.4% faster at 32 ranks, which
is within run-to-run noise rather than a material regression. At 10,000 stored
steps per rank, an interleaved repository comparison measured PR3 0.3%, 2.8%,
and 4.2% faster at 1, 8, and 32 ranks. This confirms that live cost follows
the 400-row lookback instead of total run length.

Final summary is faster because maximum-step and identity reads no longer run
per rank. On the same indexed 32-rank fixture, its median moved from 71.253 ms
on the PR2 reference to 49.011 ms in PR3.

SQLite loading, JSON decoding, alignment, and window analysis are fused in the
current loader. The benchmark therefore reports that span honestly as one
stage. Diagnosis is measured separately because it already accepts a canonical
in-memory window.

The dashboard row measures only its two Step Time reads, not unrelated System,
Process, or Step Memory work. A production-path characterization test also
proves that one real dashboard tick invokes two distinct Step Time computers.

## Reproduce

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python tests/benchmarks/bench_step_time_pipeline.py
```

Useful options:

```bash
python tests/benchmarks/bench_step_time_pipeline.py \
  --ranks 1,8,32 \
  --steps 400 \
  --window-size 100 \
  --warmups 2 \
  --repetitions 7
```

When publishing a comparison, record the commit, Python and SQLite versions,
platform, fixture shape, and full command. Compare query counts exactly;
interpret wall time only against a run from a comparable environment.
