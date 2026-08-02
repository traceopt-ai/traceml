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

The current formulas are:

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
