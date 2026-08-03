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

## PR4 canonical-analyzer comparison

PR4 removes the flat-row regrouping and nested per-step timing dictionary.
`StepTimeAnalyzer` now consumes `StepTimeSourceRow` objects directly and emits
one typed fact graph. The repository queries and data profiles are unchanged.

Recorded on 2026-08-02 in the same environment and with the same PR3 command:

| Ranks | Stage | SELECTs | Median (ms) | Change from PR3 |
|---:|---|---:|---:|---:|
| 1 | load + decode + analyze | 2 | 8.360 | +3.9% |
| 1 | one live provider | 2 | 8.435 | +3.0% |
| 1 | dashboard Step Time refresh | 4 | 17.063 | +6.2% |
| 1 | final summary | 2 | 6.913 | +6.4% |
| 8 | load + decode + analyze | 2 | 32.318 | +3.1% |
| 8 | one live provider | 2 | 33.440 | +6.3% |
| 8 | dashboard Step Time refresh | 4 | 67.531 | +6.9% |
| 8 | final summary | 2 | 17.031 | +5.9% |
| 32 | load + decode + analyze | 2 | 116.024 | +2.9% |
| 32 | one live provider | 2 | 117.410 | +2.7% |
| 32 | dashboard Step Time refresh | 4 | 241.952 | +4.2% |
| 32 | final summary | 2 | 49.686 | +1.4% |

The largest median movement is 6.9%, below the PR4 review threshold of 10%
and consistent with local run-to-run variation. At 32 logical ranks, live and
summary remain effectively unchanged while the canonical path carries fewer
payload representations.

SQLite loading, JSON decoding, alignment, and window analysis are fused in the
current loader. The benchmark therefore reports that span honestly as one
stage. Diagnosis is measured separately because it already accepts a canonical
in-memory window.

The dashboard row measures only its two Step Time reads, not unrelated System,
Process, or Step Memory work. A production-path characterization test also
proves that one real dashboard tick invokes two distinct Step Time computers.

## PR5 typed-diagnosis comparison

PR5 makes diagnosis consume `StepTimeWindow` and typed rank facts directly.
It no longer projects a nested rank mapping, rebuilds analyzer-owned shares,
or constructs unused rich attribution on normal runtime calls. The new
`StepTimePipeline` facade performs repository load, analysis, and diagnosis
once; PR6 and PR7 migrate the live surfaces onto it.

Recorded on 2026-08-02 with the same environment and command as PR4:

| Ranks | Stage | SELECTs | Median (ms) | Change from PR4 |
|---:|---|---:|---:|---:|
| 1 | diagnose | 0 | 0.031 | n/a (not recorded in PR4 table) |
| 1 | one live provider | 2 | 8.183 | -3.0% |
| 1 | dashboard Step Time refresh | 4 | 16.452 | -3.6% |
| 1 | final summary | 2 | 6.598 | -4.6% |
| 8 | diagnose | 0 | 0.059 | n/a (not recorded in PR4 table) |
| 8 | one live provider | 2 | 32.015 | -4.3% |
| 8 | dashboard Step Time refresh | 4 | 64.481 | -4.5% |
| 8 | final summary | 2 | 15.955 | -6.3% |
| 32 | diagnose | 0 | 0.161 | n/a (not recorded in PR4 table) |
| 32 | one live provider | 2 | 116.096 | -1.1% |
| 32 | dashboard Step Time refresh | 4 | 240.431 | -0.6% |
| 32 | final summary | 2 | 48.110 | -3.2% |

No end-to-end median regressed; the measured paths are 0.6-6.3% faster than
PR4. Against the PR3 diagnosis reference, diagnosis itself is 63-72% faster
because ordinary calls no longer allocate the compatibility rank map or the
unconsumed attribution payload. Query topology remains 2/4/2 for one live
provider, the current two-provider dashboard, and final summary.

## PR6 live-session comparison

PR6 moves live cache and freshness ownership into `LiveStepTimeSession` and
makes the CLI a pure presenter. The repository retains PR5's proven bounded
tail SQL. It compares the selected row cursor and rank universe before JSON
decoding, so an unchanged refresh returns the exact prior immutable analysis
without parsing, analysis, or diagnosis work.

The `one live provider` row below deliberately invalidates the test session's
cache before each repetition. It measures the changed/cold path and guards
against hiding a regression behind reuse. The cache-hit and dashboard rows
measure unchanged persisted windows after warm-up.

Recorded on 2026-08-02 with the same environment and command as PR5:

| Ranks | Stage | SELECTs | Median (ms) | Change from PR5 |
|---:|---|---:|---:|---:|
| 1 | one live provider (cache miss) | 2 | 8.540 | +4.4% |
| 1 | unchanged live cache hit | 2 | 0.442 | -94.6% |
| 1 | dashboard Step Time refresh | 4 | 0.883 | -94.6% |
| 1 | final summary | 2 | 6.655 | +0.9% |
| 8 | one live provider (cache miss) | 2 | 33.643 | +5.1% |
| 8 | unchanged live cache hit | 2 | 3.117 | -90.3% |
| 8 | dashboard Step Time refresh | 4 | 6.173 | -90.4% |
| 8 | final summary | 2 | 16.268 | +2.0% |
| 32 | one live provider (cache miss) | 2 | 120.724 | +4.0% |
| 32 | unchanged live cache hit | 2 | 13.265 | -88.6% |
| 32 | dashboard Step Time refresh | 4 | 26.716 | -88.9% |
| 32 | final summary | 2 | 48.528 | +0.9% |

Every cache-miss and summary median remains within 5.1% of PR5, below the 10%
review threshold. Query topology remains 2/4/2. The unchanged path still reads
the source cursor and run strategy in one SQLite snapshot, but it does not
decode persisted JSON or rebuild semantic facts. Dashboard still owns two
sessions until PR7; the improvement here is reuse within each session, not
cross-presenter consolidation.

## PR7 shared-dashboard comparison

PR7 removes the two dashboard compatibility providers. The NiceGUI driver now
refreshes one `LiveStepTimeSession` and fans the same immutable result to the
hero and diagnostics composer. The composer reuses the diagnosis already in
that result; neither dashboard presenter owns data access, diagnosis, or
last-good state.

Recorded on 2026-08-02 with the same environment and command as PR6:

| Ranks | Stage | SELECTs | Median (ms) | Change from PR6 |
|---:|---|---:|---:|---:|
| 1 | one live provider (cache miss) | 2 | 8.154 | -4.5% |
| 1 | unchanged live cache hit | 2 | 0.443 | +0.2% |
| 1 | shared dashboard refresh | 2 | 0.430 | -51.3% |
| 1 | final summary | 2 | 6.504 | -2.3% |
| 8 | one live provider (cache miss) | 2 | 32.897 | -2.2% |
| 8 | unchanged live cache hit | 2 | 3.147 | +1.0% |
| 8 | shared dashboard refresh | 2 | 3.044 | -50.7% |
| 8 | final summary | 2 | 15.933 | -2.1% |
| 32 | one live provider (cache miss) | 2 | 117.048 | -3.0% |
| 32 | unchanged live cache hit | 2 | 12.884 | -2.9% |
| 32 | shared dashboard refresh | 2 | 12.985 | -51.4% |
| 32 | final summary | 2 | 48.162 | -0.8% |

The dashboard row is an unchanged-source refresh after warm-up. Its roughly
50% reduction is the direct result of replacing two live-session refreshes
with one; it is not a claim about the entire dashboard tick. When the source
cursor changes, dashboard Step Time has the same cost as the single
cache-miss row because it uses that same session path. Cache-miss and summary
medians do not regress against PR6. Query topology is now 2/2/2 for one live
provider, the dashboard, and final summary.

These fixtures contain 1, 8, or 32 logical rank streams in SQLite. They do not
use that number of GPUs or measure training throughput.

## PR8/PR9 final-summary migration and cleanup

PR8 moves final summary onto the shared pipeline's summary profile and makes
reporting a pure projection. PR9 removes the displaced utility, loader,
reporting-model, formatter, rank-map, and internal compatibility paths.

Recorded on 2026-08-03 with Python 3.13.5, SQLite 3.50.2, macOS arm64, 400
rows per logical rank, a 100-step window, two warmups, and seven repetitions:

| Ranks | Stage | SELECTs | Median (ms) | Comparison with PR7 |
|---:|---|---:|---:|---:|
| 1 | one live provider (cache miss) | 2 | 7.635 | -6.4% |
| 1 | shared dashboard refresh | 2 | 0.452 | +5.1% |
| 1 | final summary | 2 | 5.716 | -12.1% |
| 8 | one live provider (cache miss) | 2 | 32.622 | -0.8% |
| 8 | shared dashboard refresh | 2 | 3.117 | +2.4% |
| 8 | final summary | 2 | 14.997 | -5.9% |
| 32 | one live provider (cache miss) | 2 | 116.708 | -0.3% |
| 32 | shared dashboard refresh | 2 | 12.962 | -0.2% |
| 32 | final summary | 2 | 46.937 | -2.5% |

Query topology stays 2/2/2. The 1-rank summary improved by more than 10%, but
the absolute movement is below one millisecond. Every measured regression is
below 6%, and the 32-rank cache-miss and summary paths remain within 3% of
PR7. This comparison supports only the conclusion that cleanup introduced no
repeatable material regression.

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
