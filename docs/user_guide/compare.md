# Compare Runs

Use `traceml compare` to compare two TraceML final summary JSON files from two different runs.

This is the cleanest way to answer questions like:

- did the run get slower or faster?
- did the diagnosis change?
- did residual time increase?
- did memory pressure or skew get worse?

`traceml compare` is designed for comparing finalized run summaries, not raw logs or raw SQLite databases.

---

## What you need

You need two TraceML final summary JSON files.

A common way to produce them is:

```bash
traceml run train.py --mode=summary
```

Summary mode writes `final_summary.json` at the end of the run.

If you are logging TraceML output into W&B or MLflow, you can also keep those summary JSON files as run artifacts and compare them later.

---

## Basic usage

```bash
traceml compare run_a.json run_b.json
```

This compares:

- `A`: the first file you pass
- `B`: the second file you pass

TraceML writes:

- a structured compare JSON
- a compact text report

By default, outputs are written under a local `compare/` directory in the current working directory.

Example:

```text
compare/run_a_vs_run_b.json
compare/run_a_vs_run_b.txt
```

If the file names are generic, such as `final_summary.json`, TraceML falls back to parent directory names when naming the compare artifacts.

---

## Choose an output name

If you want to control the output name, pass `--output`.

```bash
traceml compare run_a.json run_b.json --output=my_compare
```

This writes:

```text
my_compare.json
my_compare.txt
```

You can also pass a path:

```bash
traceml compare run_a.json run_b.json --output=artifacts/baseline_vs_candidate
```

This writes:

```text
artifacts/baseline_vs_candidate.json
artifacts/baseline_vs_candidate.txt
```

---

## Use compare in CI

Add an explicit Step Time threshold when a comparison should produce a CI
decision:

```bash
traceml compare \
  logs/reference/final_summary.json \
  logs/candidate/final_summary.json \
  --max-step-time-regression-pct 5 \
  --output compare/reference-vs-candidate
```

The first summary is the reference and the second is the candidate. The
threshold must be a finite, nonnegative percentage. Without the option,
`traceml compare` keeps its existing exploratory behavior and does not enforce
a CI threshold.

The CI policy requires both summaries to come from compatible guarded runs:

- normalized guard declarations and expected topology must match
- training and launcher completion must be recorded as completed
- both summaries must provide positive Step Time on a common CPU or GPU clock

TraceML reads these facts from the two final summaries. It does not read their
manifests or SQLite databases. Different analyzed-step counts are allowed and
are reported as context. Step Memory and the remaining compare sections also
remain descriptive context; only common-clock Step Time determines the CI
result.

The signed percentage difference is calculated from reference to candidate:

```text
100 * (candidate - reference) / reference
```

| Result | Exit code | Meaning |
| --- | ---: | --- |
| `SLOWER_IN_THIS_PAIR` | 2 | Candidate Step Time increased beyond the threshold. |
| `FASTER_IN_THIS_PAIR` | 0 | Candidate Step Time decreased beyond the threshold. |
| `WITHIN_THRESHOLD_IN_THIS_PAIR` | 0 | The difference is on or within either threshold boundary. |
| `INCONCLUSIVE` | 3 | The summaries are valid but incompatible or lack required evidence. |

Invalid thresholds, malformed input, and output-writing failures use exit code
`1`. TraceML writes the compare JSON and text artifacts before returning an
evaluated result. The decision describes this pair of runs; it does not claim
statistical significance or repeatability.

See [Regression Guard](regression-guard.md) for configuring guarded runs.

---

## What the compare output shows

The compare report is designed to stay compact and useful.

It typically focuses on:

- overall duration
- primary diagnosis changes
- step-time diagnosis changes
- average step time changes
- residual-time changes
- high-level step split shifts across input, H2D, compute, and residual time
- memory changes when they are meaningful
- process or system changes when they add useful context

The text report keeps step-time output compact by showing the aggregate compute
bucket instead of printing forward, backward, and optimizer rows by default.
The structured compare JSON still includes those compute sub-phases for deeper
inspection and downstream tooling.

The text report includes a small legend near the top:

```text
- A: <first run>
- B: <second run>
- Format: A -> B | delta = B - A
```

That means:

- `A -> B` shows the value in the first run and then the second run
- `delta` is computed as `B - A`

---

## Recommended workflow

A good workflow is:

1. run TraceML for each run you care about
2. save the TraceML final summary JSON file for each run
3. compare two runs with `traceml compare`
4. use the compare output to decide whether a regression looks real and where to dig next

Example:

```bash
traceml run train_a.py
traceml run train_b.py
traceml compare run_a.json run_b.json
```

This is often enough to tell whether the slowdown is coming from:

- more compute time
- more residual time
- a phase split change
- worse memory behavior
- a diagnosis shift

---

## What compare is best at today

TraceML compare is currently strongest for comparing:

- step time
- step memory
- process-level context
- selected system-level context

It is best used as a compact run-to-run diagnosis tool.

It is not meant to replace a full experiment tracking system.

Use W&B, MLflow, or TensorBoard for:

- run metadata
- metrics history
- artifacts
- dashboards
- experiment management

Use TraceML compare for:

- bottleneck changes
- diagnosis changes
- performance regressions you want to inspect quickly

---

## Compatibility and missing fields

`traceml compare` is designed to degrade gracefully when fields are missing.

That means:

- if one summary has a field and the other does not, comparison still runs
- if a section is missing, TraceML skips noisy output instead of failing when possible
- if newer TraceML versions add more fields later, older comparisons should still remain usable for the shared fields
- if the two summaries use different schema versions, compare emits a warning because some fields may have changed meaning

This helps keep compare useful across incremental TraceML releases.
Input Wait is compared only as selected-clock `input_wait_ms`; historical
`dataloader_ms` is never substituted for it. Pre-1.7 `dataloader_ms` is
instead adapted to the supplemental CPU `dataloader_fetch_cpu_ms` comparison
field, which does not affect Step Time verdicts or phase shares.
Schema 1.7 publishes explicit CPU and GPU outer Step Time aggregates. Compare
uses GPU when both summaries have measured GPU Step Time; otherwise it uses CPU
when both have measured CPU Step Time. If neither clock is shared, Step Time is
inconclusive. Historical outer timing is interpreted only as CPU Step Time at
the versioned compare boundary. Selected-clock phases still require matching
diagnosis clocks, so CPU and GPU phase values are never mixed. Since schema
1.7, a Step Time metric can be `null` when its timing signal was never measured
in the analyzed window; compare treats a null side as unavailable (no delta)
instead of reading it as zero.

---

## What files should you compare?

Compare:

- TraceML final summary JSON files from completed runs

Do not compare:

- raw database files
- partial logs
- screenshots
- rendered text summaries alone

The JSON file is the stable machine-readable input for compare.

---

## When compare is most useful

Use compare when:

- a training change might have made runs slower
- a new dataloader or preprocessing path may have changed throughput
- a model or optimizer change may have shifted time into a different phase
- memory behavior looks different between two runs
- dashboards look similar but throughput feels worse

---

## Related docs

- [Quickstart](quickstart.md)
- [How to Read TraceML Output](reading-output.md)
- [Use TraceML with W&B / MLflow](integrations/wandb-mlflow.md)
- [FAQ](faq.md)
