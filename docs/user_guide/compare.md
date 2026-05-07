# Compare Runs

Use `traceml compare` to compare two TraceML final summary JSON files from two different runs.

This is the cleanest way to answer questions like:

- did the run get slower or faster?
- did the diagnosis change?
- did wait share increase?
- did memory pressure or skew get worse?

`traceml compare` is designed for comparing finalized run summaries, not raw logs or raw SQLite databases.

---

## What you need

You need two TraceML final summary JSON files.

A common way to produce them is:

```bash
traceml run train.py --mode=summary
```

Then call `traceml.final_summary()` near the end of your script.

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

## What the compare output shows

The compare report is designed to stay compact and useful.

It typically focuses on:

- overall duration
- step-time diagnosis changes
- average step time changes
- wait-share changes
- step split shifts across dataloader, forward, backward, and optimizer
- memory changes when they are meaningful
- process or system changes when they add useful context

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

1. run TraceML in summary mode for each run you care about
2. save the TraceML final summary JSON file for each run
3. compare two runs with `traceml compare`
4. use the compare output to decide whether a regression looks real and where to dig next

Example:

```bash
traceml run train_a.py --mode=summary
traceml run train_b.py --mode=summary
traceml compare run_a.json run_b.json
```

This is often enough to tell whether the slowdown is coming from:

- more compute time
- more wait time
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

This helps keep compare useful across incremental TraceML releases.

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
- [How to Read TraceML Output](how-to-read-output.md)
- [Use TraceML with W&B / MLflow](use-with-wandb-mlflow.md)
- [FAQ](faq.md)
