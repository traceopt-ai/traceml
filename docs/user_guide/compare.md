# Compare Runs

Compare two training runs to see whether Step Time improved, where time
shifted, and whether memory or the bottleneck diagnosis changed.

## 1. Save two runs

Run your reference code, then your candidate code:

```bash
traceml run train.py --run-name reference
# Switch to the code or configuration you want to evaluate.
traceml run train.py --run-name candidate
```

Summary mode is the default. Each run saves `final_summary.json` in its run
folder when telemetry finalization succeeds. Use a fresh name for each launch.
Your script needs a supported automatic trainer or the required
[explicit integration](integrations.md).

For a useful performance comparison, keep hardware, data, batch size,
precision, and process topology consistent except for the change you are
investigating. Repeat runs when the difference is small or results vary.

Already have two saved summaries? Start with the command below.

## 2. Compare the summaries

```bash
traceml compare \
  logs/reference/final_summary.json \
  logs/candidate/final_summary.json
```

The first file is **A** (reference); the second is **B** (candidate). TraceML
prints a comparison and saves:

```text
compare/reference_vs_candidate.json
compare/reference_vs_candidate.txt
```

The default names come from the input filenames, or their parent folders when
both files are named `final_summary.json`. Compare uses the saved JSON reports;
it does not need the original training environment, raw logs, or databases.

## 3. Read the result

Start with **Verdict** and **Why**, then inspect the measurements behind them.
The report shows A, B, and **Delta = B − A**. For Step Time, a positive delta
means the candidate is slower; a negative delta means it is faster.

| Evidence | What it helps you understand |
| --- | --- |
| Step Time | Whether the average measured step became slower or faster. |
| Input, H2D, Compute, Residual | Where training time shifted. |
| Diagnosis changes | Whether the likely bottleneck changed between runs. |
| Peak reserved memory and memory skew | Whether memory use or rank imbalance increased. |
| Process and system measurements | CPU, RAM, and GPU context for the change. |

The text report groups forward, backward, and optimizer timing into **Compute**.
The comparison JSON retains those individual phases for closer inspection.
DataLoader Fetch (CPU) is supplemental evidence; it is not added to Input Wait
or Step Time.

Missing measurements appear as unavailable rather than zero. Read the report's
notes when evidence is partial or the timing clocks differ. For the meaning of
individual measurements, see [Reading the Output](reading-output.md).

## Choose an output name

Use `--output` to choose the base path for both artifacts:

```bash
traceml compare \
  logs/reference/final_summary.json \
  logs/candidate/final_summary.json \
  --output artifacts/reference-vs-candidate
```

This writes `artifacts/reference-vs-candidate.json` and
`artifacts/reference-vs-candidate.txt`. You can also store the original
summaries as [W&B or MLflow artifacts](integrations/wandb-mlflow.md) and compare
them later.

## Use compare in CI

Ordinary comparison reports changes without enforcing a CI threshold. To fail
CI on a Step Time regression, first configure both runs using
[Regression Guard](regression-guard.md), then add a threshold:

```bash
traceml compare \
  logs/reference/final_summary.json \
  logs/candidate/final_summary.json \
  --max-step-time-regression-pct 5 \
  --output compare/reference-vs-candidate
```

The experimental CI policy requires matching guard declarations and expected
topology, successful training and launcher completion, and positive Step Time
on a common CPU or GPU clock. Only Step Time determines this CI result;
memory and the other measurements remain context.

| Result | Exit code | Meaning |
| --- | ---: | --- |
| `SLOWER_IN_THIS_PAIR` | 4 | Candidate Step Time increased by more than the threshold. |
| `FASTER_IN_THIS_PAIR` | 0 | Candidate Step Time decreased by more than the threshold. |
| `WITHIN_THRESHOLD_IN_THIS_PAIR` | 0 | Difference is within or exactly on the threshold boundaries. |
| `INCONCLUSIVE` | 3 | Required evidence is missing or incompatible. |

The threshold must be finite and nonnegative. Invalid input or output failures
return `1`; invalid command-line usage returns `2`. Evaluated CI results,
including inconclusive ones, are saved in the JSON and text artifacts.

The percentage change is `100 × (candidate − reference) / reference`.
The decision uses aggregate Step Time from each saved summary. Guard measurement
fields declare an intended range; they do not select the analyzed steps or
require identical analyzed-step counts. The report shows both counts.

This is a decision about one pair of runs, not a statistical significance test.
Use repeated reference runs to understand normal variation before choosing a
threshold. See [Regression Guard](regression-guard.md) for setup and a CI example.

## Compatibility and missing fields

Compare requires valid TraceML summary JSON with a schema version and the
required System, Process, Step Time, and Step Memory sections. Missing optional
measurements are allowed; missing required sections or malformed JSON are
rejected. Different schema versions produce a warning.

Step Time uses GPU timing when both runs have it, otherwise CPU timing when
both have it. Without a common measured clock, Step Time comparison is
unavailable. Phase comparisons require matching diagnosis clocks, so CPU and
GPU phases are never mixed. Missing or `null` measurements have no delta.

<details markdown="1">
<summary>Comparing older summary schemas</summary>

Schema 1.7 provides explicit CPU and GPU outer Step Time measurements. Older
outer timing is interpreted as CPU Step Time at the comparison boundary.

Historical `dataloader_ms` is adapted to supplemental CPU fetch timing. It is
never substituted for Input Wait and does not affect Step Time verdicts or
phase shares.

</details>

## Related guides

- [Reading the Output](reading-output.md)
- [Regression Guard](regression-guard.md)
- [W&B / MLflow](integrations/wandb-mlflow.md)
