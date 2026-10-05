# Catch Training Regressions in CI

Compare a reference training run with a candidate and fail CI when Step Time
increases beyond your threshold.

Regression Guard is experimental. It checks that both runs declared the same
workload and topology, completed successfully, and have comparable Step Time
measurements.

## 1. Declare your workload

Add a `guard` section to `traceml.yaml` in your training project:

```yaml
mode: summary
history_enabled: true

guard:
  schema_version: 1
  workload:
    name: resnet50-imagenet-training
    parameters:
      model: resnet50
      data_version: imagenet-1k-v1
      precision: bf16
      per_rank_batch_size: 32
  measurement:
    start_step: 10
    completed_steps: 50
```

Use the same declaration for both runs. Record parameters that materially
change the workload. TraceML searches the launch directory and its parents
for `traceml.yaml`.

**The measurement fields declare the intended range; they do not select the
steps used for comparison.** The check uses aggregate Step Time from each
saved summary. Different analyzed ranges and step counts are allowed.

## 2. Run the reference and candidate

Run your reference code:

```bash
traceml run train.py --run-name reference
```

Then run your candidate code with the same workload declaration:

```bash
traceml run train.py --run-name candidate
```

Use comparable hardware and the same process topology. Choose a fresh run
name for every execution; guarded runs do not overwrite an existing manifest.

These commands assume your script uses a supported automatic trainer or the
required [explicit integration](integrations.md). Regression Guard requires
`traceml run`, summary mode, and history recording.

For DDP, use the same node count and processes per node in both runs. See
[Multi-node requirements](#multi-node-requirements) for shared storage and
per-node launch configuration.

## 3. Compare the results

```bash
traceml compare \
  logs/reference/final_summary.json \
  logs/candidate/final_summary.json \
  --max-step-time-regression-pct 5 \
  --output compare/reference-vs-candidate
```

The reference comes first. A candidate more than 5% slower fails the check.
TraceML saves JSON and text comparison reports.

Both runs need matching normalized declarations and expected topology,
completed training and launcher outcomes, and positive Step Time on a common
CPU or GPU clock. Only Step Time determines the CI result; other measurements
remain comparison context.

| Result | Exit code | Meaning |
| --- | ---: | --- |
| `WITHIN_THRESHOLD_IN_THIS_PAIR` | 0 | Difference is within the threshold. |
| `FASTER_IN_THIS_PAIR` | 0 | Candidate is faster beyond the threshold. |
| `SLOWER_IN_THIS_PAIR` | 4 | Candidate is slower beyond the threshold. |
| `INCONCLUSIVE` | 3 | Required evidence is missing or incompatible. |

Invalid input or output failures return `1`; invalid command-line usage
returns `2`. Without `--max-step-time-regression-pct`, comparison is
exploratory and does not enforce a CI threshold. See
[Compare Runs](compare.md#use-compare-in-ci) for decision details.

## 4. Add the check to CI

The experimental pilot deliberately leaves reference selection to the user.
Make the chosen reference `final_summary.json` available to the job, run the
candidate, and pass both explicit paths to `traceml compare`. For example:

```yaml
- name: Check Step Time
  run: |
    traceml compare \
      artifacts/reference/final_summary.json \
      logs/candidate/final_summary.json \
      --max-step-time-regression-pct 5 \
      --output compare/reference-vs-candidate

- name: Preserve comparison evidence
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: traceml-performance-comparison
    path: compare/
```

The comparison step succeeds for a faster candidate or a result within the
threshold. It fails for a slower candidate, invalid input, or inconclusive
evidence. See the [exit-code table](compare.md#use-compare-in-ci) when a CI
system needs to distinguish those outcomes.

## 5. Choose a useful threshold

Repeat the reference workload on comparable hardware before choosing a
threshold. Allow for normal variation, and keep the model, data, precision,
batch size, and process topology consistent.

This check describes one pair of runs. It does not establish statistical
significance or equivalent model quality. TraceML checks your declarations;
it does not verify that the training program actually used those parameters.

## Try the complete CPU DDP workflow

<details markdown="1">
<summary>Run a small reference-and-candidate trial</summary>

The checked-in minimal DDP example provides a small end-to-end trial on a
CPU-only machine. From the repository root, create this `traceml.yaml`:

```yaml
mode: summary
history_enabled: true

guard:
  schema_version: 1
  workload:
    name: ddp-minimal-guard-trial
    parameters:
      model: tiny-mlp
      data_version: synthetic-v1
  measurement:
    start_step: 1
    completed_steps: 20
```

Run the same declared workload twice with fresh run names:

```bash
OMP_NUM_THREADS=1 traceml run examples/distributed/ddp_minimal.py \
  --run-name guard-reference \
  --nproc-per-node 2 \
  --args --steps 20

OMP_NUM_THREADS=1 traceml run examples/distributed/ddp_minimal.py \
  --run-name guard-candidate \
  --nproc-per-node 2 \
  --args --steps 20
```

Then evaluate the pair:

```bash
traceml compare \
  logs/guard-reference/final_summary.json \
  logs/guard-candidate/final_summary.json \
  --max-step-time-regression-pct 1000 \
  --output compare/guard-reference-vs-candidate
```

The first summary is always the reference and the second is the candidate.
TraceML writes both JSON and text comparison artifacts before returning the
CI exit code. This tiny CPU workload can vary by tens of percent between
identical runs, so the wide threshold demonstrates the complete workflow rather
than a meaningful performance verdict.

</details>

## Advanced reference

### Configuration fields

`schema_version: 1` identifies the contract format, not a stable 1.0 release.

| Field | Required | Type | Rules |
| --- | --- | --- | --- |
| `guard.schema_version` | Yes | Integer | Must be `1`. |
| `guard.workload.name` | Yes | String | Nonempty, no surrounding whitespace, at most 128 characters. |
| `guard.workload.parameters` | No | Mapping | Defaults to `{}` and may contain at most 32 entries. |
| Parameter key | Yes per entry | String | Nonempty, no surrounding whitespace, at most 64 characters. |
| Parameter value | Yes per entry | Scalar | A nonempty string, signed integer, finite float, or Boolean. Strings may contain at most 256 characters. |
| `guard.measurement.start_step` | Yes | Integer | Declared first completed step; must be at least `1`. |
| `guard.measurement.completed_steps` | Yes | Integer | Declared number of completed steps; must be at least `1`. |

Unknown fields, lists, nested parameter mappings, and null parameter values
are rejected. Integers and the inclusive measurement end must fit in a signed
64-bit value. Workload names, parameter keys, and string values cannot contain
Unicode category C characters, including control and format characters.

### Workload identity

`workload.name` is the only required workload field. Choose a stable name that
identifies the work being compared. TraceML stores it exactly as written, so
surrounding whitespace is rejected instead of being removed silently.

`workload.parameters` is optional. Record values that materially change the
work, such as model, data revision, precision, batch size, sequence length, or
image dimensions. Model and data fields are recommended for training but are
not required because TraceML also supports synthetic and pipeline-focused
workloads.

Parameter values may be strings, integers, finite floating-point values, or
Booleans. Lists, nested mappings, and null values are not supported in schema
1. Two guarded runs are eligible for a CI decision only when their normalized
workload declarations match exactly.

YAML scalar spelling affects the stored type. PyYAML reads `1e-3` as a string,
while `0.001` and `1.0e-3` are floating-point values; `yes`, `no`, `on`, and
`off` are Booleans. It also follows YAML 1.1 integer forms: unquoted `010` is
the integer `8`, and `1:30` is the integer `90`. TraceML accepts those parsed
integers; for clarity, write step fields as ordinary decimal integers such as
`10`. Quote a value such as `"010"` when it is an identifier that must remain a
string. Use an explicit decimal form for floating-point parameters.

These values are user declarations. TraceML records them but does not infer or
verify that the training program used them. Do not include secrets, credentials,
or machine-local paths because the values are persisted in run artifacts.

### Measurement and history retention

TraceML completed steps are numbered from 1. `start_step` is inclusive and
`completed_steps` is the exact requested count. This example:

```yaml
measurement:
  start_step: 10
  completed_steps: 50
```

declares an intended range of steps 10 through 59. If `--trace-max-steps` is
used, it must include the complete requested range.

The complete requested window should remain inside `history_retention` until
the run is finalized. In the experimental pilot, this declaration is a
compatibility fingerprint: two runs must declare the same range, but TraceML
does not compare individual step IDs or use the declaration to re-aggregate
telemetry. The CI decision uses the aggregate Step Time already stored for each
final summary's `step_time.global.window`. That analyzed range can differ from
the requested range or from the other run, and its analyzed-step count remains
visible in the comparison.

### Multi-node requirements

One-process and fixed-size DDP runs on one or more nodes are allowed.
`traceml watch`, `traceml serve`, and direct `traceml.init()` launches ignore
the guard declaration.

For multi-node DDP, use the ordinary launch shape in
[Distributed Training](distributed-training.md#multi-node-ddp). Every launcher
must discover the same `guard` declaration and use the same explicit
`--run-name`, `--nnodes`, and `--nproc-per-node`; only `--node-rank` differs.
Resolve `--logs-dir` to the same shared directory on every node. Node 0 records
the declaration, and each launcher writes its outcome under that directory.

### Saved artifacts and completion checks

<details markdown="1">
<summary>Manifest, node outcomes, and final-summary context</summary>

The launcher validates and normalizes the declaration before starting the
aggregator or training workers. After writing the manifest, it prints a short
`Measurement contract captured` confirmation and stores the result in
`manifest.json`:

```json
{
  "guard": {
    "contract": {
      "schema_version": 1,
      "workload": {
        "name": "resnet50-imagenet-training",
        "parameters": {
          "data_version": "imagenet-1k-v1",
          "model": "resnet50",
          "per_rank_batch_size": 32,
          "precision": "bf16"
        }
      },
      "measurement": {
        "start_step": 10,
        "completed_steps": 50
      }
    }
  }
}
```

The source configuration path is not part of the portable contract. Changing
`traceml.yaml` after launch cannot change the declaration captured for that run.

After its local training process exits, every guarded launcher also writes:

```text
logs/<run-name>/nodes/node_<node-rank>/guard_outcome.json
```

The atomic, versioned record contains the public run name, node topology,
normalized-contract digest, training status and exit code, and completion time.
It excludes command arguments, paths, hostnames, environment contents, device
identifiers, and credentials. Exit code `0` records completed training; any
other observed exit code records failed training.

The fresh run name, topology, and contract digest bind the record to its run.

Node 0 gives outcome collection up to the configured finalization timeout,
then writes a bounded result beside the contract in `manifest.json`. Aggregator
shutdown is a separate bounded phase.

```json
{
  "guard": {
    "training": {
      "status": "completed",
      "nodes_expected": 2,
      "nodes_observed": 2,
      "reasons": [],
      "nodes": [
        {"node_rank": 0, "exit_code": 0},
        {"node_rank": 1, "exit_code": 0}
      ]
    }
  }
}
```

`completed` means every expected launcher reported exit code `0`. Missing,
malformed, conflicting, or nonzero outcomes produce `incomplete` with stable
reason codes. This establishes training-command completion only; it does not
prove telemetry or measurement-window completeness. Collection and recording
failures emit warnings but never replace the supervised training command's
exit code.

| Reason | Meaning |
| --- | --- |
| `node_outcome_missing` | An expected launcher did not report before the timeout. |
| `node_outcome_invalid` | An expected record was malformed, unreadable, or too large. |
| `node_outcome_conflict` | A record's run name, topology, or contract did not match. |
| `node_training_failed` | At least one valid record contained a nonzero exit code. |
| `node_outcome_collection_failed` | Node 0 could not complete collection because of an internal error. |

`guard.training` is independent of the manifest's top-level run and telemetry
statuses. Read all three fields when diagnosing an incomplete run.

The final report copies the portable declaration, expected topology, and
aggregate launcher result into `final_summary.json` under `run_context`.
Internal coordination fields and individual node outcomes remain only in the
manifest and node artifacts. This lets local CI comparison consume one summary
per run without reading SQLite or joining a second artifact. An ordinary run
has no `declaration` or `launcher_completion`; when no readable launcher
manifest exists, `run_context` is an empty object. An interrupted run can omit
`launcher_completion` when its summary finishes before node 0 consolidates the
launcher outcomes; `run.status` still records the interruption.

To inspect the captured declaration, format the manifest and look under
`guard.contract`:

```bash
python -m json.tool logs/reference/manifest.json
```

</details>

### Tested environments

The pilot keeps implementation support separate from environments exercised
end to end in automated tests.

| Environment | Current qualification |
| --- | --- |
| Linux, CPU, two-rank Gloo DDP | A built wheel runs two guarded jobs and compares their summaries in CI. |
| Single-process and ordinary single-node DDP | Covered by launcher, reporting, and comparison tests. |
| Multi-node outcome coordination | Covered on one machine with two launchers and a shared run directory. A real multi-machine environment is not yet qualified. |
| CUDA | Supported by ordinary TraceML paths, but the complete guarded comparison workflow has no dedicated GPU CI qualification yet. |
| Windows | The wheel and CLI surface are smoke tested; the complete guarded DDP workflow is not qualified there yet. |

The comparison is one observation about one explicit pair. It does not claim
repeatability, statistical significance, model-quality equivalence, complete
telemetry delivery, or support for changing process membership.

### Troubleshooting

| Symptom | What to check |
| --- | --- |
| A run refuses to start because its manifest exists | Choose a fresh `--run-name`; guarded runs never overwrite an earlier run. |
| `final_summary.json` is missing | Inspect the training and aggregator output. Comparison requires a finalized summary from each run. |
| The result is `INCONCLUSIVE` because run context is missing | Produce both summaries with guarded `traceml run` executions rather than an ordinary or direct-SDK run. |
| The declarations do not match | Use the same normalized `guard` section for both runs, including parameter types and values. |
| The topology does not match | Use the same node count and processes per node for the reference and candidate. |
| Launcher completion is incomplete | Inspect the per-node launcher output and the root manifest to find the missing, invalid, conflicting, or failed node outcome. |
| Step Time is unavailable on a common clock | Confirm that both finalized summaries contain measured positive CPU Step Time, or measured positive GPU Step Time. |

Preserve both `final_summary.json` files and the generated comparison JSON and
text report when investigating a CI failure. Manifests and SQLite databases
are not inputs to the comparison.
