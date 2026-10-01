# Regression guard measurement contract

TraceML's local regression guard is an experimental **v0.1 pilot**. A guarded
run captures a small workload declaration and records whether the training
command completed on every launcher node. The existing `traceml compare`
command can use two completed guarded-run summaries for an optional local CI
decision.

The contract format uses `schema_version: 1`. This identifies the first file
format; it does not indicate that the pilot is a stable 1.0 feature.

## Configure a run

Add an optional `guard` section to the existing `traceml.yaml`:

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

Run the training normally:

```bash
traceml run train.py \
  --nnodes 1 \
  --nproc-per-node 2 \
  --run-name reference
```

TraceML searches the launch directory and its parents for `traceml.yaml`. No
second configuration file or guard-specific launch option is required.

The guard pilot currently requires `traceml run`, summary mode, and history
recording. One-process and fixed-size DDP runs on one or more nodes are
allowed. `traceml watch`, `traceml serve`, and direct `traceml.init()` launches
ignore the guard declaration.

For multi-node DDP, use the ordinary TraceML launch shape documented in
[Distributed Training](distributed-training.md#multi-node-ddp). Every launcher
must discover the same `guard` declaration and use the same explicit
`--run-name`, `--nnodes`, and `--nproc-per-node`; only `--node-rank` differs.
Guarded multi-node runs must also resolve `--logs-dir` to the same shared
directory on every node. Node 0 records the normalized declaration, and every
launcher writes its outcome beneath that shared run directory. These small
files stay on the filesystem; TraceML does not send them through telemetry.

A guarded run name identifies exactly one execution. Node 0 refuses to start
when `logs/<run-name>/manifest.json` already exists, so choose a fresh
`--run-name` for every guarded launch. TraceML leaves the existing run intact.

## Contract fields

| Field | Required | Type | Rules |
| --- | --- | --- | --- |
| `guard.schema_version` | Yes | Integer | Must be `1`. |
| `guard.workload.name` | Yes | String | Nonempty, no surrounding whitespace, at most 128 characters. |
| `guard.workload.parameters` | No | Mapping | Defaults to `{}` and may contain at most 32 entries. |
| Parameter key | Yes per entry | String | Nonempty, no surrounding whitespace, at most 64 characters. |
| Parameter value | Yes per entry | Scalar | A nonempty string, signed integer, finite float, or Boolean. Strings may contain at most 256 characters. |
| `guard.measurement.start_step` | Yes | Integer | First completed step to measure; must be at least `1`. |
| `guard.measurement.completed_steps` | Yes | Integer | Number of completed steps to measure; must be at least `1`. |

Unknown fields, lists, nested parameter mappings, and null parameter values
are rejected. Integers and the inclusive measurement end must fit in a signed
64-bit value. Workload names, parameter keys, and string values cannot contain
Unicode category C characters, including control and format characters.

## Workload identity

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

## Measurement window

TraceML completed steps are numbered from 1. `start_step` is inclusive and
`completed_steps` is the exact requested count. This example:

```yaml
measurement:
  start_step: 10
  completed_steps: 50
```

requests steps 10 through 59. If `--trace-max-steps` is used, it must include
the complete requested range.

The complete requested window should remain inside `history_retention` until
the run is finalized. In v0.1, this declaration is a compatibility fingerprint:
two runs must declare the same range, but TraceML does not compare individual
step IDs or use the declaration to re-aggregate telemetry. The CI decision uses
the aggregate Step Time already stored for each final summary's
`step_time.global.window`. That analyzed range can differ from the requested
range or from the other run, and its analyzed-step count remains visible in the
comparison. Exact-window verification can be added later if pilot use shows it
is needed; it does not require SQLite or additional artifacts now.

## Captured artifact

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

## Compare two guarded runs

Pass the two portable summaries and an explicit Step Time threshold to the
existing compare command:

```bash
traceml compare \
  logs/reference/final_summary.json \
  logs/candidate/final_summary.json \
  --max-step-time-regression-pct 5 \
  --output compare/reference-vs-candidate
```

The reference comes first and the candidate second. Their normalized
declarations and expected topology must match, and both training and launcher
completion states must be completed. TraceML makes the decision from their
common-clock Step Time; it does not reopen the manifests or SQLite databases.

See [Compare Runs](compare.md#use-compare-in-ci) for result values, exit codes,
and compatibility behavior.

## Try the complete CPU DDP workflow

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
  --max-step-time-regression-pct 5 \
  --output compare/guard-reference-vs-candidate
```

The first summary is always the reference and the second is the candidate.
TraceML writes both JSON and text comparison artifacts before returning the
CI exit code. Shared development machines can produce timing noise, so choose
a threshold that reflects the stability of the environment where the runs
execute.

## Use the result in CI

The v0.1 pilot deliberately leaves reference selection to the user. Make the
chosen reference `final_summary.json` available to the job, run the candidate,
and pass both explicit paths to `traceml compare`. For example:

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

## Qualified scope

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

## Troubleshooting

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
