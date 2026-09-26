# Regression guard measurement contract

TraceML's local regression guard is an experimental **v0.1 pilot**. The first
piece is a run-bound measurement contract: a small declaration of the workload
and completed training steps that a later guard check will evaluate.

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

The guard pilot currently requires `traceml run`, one node, summary mode, and
history recording. One-process and multi-process single-node runs are allowed.
`traceml watch`, `traceml serve`, and direct `traceml.init()` launches ignore
the guard declaration.

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
64-bit value.

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
1. Two runs will later be comparable only when their normalized workload
declarations match exactly.

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

The complete window must also remain inside `history_retention` until the run
is finalized. This capture stage records the request but does not extend
retention. A later guard check will treat an evicted or incomplete window as
inconclusive.

This first implementation records the request. A later guard stage will verify
that every expected rank produced the complete window before allowing a
comparison.

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
        "parameters": {}
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

To inspect the captured declaration, format the manifest and look under
`guard.contract`:

```bash
python -m json.tool logs/reference/manifest.json
```

This stage does not compare runs or return a CI regression decision. Those
capabilities will consume the captured contract in later pilot releases.
