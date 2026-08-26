# Public API

Import the stable core API from `traceml_ai`:

```python
import traceml_ai as traceml
```

The reference below documents every symbol in `traceml_ai.__all__`. The old
`import traceml` path remains a compatibility import and emits a
`FutureWarning`; new code should use `traceml_ai`.

## Stable Core API

### `traceml.__version__`

A string identifying the installed TraceML version. It is useful when recording
the environment for a run or bug report.

### Lifecycle

::: traceml_ai.api.init
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: traceml_ai.api.start
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: false

### Step boundary

::: traceml_ai.api.trace_step
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: false

### End-of-run summaries

::: traceml_ai.api.summary
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: traceml_ai.api.final_summary
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: false

### Manual instrumentation helpers

Use these only for manual or selective instrumentation. Automatic mode already
times the matching PyTorch paths, and manual wrappers reject duplicate automatic
instrumentation where double-counting would be possible.

::: traceml_ai.api.wrap_dataloader_fetch
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: traceml_ai.api.wrap_forward
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: traceml_ai.api.wrap_backward
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: traceml_ai.api.wrap_optimizer
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: traceml_ai.api.wrap_h2d
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: false

## CLI

TraceML installs the `traceml` command-line entry point:

```bash
traceml run <script>                  # final summary JSON/text
traceml run <script> --mode=summary   # explicit summary mode
traceml run <script> --mode=cli       # live terminal view
traceml run <script> --mode=dashboard # live browser view
traceml watch <script>                # zero-code system/process summary
traceml serve                         # standalone aggregator
```

`run`, `watch`, and `serve` accept `--history-retention DURATION`. The default
is `30m`; bare values are seconds, and `s`, `m`, `h`, and `d` suffixes are
accepted. Step Time and Step Memory are pruned through the minimum step that is
aligned across every expected rank in both streams and older than the selected
duration. System, Process, GPU, and stdout/stderr history use that same step's
timestamp. Later arrivals at or before a deleted step or timestamp are dropped
before insertion. Use `--no-history` on `run` or `watch` when history should be
disabled entirely.

The same setting is available as `history_retention` in `traceml.yaml` and as
`TRACEML_HISTORY_RETENTION`. Precedence remains CLI, environment, YAML, then
the 30-minute built-in default.

Summary mode is the default for every topology. Live `cli` and `dashboard`
modes are intended for single-node runs. Use PyTorch Profiler or Nsight for
operator- or kernel-level profiling.

## Direct Launch with `traceml serve`

`traceml run` starts the aggregator and your script together. For direct
`python` or `torchrun` launches, start the aggregator yourself and call
`traceml.init(...)` inside the training script:

```bash
# terminal 1
traceml serve --aggregator-host 127.0.0.1 --aggregator-port 29765

# terminal 2
python train.py
```

`traceml serve` owns only the aggregator. It binds the endpoint, waits for a
shutdown signal, and writes the final summary; it never launches or wraps the
training script.

For multi-node workers, bind the aggregator on a reachable address and set the
endpoint on every training node:

```bash
traceml serve --aggregator-bind-host 0.0.0.0 --aggregator-host <node0-ip> \
  --aggregator-port 29765 --nnodes <N> --nproc-per-node <M>

TRACEML_AGGREGATOR_HOST=<node0-ip> TRACEML_AGGREGATOR_PORT=29765 \
  torchrun ... train.py
```

Workers resolve `TRACEML_AGGREGATOR_HOST` as `127.0.0.1` by default, so every
non-aggregator node needs the reachable node-0 address above.

`traceml serve` flags:

| Flag | Meaning |
|---|---|
| `--aggregator-host` | Address workers connect to; default `127.0.0.1`. |
| `--aggregator-bind-host` | Bind address; use `0.0.0.0` for multi-node. |
| `--aggregator-port` | Aggregator TCP port; default `29765`. |
| `--nnodes` / `--nproc-per-node` | Expected world size; the aggregator waits for all ranks before finalizing. |
| `--mode` | `summary` (default), `cli`, or `dashboard`. |
| `--logs-dir` | Directory for session logs. |
| `--run-name` / `--session-id` | Shared run identity for worker artifacts. |
| `--history-retention` | Aligned raw-history duration; default `30m`. |

### Missing-aggregator behavior

If the aggregator cannot be reached, `traceml.init(...)` retries for its
bounded timeout, writes one stderr warning, and continues with tracing disabled
as a no-op. This is the default `warn` policy; it does not stop training.

Use strict behavior when telemetry is required, for example in CI:

```python
traceml.init(on_missing_aggregator="raise")
```

The policy resolves in this order: the explicit
`on_missing_aggregator` argument, `TRACEML_ON_MISSING_AGGREGATOR`, then `warn`.
It is not read from `traceml.yaml`.

`aggregator_host` and `aggregator_port` are direct-launch settings, not
`traceml.yaml` settings. Other runtime settings resolve as explicit
`traceml.init(...)` arguments, then `TRACEML_*` environment variables, then
`traceml.yaml`, then built-in defaults.

### Matching display modes across processes

In direct-launch mode, set the aggregator display with `traceml serve --mode`
and the worker display with `traceml.init(ui_mode=...)` or `TRACEML_UI_MODE`.
Use `cli` for both when the live terminal panel should include worker output:

```bash
traceml serve --mode cli --run-name demo --aggregator-port 29765
TRACEML_UI_MODE=cli TRACEML_SESSION_ID=demo python train.py
```

If the modes differ, telemetry, diagnosis, and final artifacts are unaffected;
only worker stdout mirroring into the live panel is skipped.

## Framework Integrations

Framework integrations are separate from the stable core API above. Use the
matching integration guide for installation and runtime requirements.

### Hugging Face

Preferred path: call the integration `init()` once and register
`TraceMLTrainerCallback` with your existing `transformers.Trainer`.

::: traceml_ai.integrations.huggingface.init
    options:
      show_root_heading: true
      show_source: false

::: traceml_ai.integrations.huggingface.TraceMLTrainerCallback
    options:
      show_root_heading: true
      show_source: false

#### Legacy compatibility: `TraceMLTrainer`

`TraceMLTrainer` remains supported for existing users. New code should prefer
`TraceMLTrainerCallback`; see the [Hugging Face guide](integrations/huggingface.md)
for its trade-offs.

::: traceml_ai.integrations.huggingface.TraceMLTrainer
    options:
      show_root_heading: true
      show_source: false

### PyTorch Lightning

::: traceml_ai.integrations.lightning.init
    options:
      show_root_heading: true
      show_source: false

::: traceml_ai.integrations.lightning.TraceMLCallback
    options:
      show_root_heading: true
      show_source: false

### Ray Train

::: traceml_ai.integrations.ray.TraceMLTorchTrainer
    options:
      show_root_heading: true
      show_source: false

::: traceml_ai.integrations.ray.TraceMLRayConfig
    options:
      show_root_heading: true
      show_source: false
