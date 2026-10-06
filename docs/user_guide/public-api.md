# Public API

For supported Hugging Face Trainer, Lightning, and RF-DETR training, start
with `traceml run train.py`. The launcher attaches the integration; you do not
need the Python setup calls below. See [Integrations](integrations.md) for
supported paths and explicit setup in other frameworks.

Use this reference for custom-loop instrumentation, summary export, and
advanced launches. Import the stable core API from `traceml_ai`:

```python
import traceml_ai as traceml
```

The reference below documents every symbol in `traceml_ai.__all__`.

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

Use these for manual or selective instrumentation. `init(mode="auto")` already
owns the matching PyTorch paths, so wrappers reject duplicate instrumentation.
The exception is `wrap_dataloader_fetch(...)` for a custom non-PyTorch iterator
that automatic DataLoader instrumentation cannot observe. When TraceML is
disabled, each wrapper is an identity no-op. Otherwise, invalid targets raise
their documented `TypeError` before initialization and ownership are checked.

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

TraceML installs the `traceml` command:

```bash
traceml run train.py                  # training diagnosis and saved reports
traceml run train.py --mode=cli       # live terminal view
traceml run train.py --mode=dashboard # live browser view
traceml watch train.py                # system/process visibility
traceml serve                         # standalone aggregator for direct launches
```

Summary mode is the default for every topology. Live CLI and dashboard modes
are intended for single-node runs. `watch` does not install automatic trainer
instrumentation or provide step timing.

Use `traceml run --help` for launch flags. See
[Distributed Training](distributed-training.md) for multi-node commands,
[Compare Runs](compare.md) for saved report comparison, and
[Regression Guard](regression-guard.md) for CI thresholds.

### Training output

`run` and `watch` save training stdout and stderr by default under
`logs/<run-name>/nodes/node_<node-rank>/`. Summary and dashboard modes mirror
both streams; CLI mode shows a bounded stderr excerpt after failure.

Use `--no-save-training-output` to inherit the terminal instead. Captured
streams are pipes, so `isatty()` is false. TraceML does not replace Python's
`sys.stdout` or `sys.stderr`. Aggregator diagnostics are saved separately.
See [Training Crashes](training-crashes.md) for artifact paths and failure
behavior.

### History and configuration

`run`, `watch`, and `serve` accept `--history-retention DURATION`. The default
is `30m`; bare values are seconds, with `s`, `m`, `h`, and `d` suffixes accepted.
Use `--no-history` on `run` or `watch` only with a live display mode.
Summary mode, HTML reports, and summary APIs require history.

The retention setting is also available as `history_retention` in
`traceml.yaml` and `TRACEML_HISTORY_RETENTION`. CLI settings take precedence
over environment variables, YAML, and built-in defaults, in that order.

<details markdown="1">
<summary>How history retention works</summary>

Step Time and Step Memory are pruned through a shared step boundary aligned
across every expected rank in both streams and older than the chosen duration.
System, Process, and GPU history use that step's timestamp. Later arrivals at
or before a deleted step or timestamp are dropped before insertion.

</details>

## Direct Launch with `traceml serve`

For a direct `python` or `torchrun` launch, start the aggregator separately
and use the [manual setup for your integration](integrations.md) in the script.
Custom PyTorch loops use `traceml.init()` and `traceml.trace_step()`; framework
integrations may require their own initializer and callback or handler.

```bash
# terminal 1
traceml serve --aggregator-host 127.0.0.1 --aggregator-port 29765

# terminal 2: script already contains its explicit TraceML setup
python train.py
```

`serve` does not launch training or attach framework callbacks. Stop it after
training to finalize its reports. Training stdout/stderr remain owned by your
terminal, scheduler, or container runtime.

<details markdown="1">
<summary>Multi-node direct launches and serve flags</summary>

Bind the aggregator on a reachable address and set its endpoint on every
training node:

```bash
traceml serve --aggregator-bind-host 0.0.0.0 --aggregator-host <node0-ip> \
  --aggregator-port 29765 --nnodes <N> --nproc-per-node <M>

TRACEML_AGGREGATOR_HOST=<node0-ip> TRACEML_AGGREGATOR_PORT=29765 \
  torchrun ... train.py
```

Workers default to `127.0.0.1`, so non-aggregator nodes need the reachable
node-0 endpoint. Configure the expected world size to match your workers.

| Flag | Meaning |
| --- | --- |
| `--aggregator-host` | Worker connection address; default `127.0.0.1`. |
| `--aggregator-bind-host` | Bind address; use `0.0.0.0` for multi-node. |
| `--aggregator-port` | Telemetry port; default `29765`. |
| `--nnodes` / `--nproc-per-node` | Expected world size for finalization. |
| `--mode` | `summary` (default), `cli`, or `dashboard`. |
| `--logs-dir` | Session log directory. |
| `--run-name` / `--session-id` | Shared run identity; choose a fresh name. |
| `--history-retention` | History duration; default `30m`. |

An existing run folder is not overwritten. Telemetry with a different
explicit worker session ID is ignored. Workers that generate their own ID,
such as a plain `python train.py`, are still admitted.

</details>

### Missing-aggregator behavior

Launcher startup and in-script initialization have different defaults:

| Entry point | Default if the aggregator is unavailable |
| --- | --- |
| `traceml run` / `traceml watch` | Stop before launching training. |
| Explicit `traceml.init()` | Retry within a bounded timeout, warn, and continue with tracing disabled. |

To let the launcher continue without telemetry:

```bash
traceml run train.py --on-missing-aggregator=warn
```

To require telemetry during explicit setup:

```python
traceml.init(on_missing_aggregator="raise")
```

The policy resolves from the explicit flag or argument, then
`TRACEML_ON_MISSING_AGGREGATOR`, then the entry point's default. It is not a
`traceml.yaml` setting. Calls that require an aggregator, including
`traceml.summary()`, still fail when telemetry is unavailable.

An occupied telemetry endpoint is also a startup failure. Stop the earlier
aggregator or choose another `--aggregator-port`.

After training starts, telemetry failures are reported separately and do not
replace the training exit code. Only the aggregator-owning launcher reports
final telemetry health; in multi-node runs this is node 0. See
[Training Crashes](training-crashes.md) for the manifest and diagnostics.

### Direct-launch configuration

Runtime settings resolve from explicit `init()` arguments, environment
variables, `traceml.yaml`, then built-in defaults. For display mode, the Python
argument is `ui_mode`; `init(mode=...)` selects instrumentation behavior.

Aggregator host and port resolve from explicit arguments, environment
variables, then defaults. Run identity and aggregator endpoints are not read
from YAML. See the function references above for accepted arguments.

## Framework Integrations

Framework integrations are separate from the stable core API above. Use the
matching integration guide for installation and runtime requirements.

### Hugging Face

For a standard `transformers.Trainer`, use `traceml run train.py`; the launcher
attaches `TraceMLTrainerCallback` when training starts. Direct launches and
custom training loops can still call the integration `init()` and register the
callback manually. See the [Hugging Face guide](integrations/huggingface.md)
for setup and limitations.

::: traceml_ai.integrations.huggingface.init
    options:
      show_root_heading: true
      show_source: false

::: traceml_ai.integrations.huggingface.TraceMLTrainerCallback
    options:
      show_root_heading: true
      show_source: false

### PyTorch Lightning

Run a standard Lightning script with `traceml run train.py` for automatic
initialization and callback attachment. Both Lightning namespaces are supported.
For direct launches, the manual path uses both `init()` and `TraceMLCallback()`.
Compatible existing setup is reused under automatic launch. See the
[Lightning guide](integrations/lightning.md) for step semantics and limits.

::: traceml_ai.integrations.lightning.init
    options:
      show_root_heading: true
      show_source: false

::: traceml_ai.integrations.lightning.TraceMLCallback
    options:
      show_root_heading: true
      show_source: false

### RF-DETR

Run an ordinary RF-DETR `model.train()` script with `traceml run train.py` for
automatic attachment. Manual setup needs only `rfdetr.init()` before training;
the integration supplies its specialized callback. Compatible existing setup
is reused. See the [RF-DETR guide](integrations/rfdetr.md) for supported modes.

::: traceml_ai.integrations.rfdetr.init
    options:
      show_root_heading: true
      show_source: false

### MONAI

Call `init()` before building the trainer, then pass `TraceMLHandler()` in
`train_handlers`. See the [MONAI guide](integrations/monai.md) for the seam
each phase is measured from.

::: traceml_ai.integrations.monai.init
    options:
      show_root_heading: true
      show_source: false

::: traceml_ai.integrations.monai.TraceMLHandler
    options:
      show_root_heading: true
      show_source: false

### Ray Train

Use `TraceMLTorchTrainer` and its configuration for explicit worker-side
setup. See the [Ray Train guide](integrations/ray.md).

::: traceml_ai.integrations.ray.TraceMLTorchTrainer
    options:
      show_root_heading: true
      show_source: false

::: traceml_ai.integrations.ray.TraceMLRayConfig
    options:
      show_root_heading: true
      show_source: false
