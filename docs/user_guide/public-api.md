# Public API

The stable surface that user code imports and calls. Everything in this page is covered by TraceML's compatibility contract across v0.x minor releases.

## Core API

```python
import traceml_ai as traceml

traceml.init(mode="auto")

with traceml.trace_step(model):
    ...
```

The old `import traceml` path remains available for compatibility, but emits a
`FutureWarning` and may be removed in a future release. Do not import from
decorator compatibility paths.

## Hugging Face integration

::: traceml_ai.integrations.huggingface.init
    options:
      show_root_heading: true
      show_source: true

::: traceml_ai.integrations.huggingface.TraceMLTrainerCallback
    options:
      show_root_heading: true
      show_source: true

::: traceml_ai.integrations.huggingface.TraceMLTrainer
    options:
      show_root_heading: true
      show_source: true

## PyTorch Lightning integration

::: traceml_ai.integrations.lightning.init
    options:
      show_root_heading: true
      show_source: true

::: traceml_ai.integrations.lightning.TraceMLCallback
    options:
      show_root_heading: true
      show_source: true

## Ray Train integration

::: traceml_ai.integrations.ray.TraceMLTorchTrainer
    options:
      show_root_heading: true
      show_source: true

::: traceml_ai.integrations.ray.TraceMLRayConfig
    options:
      show_root_heading: true
      show_source: true

## CLI

TraceML ships with a CLI entry point installed as `traceml`.

```bash
traceml run <script>                 # default: final summary JSON/text
traceml run <script> --mode=cli      # live terminal view
traceml run <script> --mode=dashboard # live browser view
traceml watch <script>               # zero-code system/process view
traceml serve                        # standalone aggregator for direct launch
```

Live `cli` and `dashboard` modes are intended for single-node runs. For
multi-node runs, use the default summary mode.
Dashboard mode requires the optional dashboard extra:
`pip install "traceml-ai[dashboard]"`.

Deep/layer profiling has been removed from the public CLI for now.

See `traceml --help` for the full set of options.

## Direct launch with `traceml serve`

`traceml run` starts the aggregator and your script together. If you would
rather launch training yourself with `python` or `torchrun`, run the aggregator
as a standalone process and call `traceml.init(...)` inside your script.

`traceml serve` owns only the aggregator. It binds a host/port, prints the
reachable endpoint, blocks until SIGINT/SIGTERM, shuts down cleanly, and writes
the final summary on exit. It never launches or wraps your training script.

```bash
# terminal 1: start the aggregator
traceml serve --aggregator-host 127.0.0.1 --aggregator-port 29765

# terminal 2: run your script directly
python train.py
```

For torchrun and multi-node, bind the aggregator so workers on other nodes can
connect:

```bash
traceml serve --aggregator-bind-host 0.0.0.0 --aggregator-host <node0-ip> --aggregator-port 29765
torchrun ... train.py
```

`traceml serve` flags:

| Flag | Meaning |
|---|---|
| `--aggregator-host` | Address workers connect to. Default `127.0.0.1`. |
| `--aggregator-bind-host` | Address the aggregator binds to. Use `0.0.0.0` for multi-node. Default `127.0.0.1`. |
| `--aggregator-port` | Aggregator TCP port. Default `29765`. |
| `--mode` | Display mode: `summary` (default), `cli`, or `dashboard`. |
| `--logs-dir` | Directory for session logs. |
| `--run-name` / `--session-id` | Run identity. Workers must use the same value for shared artifacts. |

### Configuration in the direct-launch path

In the direct `python` path you cannot pass `traceml run` flags, so runtime
settings are resolved with this precedence:

1. explicit `traceml.init(...)` arguments
2. `TRACEML_*` environment variables
3. `traceml.yaml`
4. built-in defaults

```python
traceml.init(
    mode="auto",          # instrumentation mode
    ui_mode="summary",    # display/telemetry mode
    logs_dir="logs",
    session_id="my_run",
    aggregator_host="127.0.0.1",
    aggregator_port=29765,
)
```

The aggregator endpoint (`aggregator_host`, `aggregator_port`) is a launch
setting, not read from `traceml.yaml`. If the aggregator is not reachable,
`traceml.init(...)` retries for a short bounded period and then raises a clear
error. Use `traceml.init(disabled=True)` (or `TRACEML_DISABLED=1`) to run with
tracing fully disabled as a no-op.

### Matching display modes across processes

`traceml run` configures one display mode for the whole run. In the direct-launch
path the aggregator and the worker are launched separately, so their display
modes are set independently: `traceml serve --mode` for the aggregator, and
`ui_mode` (via `traceml.init(ui_mode=...)` or `TRACEML_UI_MODE`) for the worker.

For the live `cli` STDOUT/STDERR panel to show your script's output, set both to
`cli` so the worker mirrors its stdout to the aggregator:

```bash
traceml serve --mode cli --run-name demo --aggregator-port 29765
TRACEML_UI_MODE=cli TRACEML_SESSION_ID=demo python train.py
```

If the modes differ (for example a `cli` aggregator with a worker left on the
default `summary`), telemetry, diagnosis, and the final summary are unaffected;
only worker stdout mirroring into the live panel is skipped.

## Summary APIs

### `traceml.summary()`

Returns a compact flat dict for experiment trackers such as W&B, MLflow, or
internal dashboards. Call it near the end of training; it reuses the canonical
`final_summary.json` if one already exists.

```python
summary = traceml.summary(print_text=True)
if summary is not None:
    wandb.log(summary)
```

### `traceml.final_summary()`

Returns the full `final_summary.json` payload. Use this when you need the
complete structured report or want to store the artifact for `traceml compare`.
TraceML generates this canonical artifact once per run and reuses it on later
calls.
