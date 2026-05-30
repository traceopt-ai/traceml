# Public API

The stable surface that user code imports and calls. Everything in this page is covered by TraceML's compatibility contract across v0.x minor releases.

## Core API

```python
import traceml_ai as traceml

traceml.init(mode="auto")

with traceml.trace_step(model):
    ...
```

The old `import traceml` path still works for now, but emits a `FutureWarning`
and will be removed in a future release. New code should use
`import traceml_ai as traceml`.

## Hugging Face integration

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
```

Live `cli` and `dashboard` modes are intended for single-node runs. For
multi-node runs, use the default summary mode.
Dashboard mode requires the optional dashboard extra:
`pip install "traceml-ai[dashboard]"`.

Deep/layer profiling has been removed from the public CLI for now.

See `traceml --help` for the full set of options.

## Summary APIs

### `traceml.summary()`

Returns a compact flat dict for experiment trackers such as W&B, MLflow, or
internal dashboards.

```python
summary = traceml.summary(print_text=True)
if summary is not None:
    wandb.log(summary)
```

### `traceml.final_summary()`

Returns the full `final_summary.json` payload. Use this when you need the
complete structured report or want to store the artifact for `traceml compare`.
