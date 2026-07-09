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
traceml run <script>                 # default: live CLI on single-node runs
traceml run <script> --mode=summary  # final summary JSON/text only
traceml run <script> --mode=cli      # explicit live terminal view
traceml run <script> --mode=dashboard # live browser view
traceml watch <script>               # zero-code system/process view
```

Live `cli` and `dashboard` modes are intended for single-node runs. Multi-node
runs default to summary mode.
Dashboard mode requires the optional dashboard extra:
`pip install "traceml-ai[dashboard]"`.

TraceML no longer ships layer-level/deep profiling. Use PyTorch Profiler,
Nsight, or another operator-level profiler when you need that detail.

See `traceml --help` for the full set of options.

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
