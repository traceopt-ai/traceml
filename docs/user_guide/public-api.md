# Public API

The stable surface that user code imports and calls. Everything in this page is covered by TraceML's compatibility contract across v0.x minor releases.

## Decorators

::: traceml.decorators.trace_step
    options:
      show_root_heading: true
      show_source: true

::: traceml.decorators.trace_model_instance
    options:
      show_root_heading: true
      show_source: true

## Hugging Face integration

::: traceml.integrations.huggingface.TraceMLTrainer
    options:
      show_root_heading: true
      show_source: true

## PyTorch Lightning integration

::: traceml.integrations.lightning.TraceMLCallback
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
traceml deep <script>                # deeper layer-level diagnostics
```

Live `cli` and `dashboard` modes are intended for single-node runs. For
multi-node runs, use the default summary mode.

See `traceml --help` for the full set of options.
