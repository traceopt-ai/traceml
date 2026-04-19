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
traceml watch <script>    # run script with live terminal dashboard
traceml run <script>      # run script with minimal instrumentation
traceml deep <script>     # run with full instrumentation (step + memory + layer)
```

See the [CLI module reference](../developer_guide/subsystems/cli.md) for the implementation.
