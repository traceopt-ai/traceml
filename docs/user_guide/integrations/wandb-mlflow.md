# Use TraceML with W&B, MLflow, or TensorBoard

Keep your existing experiment tracking and logging. TraceML adds a training
bottleneck diagnosis and saves evidence for comparison or CI.

## Install

This guide assumes your training framework and tracker are already installed.

```bash
pip install traceml-ai
```

## Run

For standard Hugging Face Trainer, PyTorch Lightning, or RF-DETR training,
launch your existing script through TraceML:

```bash
traceml run train.py
```

Training is instrumented automatically. Keep your existing tracker setup;
no tracker changes are required to receive the TraceML report.

For custom PyTorch loops, add the
[explicit setup below](#advanced-custom-pytorch-loops) before launching.
MONAI, Ray, Accelerate, and DeepSpeed use the setup in their
[matching integration guide](../integrations.md#choose-your-integration).

## Read the result

When training finishes, TraceML prints a diagnosis of the likely bottleneck
and saves the report for comparison or CI. Your tracker continues to record
its existing metrics and artifacts.

See [How to Read Output](../reading-output.md) for an example report and
explanations. Exporting TraceML results into a tracker is optional.

## Export the diagnosis to your tracker

Add one of the following snippets after training finishes, while your tracker
run is still active. They assume you already initialized W&B or started an
MLflow run.

`traceml.summary()` returns a flat dictionary of diagnosis fields and average
metrics. By default, it returns `None` on non-primary ranks, so the export
runs only on the primary rank.

### W&B

```python
import traceml_ai as traceml
import wandb

# After trainer.train(), trainer.fit(...), or your training loop:
summary = traceml.summary()
if summary is not None:
    wandb.log(summary)
```

### MLflow

MLflow stores numeric values as metrics and diagnosis strings as tags:

```python
import traceml_ai as traceml
import mlflow

# After training, inside your active MLflow run:
summary = traceml.summary()
if summary is not None:
    metrics = {
        key: value for key, value in summary.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    tags = {
        key.replace("/", "."): value for key, value in summary.items()
        if isinstance(value, str)
    }
    mlflow.log_metrics(metrics)
    mlflow.set_tags(tags)
```

### Save the full report

Keep `logs/<run_name>/final_summary.json` for local comparison or CI. You can
also attach it using your tracker's artifact API. For an active MLflow run:

```python
full = traceml.final_summary()
if full is not None:
    mlflow.log_dict(full, "traceml/final_summary.json")
```

`final_summary()` returns the complete structured report; `summary()` returns
its compact projection. Later calls reuse the saved final report.

## Advanced: custom PyTorch loops

Initialize TraceML once and wrap the training step. Keep tracker logging
outside the timed block. The following assumes your model, optimizer,
DataLoader, and active tracker run already exist:

```python
import traceml_ai as traceml

traceml.init(mode="auto")

for step, batch in enumerate(dataloader):
    with traceml.trace_step(model):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["x"])
        loss = criterion(outputs, batch["y"])
        loss.backward()
        optimizer.step()

    # Keep your existing tracker logging here.
```

For W&B, the logging line can be:

```python
wandb.log({"loss": loss.item()})
```

For MLflow:

```python
mlflow.log_metric("loss", loss.item(), step=step)
```

Place the chosen logging call inside the loop, after the timed block. Then
launch with `traceml run train.py`. For batch transfers, accumulation, and
custom input sources, see the [Custom PyTorch Loop quickstart](../quickstart.md#custom-pytorch-loop)
and [Public API](../public-api.md).

A [runnable summary example](https://github.com/traceopt-ai/traceml/blob/main/examples/summary_logging_minimal.py)
shows the same initialization and final-summary API without requiring a tracker.

## Limitations

- W&B, MLflow, and TensorBoard do not define TraceML's training boundaries.
  Use automatic trainer instrumentation or the explicit setup for your loop.
- TraceML does not export to trackers automatically. The snippets above log
  results through W&B and MLflow's own APIs. Existing TensorBoard logging can
  remain enabled; this guide does not provide a TensorBoard export adapter.
- Request the final summary only after all training workers finish their work.
  It is a finalized report, not a live metrics stream.
- Summary requests require an active TraceML session with retained history.
  They can raise `RuntimeError` if history is unavailable, the aggregator
  reports an error, or the request times out.

## Next Steps

- [How to Read Output](../reading-output.md)
- [Compare Runs](../compare.md)
- [Catch Regressions in CI](../regression-guard.md)
- [Training Integrations](../integrations.md)
- [Summary API](../public-api.md#end-of-run-summaries)
