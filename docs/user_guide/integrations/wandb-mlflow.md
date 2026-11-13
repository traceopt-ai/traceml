# Use TraceML with W&B, MLflow, or TensorBoard

Keep your existing experiment tracking and logging. TraceML adds a training
bottleneck diagnosis and saves evidence for comparison or CI. Launch your
script with `traceml run train.py`, then publish the diagnosis into your
tracker with the page that matches it below.

`traceml run` instruments only the supported trainers automatically. They
are Hugging Face Trainer, PyTorch Lightning and RF-DETR. A custom PyTorch
loop needs `traceml.init()` and `trace_step()` first, as shown in
[Custom PyTorch Loop](../quickstart.md#custom-pytorch-loop). Other stacks
use the setup in their
[integration guide](../integrations.md#choose-your-integration).

## W&B

The [W&B integration](wandb.md) publishes the diagnosis card and three
comparison fields into your existing run with one call,
`log_training_report()`.

## MLflow

The [MLflow integration](mlflow.md) publishes the diagnosis card and three
comparison values into your existing run with one call,
`log_training_report()`.

## Save the full report

Keep `logs/<run_name>/final_summary.json` for local comparison or CI. You can
also attach it using your tracker's artifact API. For an active MLflow run:

```python
full = traceml.final_summary()
if full is not None:
    mlflow.log_dict(full, "traceml/final_summary.json")
```

`final_summary()` returns the complete structured report; `summary()` returns
its compact projection. Later calls reuse the saved final report.

A [runnable MLflow example](https://github.com/traceopt-ai/traceml/blob/main/examples/mlflow_summary_minimal.py)
logs the whole compact summary as metrics and tags at the end of a small
loop.

## TensorBoard

Existing TensorBoard logging can remain enabled next to TraceML. TraceML
does not provide a TensorBoard export adapter.

## Limitations

- W&B, MLflow, and TensorBoard do not define TraceML's training boundaries.
  Use automatic trainer instrumentation or the explicit setup for your loop.
- The W&B and MLflow helpers write the end-of-run diagnosis only, not live
  per-step metrics.
- Request the final summary only after all training workers finish their work.
  It is a finalized report, not a live metrics stream.
- Summary requests require an active TraceML session with retained history.
  They can raise `RuntimeError` if history is unavailable, the aggregator
  reports an error, or the request times out.

## Next Steps

- [W&B integration](wandb.md)
- [MLflow integration](mlflow.md)
- [How to Read Output](../reading-output.md)
- [Compare Runs](../compare.md)
- [Catch Regressions in CI](../regression-guard.md)
- [Training Integrations](../integrations.md)
- [Summary API](../public-api.md#end-of-run-summaries)
