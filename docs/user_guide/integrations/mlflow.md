# MLflow Integration

TraceML can publish its end-of-run diagnosis into the MLflow run you already
use. The run then carries the bottleneck verdict next to your own metrics,
and you can compare runs by it.

## Install

This guide assumes your training framework is already installed.

```bash
pip install "traceml-ai[mlflow]"
```

The extra installs `mlflow-skinny`, which has no local store and needs a
tracking server. Install the full `mlflow` package to write to a local
SQLite file instead. Either package satisfies the extra.

## Run

Launch your existing script through TraceML:

```bash
traceml run train.py
```

Hugging Face Trainer, PyTorch Lightning and RF-DETR training are
instrumented automatically. Other stacks use the setup in their
[integration guide](../integrations.md#choose-your-integration). Your MLflow
setup stays as it is.

## Publish the diagnosis into your run

Call `log_training_report()` once after training finishes. With no
arguments it writes into `mlflow.active_run()`, or else into
`mlflow.last_active_run()`, the run that last ended in this thread.

### A run you know by id

Pass the run id, and the tracking URI when it is not the global one:

```python
from traceml_ai.integrations.mlflow import log_training_report

log_training_report(
    run_id="<run id>",
    tracking_uri="http://mlflow.example.com:5000",
)
```

You can pass your own `MlflowClient` with `client=` instead of
`tracking_uri=`.

### Hugging Face Trainer

The Trainer's MLflow callback ends its run when `train()` returns. Call the
helper right after training, in the same thread. It finds the ended run
through `mlflow.last_active_run()`:

```python
from traceml_ai.integrations.mlflow import log_training_report

trainer.train()
log_training_report()
```

### PyTorch Lightning

`MLFlowLogger` creates its run through a client, so the fluent API does not
see it. Pass the logger's run id and client explicitly:

```python
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from traceml_ai.integrations.mlflow import log_training_report

logger = MLFlowLogger(experiment_name="my-experiment")
trainer = Trainer(logger=logger)
trainer.fit(model, datamodule=datamodule)

if getattr(logger, "_initialized", False):
    log_training_report(logger._run_id, client=logger._mlflow_client)
```

The public `logger.run_id` and `logger.experiment` properties are avoided on
purpose. On a logger that has not started a run, reading them creates one
and resets the global tracking URI.

### Custom PyTorch loop

Open the run yourself, and call the helper before it closes:

```python
import mlflow
import traceml_ai as traceml
from traceml_ai.integrations.mlflow import log_training_report

traceml.init()

with mlflow.start_run():
    for batch in dataloader:
        with traceml.trace_step(model):
            ...

    log_training_report()
```

The [minimal example](https://github.com/traceopt-ai/traceml/blob/main/examples/mlflow_report_minimal.py)
runs this loop end to end. It writes to `mlflow.db` in the current
directory, so it needs no server:

```bash
traceml run examples/mlflow_report_minimal.py --args --steps 20
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## What lands in the run

- **Diagnosis card.** A compact HTML card is logged as the artifact
  `traceml/training-report.html`. The run's Artifacts tab renders it inline.
  The verdict comes first, then coverage, the phase bar and three Step Time
  averages. Links inside the card do not open from the MLflow UI.
- **Two metrics.** `traceml/step_time/step_time_ms` and
  `traceml/step_time/input_wait_ms` are logged at step 0. The run's charts
  group them under a `traceml/step_time` section, where a single-point
  metric is drawn as a bar. A value TraceML did not measure is left out
  rather than logged as zero.
- **One tag.** `traceml/primary/status` holds the verdict, cut to 5000
  characters.

Because these are ordinary metrics and tags, `mlflow.search_runs()` and the
runs table can filter and sort runs by them.

## How the helper behaves

- It writes through `MlflowClient` only. It never creates, resumes, ends or
  deletes a run, and never changes the run description.
- It warns instead of raising. A missing `mlflow` install, a store it cannot
  open, a missing run or a failed write prints one `[TraceML] MLflow:` line
  to stderr and returns `False`. It returns `True` when every write it had
  to make succeeded. A summary with no comparison values gets the card
  only.
- A finished, failed or killed run still accepts the report. A deleted run
  is refused.
- It writes synchronously even when `MLFLOW_ENABLE_ASYNC_LOGGING=true`, so
  the values are visible as soon as it returns.
- MLflow retries a request to an unreachable tracking server for about four
  minutes before the helper reports the failure. Lower
  `MLFLOW_HTTP_REQUEST_MAX_RETRIES` to fail sooner.
- Without `tracking_uri=` or `client=`, the store comes from
  `mlflow.set_tracking_uri()`, then `MLFLOW_TRACKING_URI`, then MLflow's
  default `sqlite:///mlflow.db`. From MLflow 3.13, a local file-path store
  such as `./mlruns` is refused unless `MLFLOW_ALLOW_FILE_STORE=true`.
- With `mlflow-skinny` only, point it at a tracking server. Skinny cannot
  open a local SQLite store.
- When it requests the summary itself, it writes from the primary rank
  only. On other ranks it returns `False` without requesting a summary, so
  you can call it on every rank.
- When it requests the summary itself, it needs a mode that keeps local
  history. In `telemetry` mode TraceML keeps no history and generates no
  summary, so the helper prints one line and returns `False`.
- A payload passed with `final_summary=` skips both checks. It is written
  wherever the helper is called, so pass it from one rank only.
- Once it has read the run, it requests the summary. When no TraceML
  session is available, or the summary request fails, it then raises the
  same `RuntimeError` as `traceml.final_summary()`. Pass `final_summary=`
  with a saved payload to publish an existing report.

## Limitations

- The helper does not define training boundaries. Use automatic trainer
  instrumentation or the explicit setup for your loop.
- Call it after training finishes. The card is a finalized report, not a
  live metric.
- `mlflow.last_active_run()` sees only runs started through the fluent API
  in the same thread. Pass `run_id=` otherwise.
- The Lightning recipe reads private `MLFlowLogger` fields, which a future
  Lightning release may rename.
- The repository tests the helper against a stand-in client, and against a
  real SQLite-backed client when MLflow is installed locally. No tracking
  server runs in CI.

## Next Steps

- [How to Read Output](../reading-output.md)
- [Compare Runs](../compare.md)
- [W&B, MLflow, and TensorBoard](wandb-mlflow.md)
- [Public API](../public-api.md#experiment-trackers)
