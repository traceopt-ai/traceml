# Weights & Biases Integration

TraceML can publish its end-of-run diagnosis into the W&B run you already
use. The run then carries the bottleneck verdict next to your own metrics,
and the runs table can sort and compare runs by it.

## Install

This guide assumes your training framework is already installed.

```bash
pip install "traceml-ai[wandb]"
```

## Run

Launch your existing script through TraceML:

```bash
traceml run train.py
```

Hugging Face Trainer, PyTorch Lightning and RF-DETR training are
instrumented automatically. Other stacks use the setup in their
[integration guide](../integrations.md#choose-your-integration). Your W&B
setup stays as it is.

## Publish the diagnosis into your run

Call `log_training_report()` once after training finishes, while the W&B run
is still open. With no arguments it writes into the active `wandb.run`.

### Hugging Face Trainer

The Trainer's W&B callback keeps its run open after `train()` returns, so call
the helper right after training:

```python
from traceml_ai.integrations.wandb import log_training_report

trainer.train()
log_training_report()
```

### PyTorch Lightning

`WandbLogger` owns its run. Its `finalize()` does not call `wandb.finish()`,
so the run stays open after `trainer.fit()` returns. Call the helper then and
pass that run explicitly:

```python
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from traceml_ai.integrations.wandb import log_training_report

logger = WandbLogger(project="my-project")
trainer = Trainer(logger=logger)
trainer.fit(model, datamodule=datamodule)
log_training_report(run=trainer.logger.experiment)
```

### Custom PyTorch loop

Create and finish the run yourself, and call the helper in between:

```python
import traceml_ai as traceml
import wandb
from traceml_ai.integrations.wandb import log_training_report

traceml.init()
run = wandb.init(project="my-project")

for batch in dataloader:
    with traceml.trace_step(model):
        ...

log_training_report()
run.finish()
```

The [minimal example](https://github.com/traceopt-ai/traceml/blob/main/examples/wandb_report_minimal.py)
runs this loop end to end. Set `WANDB_MODE=offline` to try it without a W&B
account:

```bash
WANDB_MODE=offline traceml run examples/wandb_report_minimal.py --args --steps 20
```

## What lands in the run

- **Diagnosis card.** A compact HTML card is logged under
  `traceml/diagnosis_card`. W&B shows it as a media panel in the run
  workspace. The verdict comes first, because a workspace panel shows about
  330 by 180 px before you expand it. Below it are coverage, the phase bar
  and three Step Time averages.
- **Summary fields.** `traceml/primary/status`,
  `traceml/step_time/step_time_ms` and `traceml/step_time/input_wait_ms` are
  written to the run summary. They appear as columns in the runs table, so
  you can sort and filter runs by the verdict and the two timings. A field
  TraceML did not measure is left out rather than written as zero.

## How the helper behaves

- It never creates, configures or finishes a run. It reads no credentials
  and never calls `wandb.init()` or `wandb.login()`.
- It warns instead of raising. A missing `wandb` install, a missing run, a
  payload it cannot render or a failed write prints one `[TraceML] W&B:` line
  to stderr and returns `False`.
- It returns `True` when every write that had something to write succeeded.
  A summary with no diagnosis and no Step Time averages gets the card only
  and still returns `True`.
- When it fetches the summary itself, it writes from the primary rank only.
  On other ranks it returns `False` without requesting a summary, so you can
  call it on every rank.
- When it fetches the summary itself, it needs a mode that keeps local
  history. In `telemetry` mode TraceML keeps no history and generates no
  summary, so the helper prints one line and returns `False`.
- A payload passed with `final_summary=` skips both checks. It is written
  wherever the helper is called, so pass it from one rank only.
- Once it has a run, it requests the summary. When no TraceML session is
  available, or the summary request fails, it then raises the same
  `RuntimeError` as `traceml.final_summary()`. Pass `final_summary=` with a
  saved payload to publish an existing report.

## Live per-step export

This helper writes the end-of-run diagnosis only. Streaming per-step records
into W&B while training runs is the job of a `wandb` exporter, which lands in
a later release. Until then, see [Live Export](../live-export.md) for the
built-in JSON Lines exporter.

## Limitations

- The helper does not define training boundaries. Use automatic trainer
  instrumentation or the explicit setup for your loop.
- Call it after training finishes. The card is a finalized report, not a
  live metric.
- The repository tests the helper against a stand-in `wandb` module. No real
  W&B client runs in CI.

## Next Steps

- [How to Read Output](../reading-output.md)
- [Compare Runs](../compare.md)
- [W&B, MLflow, and TensorBoard](wandb-mlflow.md)
- [Public API](../public-api.md#experiment-trackers)
