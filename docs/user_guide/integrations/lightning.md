# PyTorch Lightning Integration

## Install

This guide assumes PyTorch and Lightning are already installed. Both
`lightning.pytorch` and `pytorch_lightning` are supported. Keep your Trainer
and LightningModule imports in the same namespace.

```bash
pip install traceml-ai
```

## Run

```bash
traceml run train.py
```

Run your existing script with TraceML. Standard `Trainer.fit()` training is
instrumented automatically, with no code changes required. For custom training
loops or direct `python`/`torchrun` launches, see
[Advanced: manual setup](#advanced-manual-setup).

## Read the result

When training finishes, TraceML prints a diagnosis of the likely bottleneck
and saves the report for comparison or CI.

The timing breakdown shows input waiting, forward, backward, and optimizer
work. CUDA runs also show available host-to-device and memory measurements.

See [How to Read Output](../reading-output.md) for an example report and
explanations.

## How TraceML measures Lightning training

TraceML automatically attaches its callback after Lightning combines Trainer
callbacks with `LightningModule.configure_callbacks()`. Lightning continues
to manage batch transfer, gradient accumulation, and optimizer updates.

```text
Lightning training loop              TraceML measurement
──────────────────────────────────────────────────────────────────
Training DataLoader fetch            Input Wait
          ↓
strategy.batch_to_device()           Open capture + GPU transfer
          ↓
training_step()                      Observe forward module calls
          ↓
on_before_backward()                 Start backward timing
          ↓
on_after_backward()                  End backward timing
(repeat for accumulating microbatches)
          ↓
on_before_optimizer_step()           Start optimizer timing
          ↓
on_train_batch_end()                 Complete step if update is due
```

Under automatic optimization, one reported step covers one optimizer update
attempt. With `accumulate_grad_batches=4`, four microbatches contribute to
that step. TraceML adds their timings and reports peak CUDA memory within
the group's memory window. The final group can contain fewer microbatches.

**Forward** measures outermost module calls during `training_step()`, including
common `self(x)` and `self.model(x)` patterns. Nested calls are not counted
again. Input waiting is separate from traced training time; validation,
sanity-check, test, and prediction input work is excluded.

**Checkpoint resume:** Continue using Lightning's normal
`trainer.fit(..., ckpt_path=...)`. TraceML records resumed training with step
numbering local to the process, which may differ from `trainer.global_step`.
See [Limitations](#limitations) for timing coverage, or the
[step-time contract](../../developer_guide/step-time-pipeline-contract.md#lightning-steps)
for exact boundaries.

## Multi-GPU training

For single-node multi-GPU DDP:

```bash
traceml run train.py --nproc-per-node=4
```

Pass the same device count to Lightning (`Trainer(devices=4)`). TraceML launches
ranks with `torchrun`; Lightning picks up that environment. Without the matching
process count, Lightning reports a world-size mismatch before training starts.
For multi-node launch commands, see [Distributed Training](../distributed-training.md).

## Advanced: manual setup

Existing scripts using `init()` and `TraceMLCallback()` also work with
`traceml run`; TraceML reuses compatible setup without adding another callback.

Use this path for direct `python`/`torchrun` launches or custom loops that still
dispatch Lightning callbacks. For a direct launch, start an aggregator with
`traceml serve` first; see
[Direct Launch](../public-api.md#direct-launch-with-traceml-serve). Initialize
the integration and supply its callback before fitting:

```python
import lightning as L
from traceml_ai.integrations import lightning as traceml_lightning

traceml_lightning.init()
trainer = L.Trainer(callbacks=[traceml_lightning.TraceMLCallback()])
trainer.fit(model, train_dataloaders=loader)
```

Legacy projects can keep `import pytorch_lightning as L`. Both manual calls are
needed for the complete supported timing path. Compatible repeated `init()` is
a no-op; an explicit incompatible initialization keeps its existing error
behavior. Without a configured runtime/aggregator, initialization warns and
training continues without telemetry.

For a custom iterator or non-PyTorch loader, initialize first and wrap the source
with `traceml.wrap_dataloader_fetch(...)` before passing it to `fit()`. Do not wrap
a normal PyTorch DataLoader: the integration already owns its timing. For Ray
Data with Lightning, see [Ray Train](ray.md).

### Input-pipeline comparison example

The existing manual example trains ResNet-18 on Imagenette and compares two
DataLoader configurations. Its initialization and callback also work under
`traceml run`.

<details markdown="1">
<summary>Run the comparison</summary>

From the repository root, with the example's dependencies installed:

```bash
traceml run --logs-dir logs --run-name lightning_baseline \
    examples/integrations/lightning_dataloading_bottleneck.py \
    --args --profile baseline --max-steps 300 --batch-size 64
traceml run --logs-dir logs --run-name lightning_optimized \
    examples/integrations/lightning_dataloading_bottleneck.py \
    --args --profile optimized --max-steps 300 --batch-size 64
traceml compare logs/lightning_baseline/final_summary.json \
    logs/lightning_optimized/final_summary.json
```

The same experiment is available in the
[Colab notebook](https://colab.research.google.com/github/traceopt-ai/traceml/blob/main/notebooks/lightning_dataloading_bottleneck.ipynb).

</details>

## Limitations

- **Automatic attachment.** Requires Lightning's standard callback connector and
  fitting lifecycle. CPU/CUDA single-device and ordinary DDP strategies are
  supported. DeepSpeed, other strategies, and spawn/fork launch modes are not
  attached automatically. Existing manual callbacks remain unchanged.
- **Specialized integrations.** RF-DETR keeps its dedicated, mode-aware adapter;
  `traceml run train.py` activates it for standard RF-DETR `model.train()` runs.
  Generic Lightning attachment is skipped. See the [RF-DETR guide](rfdetr.md)
  for supported modes and advanced manual setup.
- **Forward coverage.** Observes module calls during standard `training_step()`;
  deeper modules called without their direct parent, arbitrary functional-only
  computation, or direct `.forward()` calls can bypass module hooks. A completed
  group with no observed calls produces a warning and leaves Forward unavailable.
  Hooks cover the training module and its direct children. Module-based losses
  or metrics can be included; functional losses and Python logging outside
  observed calls are excluded. Backward recomputation is excluded from Forward.
- **Transfers and backward.** Custom streams, `.cuda()` calls, transfers inside
  `training_step()`, and direct `loss.backward()` calls can bypass the measured
  boundaries. CUDA graphs are outside this MVP's coverage. Automatic attachment
  skips `torch.compile` models; advanced manual setup remains user-controlled.
  Optimizer timing extends from `on_before_optimizer_step` to the end-of-batch
  callback, so it can include step learning-rate scheduler work.
- **Input timing.** Input Wait measures fetching in the training process, not
  worker-side decoding or GPU idle time. Unknown-length loaders can trigger
  look-ahead fetches, so fetch counts and step attribution can differ from one
  fetch per microbatch. Small transfers can display `H2D 0.0ms` due to rounding.
- **Memory window.** CUDA peak measurement starts at `on_train_batch_start`,
  after batch transfer. Earlier allocation peaks are excluded. Validation
  between accumulating microbatches can contribute to the group's memory peak.
  CPU runs do not report this memory metric.
- **Manual optimization.** Steps remain batch-scoped. Several backward or
  optimizer calls in one batch share one reported step; arbitrary manual
  accumulation is not inferred. Unfinished groups are discarded on cleanup.
- **Custom input.** Non-PyTorch loaders require manual input wrapping; see
  [Advanced: manual setup](#advanced-manual-setup).
- **Existing configuration.** An incompatible TraceML initialization is left
  intact with a warning. Automatic attachment does not replace the user's
  configuration or callbacks.
- **Hardware validation.** CPU integration tests cover both namespaces. CUDA and
  distributed recipes retain their documented validation status; see the
  [support matrix](../integrations.md#integration-support-matrix).

## Next Steps

- [Minimal Lightning example](https://github.com/traceopt-ai/traceml/blob/main/examples/integrations/lightning_minimal.py)
- [How to Read Output](../reading-output.md)
- [Compare Runs](../compare.md)
- [Catch Regressions in CI](../regression-guard.md)
- [Distributed Training](../distributed-training.md)
- [W&B / MLflow](wandb-mlflow.md)
- [Open an issue](https://github.com/traceopt-ai/traceml/issues)
