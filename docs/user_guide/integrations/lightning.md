# PyTorch Lightning

Trace an existing Lightning `Trainer.fit()` script without adding TraceML code.
Both `lightning.pytorch` and legacy `pytorch_lightning` are supported. Keep your
Trainer and LightningModule imports in the same namespace.

## Install

If Lightning is already installed:

```bash
pip install traceml-ai
```

To install the modern Lightning package with TraceML:

```bash
pip install "traceml-ai[lightning]"
```

## Run

```bash
traceml run train.py
```

TraceML initializes timing and attaches its callback after Lightning combines
Trainer callbacks with `LightningModule.configure_callbacks()`. Compatible
existing initialization and callbacks, including callback subclasses, are
reused. No TraceML code is needed in a standard training script.

For direct Python launches or custom loops, see
[Advanced: manual setup](#advanced-manual-setup).

## Read the result

TraceML prints a training summary and saves run artifacts. The step view shows
Input Wait, forward, backward, and optimizer time when those signals are
available. CUDA runs can also show host-to-device time and step memory. See
[How to Read Output](../reading-output.md) for each value's meaning.

Input Wait measures fetching a batch in the training process, not worker-side
decoding time or GPU idle time. Small transfers can display `H2D 0.0ms` because
the measured value is below display precision.

## What one step includes

Under automatic optimization, one TraceML step covers one Lightning optimizer
update attempt. With `accumulate_grad_batches=4`, four microbatches contribute
to one capture and share one step number. Forward, backward, H2D, input wait,
and traced-region times are added across that group. A final group can contain
fewer microbatches.

The sampler aggregates events inside the completed group; it does not merge
separately published microbatch records. CUDA memory is reset once at the
start of the group's memory window and read once at its end, so it reports a
peak rather than a sum. Temporary allocation peaks during batch transfer before
the memory reset are outside that window. CPU memory is unavailable for this
metric.

**Forward** measures calls to the selected training module or one of its direct
children during `training_step()`. This covers common `self(x)` and
`self.model(x)` patterns without placing hooks on every layer. Nested calls are
not counted again. Losses or metrics called as direct child modules can be
included; functional losses and ordinary Python logging outside observed module
calls are excluded. Backward recomputation is excluded from Forward.

H2D observes CPU-to-CUDA `Tensor.to()` calls during Lightning's batch transfer.
Backward follows Lightning's backward hooks, including `manual_backward()`.
Optimizer timing runs from `on_before_optimizer_step` to the end-of-batch
callback, including intervening work such as the step learning-rate scheduler.

Validation, sanity-check, test, and prediction input work is excluded. For a
DataLoader of unknown length, Lightning performs look-ahead and exhaustion
fetches; Input Wait reports those actual calls, so counts can differ from one
fetch per microbatch and individual fetches can appear one training step early.
If validation interrupts an accumulation group, its allocator activity can
contribute to that group's memory peak because the memory window remains open
until the optimizer-update boundary.

Manual optimization remains batch-scoped: a batch can contain several backward
and optimizer events in one TraceML step. Arbitrary manual accumulation is not
inferred. Failed or unfinished accumulation groups are discarded, and timing
hooks are removed after training or failure.

Checkpoint resume through `trainer.fit(..., ckpt_path=...)` keeps Lightning's
training state. TraceML step IDs remain local to the process and are not
restored from the checkpoint, so they may differ from `trainer.global_step`.
See the [step-time contract](../../developer_guide/step-time-pipeline-contract.md#lightning-steps).

## Advanced launch options

For single-node multi-GPU DDP:

```bash
traceml run train.py --nproc-per-node=4
```

Pass the same device count to Lightning (`Trainer(devices=4)`). TraceML launches
ranks with `torchrun`; Lightning picks up that environment. Without the matching
process count, Lightning reports a world-size mismatch before training starts.
For multi-node launch commands, see [Distributed Training](../distributed-training.md).
Distributed recipes are documented; they are not a broad tested strategy matrix.

For a browser dashboard on a single node:

```bash
traceml run train.py --mode=dashboard
```

TraceML can run alongside W&B, TensorBoard, and CSVLogger. These optional settings
make local terminal output easier to read:

| Setting | Purpose |
|---|---|
| `enable_progress_bar=False` | Avoid overlapping progress displays |
| `enable_model_summary=False` | Reduce startup output |
| `logger=False` | Disable Lightning loggers for a local diagnostic run |

The automatic example is `examples/integrations/lightning_minimal.py`. It also
accepts `--devices`, `--num-nodes`, `--max-steps`, `--delay-rank`, and `--delay-ms`.
The delay flags create a deliberate straggler.

For a real input-pipeline comparison, the existing manual example
`examples/integrations/lightning_dataloading_bottleneck.py` trains ResNet-18 on
Imagenette. Its `--profile` changes DataLoader settings:

```bash
traceml run --mode summary --logs-dir logs --run-name lightning_baseline \
    examples/integrations/lightning_dataloading_bottleneck.py \
    --args --profile baseline --max-steps 300 --batch-size 64
traceml run --mode summary --logs-dir logs --run-name lightning_optimized \
    examples/integrations/lightning_dataloading_bottleneck.py \
    --args --profile optimized --max-steps 300 --batch-size 64
traceml compare logs/lightning_baseline/final_summary.json \
    logs/lightning_optimized/final_summary.json
```

Its manual initialization and callback continue to work under automatic launch.
The same experiment is available in the
[Colab notebook](https://colab.research.google.com/github/traceopt-ai/traceml/blob/main/notebooks/lightning_dataloading_bottleneck.ipynb).

## Limitations

- **Automatic attachment.** Requires Lightning's standard callback connector and
  fitting lifecycle. CPU/CUDA single-device and ordinary DDP strategies are
  supported. DeepSpeed, other strategies, and spawn/fork launch modes are not
  attached automatically. Existing manual callbacks remain unchanged.
- **Specialized integrations.** RF-DETR keeps its dedicated, mode-aware adapter;
  call `traceml_ai.integrations.rfdetr.init()` as documented in the
  [RF-DETR guide](rfdetr.md). Generic Lightning attachment is skipped.
- **Forward coverage.** Observes module calls during standard `training_step()`;
  deeper modules called without their direct parent, arbitrary functional-only
  computation, or direct `.forward()` calls can bypass module hooks. A completed
  group with no observed calls produces a warning and leaves Forward unavailable.
- **Transfers and backward.** Custom streams, `.cuda()` calls, transfers inside
  `training_step()`, and direct `loss.backward()` calls can bypass the measured
  boundaries. CUDA graphs are outside this MVP's coverage. Automatic attachment
  skips `torch.compile` models; advanced manual setup remains user-controlled.
- **Custom input.** Non-PyTorch loaders require manual input wrapping; see below.
- **Existing configuration.** An incompatible TraceML initialization is left
  intact with a warning. Automatic attachment does not replace the user's
  configuration or callbacks.
- **Hardware validation.** CPU integration tests cover both namespaces. CUDA and
  distributed recipes retain their documented validation status; see the
  [support matrix](../integrations.md#integration-support-matrix).

For a baseline, run the same script with tracing disabled:

```bash
traceml run train.py --disable-traceml
```

Disabled launch installs no automatic import observer or timing hooks. Disabling
tracing later makes installed timing hooks stop recording.

`traceml watch train.py` remains resource-only and does not install Lightning
or Hugging Face training instrumentation.

## Advanced: manual setup

The current manual API remains available. Initialize the integration and supply
its callback before fitting:

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
behavior. Launching this script through `traceml run` reuses the callback.
Direct Python launches need a configured TraceML runtime/aggregator; without it,
initialization warns and training continues without telemetry.

For a custom iterator or non-PyTorch loader, initialize first and wrap the source
with `traceml.wrap_dataloader_fetch(...)` before passing it to `fit()`. Do not wrap
a normal PyTorch DataLoader: the integration already owns its timing. For Ray
Data with Lightning, see [Ray Train](ray.md).
