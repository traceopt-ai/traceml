# Ray Train Integration

## Install

This guide assumes PyTorch and Ray Train are already installed. Add TraceML
in the driver and training-worker environments:

```bash
pip install traceml-ai
```

## Add TraceML to your training job

Use `TraceMLTorchTrainer` in place of Ray's `TorchTrainer`. Keep your existing
worker function, datasets, and Ray scaling configuration:

```python
from traceml_ai.integrations.ray import TraceMLRayConfig, TraceMLTorchTrainer

trainer = TraceMLTorchTrainer(
    train_loop_per_worker,
    train_loop_config=train_loop_config,
    scaling_config=scaling_config,
    datasets=datasets,
    traceml_config=TraceMLRayConfig(),
)
trainer.fit()
```

The adapter initializes TraceML in each worker before calling your function.
You do not need a separate `traceml.init()` call in that function.

Inside your existing PyTorch worker loop, mark each training step:

```python
import traceml_ai as traceml

# Model, optimizer, loss function, and loader are created in this worker.
for batch in train_loader:
    with traceml.trace_step(model):
        x = batch["x"].to(device)
        y = batch["y"].to(device).long()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
```

For Ray Data, wrap the iterator before entering the loop to measure input
waiting:

```python
from ray import train

train_ds = train.get_dataset_shard("train")
train_loader = traceml.wrap_dataloader_fetch(
    train_ds.iter_torch_batches(batch_size=64, prefetch_batches=1)
)
```

A normal PyTorch DataLoader is timed automatically by the default worker
initialization; do not wrap it again.

## Run

Launch your configured Ray training script with Python:

```bash
python train.py
```

Ray launches the workers, and `TraceMLTorchTrainer.fit()` starts the TraceML
aggregator. This path does not require `traceml run` or `traceml serve`.
Keep your existing Ray cluster connection and `ScalingConfig` settings.

## Read the result

When training finishes, TraceML prints a diagnosis of the likely bottleneck
and saves the report for comparison or CI.

The timing breakdown shows available input waiting, forward, backward, and
optimizer measurements across workers. CUDA runs can also show host-to-device
and memory measurements. Reports are saved under the configured `logs_dir`
and session ID.

See [How to Read Output](../reading-output.md) for an example report and
explanations.

## How TraceML measures Ray training

Ray continues to manage worker placement, ranks, and distributed training.
The adapter starts one TraceML aggregator actor and connects each worker's
telemetry to it. Step measurements follow the boundaries you mark in the
worker function:

```text
Ray training worker                 TraceML measurement
──────────────────────────────────────────────────────────────
Wrapped Ray Data fetch              Input Wait
          ↓
trace_step(model)                   Open the step capture
          ↓
Batch transfer → Forward            H2D + forward timing
          ↓
Backward → Optimizer                Backward + optimizer timing
          ↓
Exit trace_step                     Complete one reported step
```

In the loop above, one reported step contains one batch and one optimizer
update attempt. Ray does not automatically group gradient accumulation for
TraceML: the placement of `trace_step()` determines the reported boundary.
The Ray + Lightning setup below instead uses Lightning's callback to group
accumulating microbatches.

Input Wait measures how long the worker waits for the iterator, not total
preprocessing time inside Ray Data. Transfers must occur inside `trace_step()`
to be included in its H2D measurements.

## Advanced options

### Complete example

<details markdown="1">
<summary>Minimal CPU example</summary>

```python
import ray
from ray.train import ScalingConfig

from traceml_ai.integrations.ray import TraceMLRayConfig, TraceMLTorchTrainer


def train_loop_per_worker(config):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from ray import train

    import traceml_ai as traceml

    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 4))
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_ds = train.get_dataset_shard("train")
    train_loader = train_ds.iter_torch_batches(
        batch_size=64,
        prefetch_batches=1,
    )
    train_loader = traceml.wrap_dataloader_fetch(train_loader)

    for step, batch in enumerate(train_loader):
        if step >= config["steps"]:
            break

        with traceml.trace_step(model):
            x = batch["x"]
            y = batch["y"].long()

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()


ray.init()
train_dataset = ray.data.from_items(
    [{"x": [0.0] * 32, "y": 0} for _ in range(4096)]
)

trainer = TraceMLTorchTrainer(
    train_loop_per_worker,
    train_loop_config={"steps": 10},
    scaling_config=ScalingConfig(num_workers=4, use_gpu=False),
    datasets={"train": train_dataset},
    traceml_config=TraceMLRayConfig(mode="summary"),
)

trainer.fit()
```

</details>

### Example scripts

Use ``--ray-address=auto`` when you already have a Ray cluster running, or omit
it for a local Ray run.

Minimal Ray Train example:

```bash
python examples/integrations/ray/torchtrainer_minimal.py \
  --ray-address=auto \
  --num-workers=2 \
  --steps=100 \
  --use-gpu
```

To make input timing visible in the minimal example:

```bash
python examples/integrations/ray/torchtrainer_minimal.py \
  --ray-address=auto \
  --num-workers=2 \
  --steps=100 \
  --use-gpu \
  --input-delay-ms=100
```

### Ray + Lightning

When combining Ray Train and PyTorch Lightning, add ``TraceMLCallback()`` to the
Lightning ``Trainer`` and keep wrapping Ray Data iterators with
``traceml.wrap_dataloader_fetch(...)``. To capture Lightning H2D timing inside
Ray workers, initialize the worker patches selectively:

```python
TraceMLRayConfig(
    mode="summary",
    init_mode="selective",
    patch_dataloader=True,
    patch_h2d=True,
)
```

The ``examples/integrations/ray/lightning_text_classifier.py`` demo also includes
``--input-delay-ms`` / ``--input-delay-rank`` for input-straggler demos,
``--delay-ms`` / ``--delay-rank`` for compute-straggler demos, and
``--transfer-dim`` to make Lightning H2D timing visible.
``--transfer-dim`` creates a reusable per-batch CPU tensor; it does not add a
full dataset-sized tensor.

Baseline Ray + Lightning run:

```bash
python examples/integrations/ray/lightning_text_classifier.py \
  --ray-address=auto \
  --num-workers=2 \
  --max-steps=100 \
  --use-gpu
```

Input-straggler demo:

```bash
python examples/integrations/ray/lightning_text_classifier.py \
  --ray-address=auto \
  --num-workers=2 \
  --max-steps=100 \
  --use-gpu \
  --input-delay-rank=0 \
  --input-delay-ms=200
```

Compute-straggler demo:

```bash
python examples/integrations/ray/lightning_text_classifier.py \
  --ray-address=auto \
  --num-workers=2 \
  --max-steps=100 \
  --use-gpu \
  --delay-rank=0 \
  --delay-ms=200
```

For CPU-only runs, remove ``--use-gpu``.

### Network Model

The aggregator runs as a normal Ray actor and binds a TCP server. By default it
binds ``0.0.0.0`` on port ``0``:

- ``0.0.0.0`` lets workers on other Ray nodes connect to the actor node.
- port ``0`` lets the operating system choose a free port.
- workers receive the actor's reachable node IP and chosen port through the
  wrapped trainer.

If your cluster requires a fixed open port, set it explicitly:

```python
TraceMLRayConfig(port=29765)
```

### Configuration

```python
TraceMLRayConfig(
    mode="summary",
    profile="run",
    init_mode="auto",
    patch_dataloader=None,
    patch_forward=None,
    patch_backward=None,
    patch_h2d=None,
    logs_dir="./logs",
    session_id="",
    sampler_interval_sec=2.0,
    history_retention="30m",
    bind_host="0.0.0.0",
    port=0,
)
```

The default ``mode="summary"`` is recommended for Ray because distributed worker
logs are noisy. Use ``mode="cli"`` only when you specifically want live terminal
rendering from the aggregator actor.

``sampler_interval_sec`` defaults to ``2.0`` seconds. It controls worker sampling
and the aggregator actor's live UI refresh; incoming TCP telemetry is drained as
soon as it arrives.

``history_retention`` accepts positive seconds or durations such as ``"30m"``,
``"2h"``, or ``"1d"``. It defaults to 30 minutes of final-report analysis
history. Retention advances at completed, rank-aligned step boundaries and
does not create a rollup.

#### Migrating from `summary_window_rows`

The former row-count setting is no longer accepted. Configure the history
duration instead:

```python
# Before: no longer supported.
TraceMLRayConfig(summary_window_rows=1_000)

# Now: retain the duration needed for analysis.
TraceMLRayConfig(history_retention="30m")
```

A row count does not map to a fixed duration, so choose the duration explicitly
for your workload. The default is `"30m"`; it is not an automatic conversion
from 1,000 rows.

`init_mode` selects the mode passed to `traceml.init()` inside each Ray
worker. The Ray Data ``wrap_dataloader_fetch(...)`` pattern above works with
the default auto mode because Ray Data iterators are separate from PyTorch
``DataLoader``; wrapping a PyTorch ``DataLoader`` in the same mode raises to
prevent duplicate timing. Use ``init_mode="manual"`` only if your training
loop wraps dataloader, forward, backward, and optimizer timing explicitly.
Use ``init_mode="selective"`` with the ``patch_*`` options when you only want
some automatic patches.

### Lifecycle

``TraceMLTorchTrainer.fit()`` starts the aggregator actor, runs Ray Train, and
then stops the actor in a ``finally`` block. Each worker also stops its local
TraceML runtime in a ``finally`` block. Normal exceptions and keyboard
interrupts should therefore release TraceML resources. A hard ``SIGKILL`` cannot
run Python cleanup code in any framework.

If aggregator finalization fails, the actor records the failure in
``aggregator/traceml_errors.log``. Cleanup remains best effort at the Ray driver
boundary, so a TraceML shutdown failure does not replace the Ray Train result.

## Limitations

- **Explicit step setup.** The adapter initializes worker instrumentation,
  but custom loops still need `trace_step()`. Ray Data iterators also need
  `wrap_dataloader_fetch()` for input timing.
- **Model preparation.** Keep Ray's normal device placement and distributed
  model preparation in your worker function. The TraceML trainer adapter does
  not prepare or distribute the model for you.
- **Network access.** Workers must be able to reach the aggregator actor's
  TCP endpoint. Configure an accessible port when your cluster requires one.
- **Validation evidence.** See the
  [support matrix](../integrations.md#integration-support-matrix) for tested
  versions and hardware coverage.

## Next Steps

- [How to Read Output](../reading-output.md)
- [Compare Runs](../compare.md)
- [Catch Regressions in CI](../regression-guard.md)
- [Ray Train example](https://github.com/traceopt-ai/traceml/blob/main/examples/integrations/ray/torchtrainer_minimal.py)
- [W&B / MLflow](wandb-mlflow.md)
- [Open an issue](https://github.com/traceopt-ai/traceml/issues)
