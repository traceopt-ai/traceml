# Ray Train

TraceML's Ray integration is intentionally thin: Ray launches and manages the
training workers, while TraceML observes those workers over its normal TCP
telemetry path.

```text
driver process
  -> starts one TraceML aggregator actor
  -> runs Ray TorchTrainer
       -> Ray starts training workers
            -> each worker starts one TraceML runtime
            -> each worker sends telemetry to the aggregator actor
```

Ray still owns scheduling, worker placement, ranks, process groups, and
DDP/NCCL/Gloo communication. TraceML does not replace Ray's launcher or reach
into Ray Train internals.

## Install

```bash
pip install "traceml-ai[ray]"
```

## Minimal Usage

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

Use the same ``train_loop_per_worker`` shape you would pass to Ray's
``TorchTrainer``. The wrapper starts TraceML before your loop runs and stops it
after the loop exits.

When using Ray Data, wrap the ``iter_torch_batches(...)`` iterator with
``traceml.wrap_dataloader_fetch(...)``. Ray Data is not a PyTorch
``DataLoader``, so the PyTorch DataLoader patch cannot see those fetches.

## Ray + Lightning

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

The ``examples/ray/lightning_text_classifier.py`` demo also includes
``--input-delay-ms`` / ``--input-delay-rank`` for input-straggler demos,
``--delay-ms`` / ``--delay-rank`` for compute-straggler demos, and
``--transfer-dim`` to make Lightning H2D timing visible.
``--transfer-dim`` creates a reusable per-batch CPU tensor; it does not add a
full dataset-sized tensor.

## Network Model

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

## Configuration

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
    sampler_interval_sec=1.0,
    bind_host="0.0.0.0",
    port=0,
)
```

The default ``mode="summary"`` is recommended for Ray because distributed worker
logs are noisy. Use ``mode="cli"`` only when you specifically want live terminal
rendering from the aggregator actor.

``init_mode`` is passed to ``traceml.init(mode="auto")`` inside each Ray
worker. The Ray Data ``wrap_dataloader_fetch(...)`` pattern above works with
the default auto mode because Ray Data iterators are separate from PyTorch
``DataLoader``. Use ``init_mode="manual"`` only if your training loop wraps
dataloader, forward, backward, and optimizer timing explicitly. Use
``init_mode="selective"`` with the ``patch_*`` options when you only want some
automatic patches.

## Lifecycle

``TraceMLTorchTrainer.fit()`` starts the aggregator actor, runs Ray Train, and
then stops the actor in a ``finally`` block. Each worker also stops its local
TraceML runtime in a ``finally`` block. Normal exceptions and keyboard
interrupts should therefore release TraceML resources. A hard ``SIGKILL`` cannot
run Python cleanup code in any framework.
