# Ray Train

TraceML AI's Ray integration is intentionally thin: Ray launches and manages the
training workers, while TraceML AI observes those workers over its normal TCP
telemetry path.

```text
driver process
  -> starts one TraceML AI aggregator actor
  -> runs Ray TorchTrainer
       -> Ray starts training workers
            -> each worker starts one TraceML AI runtime
            -> each worker sends telemetry to the aggregator actor
```

Ray still owns scheduling, worker placement, ranks, process groups, and
DDP/NCCL/Gloo communication. TraceML AI does not replace Ray's launcher or reach
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

    import traceml_ai as tml

    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 4))
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    tml.trace_model_instance(model)

    for _ in range(config["steps"]):
        with tml.trace_step(model):
            x = torch.randn(64, 32)
            y = torch.randint(0, 4, (64,))

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()


ray.init()

trainer = TraceMLTorchTrainer(
    train_loop_per_worker,
    train_loop_config={"steps": 10},
    scaling_config=ScalingConfig(num_workers=4, use_gpu=False),
    traceml_config=TraceMLRayConfig(mode="summary"),
)

trainer.fit()
```

Use the same ``train_loop_per_worker`` shape you would pass to Ray's
``TorchTrainer``. The wrapper starts TraceML AI before your loop runs and stops it
after the loop exits.

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

``init_mode`` is passed to ``tml.init()`` inside each Ray worker. Use
``init_mode="manual"`` if your training loop wraps dataloader, forward,
backward, and optimizer timing explicitly. Use ``init_mode="selective"`` with
the ``patch_*`` options when you only want some automatic patches.

## Lifecycle

``TraceMLTorchTrainer.fit()`` starts the aggregator actor, runs Ray Train, and
then stops the actor in a ``finally`` block. Each worker also stops its local
TraceML AI runtime in a ``finally`` block. Normal exceptions and keyboard
interrupts should therefore release TraceML AI resources. A hard ``SIGKILL`` cannot
run Python cleanup code in any framework.
