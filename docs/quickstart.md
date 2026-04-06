# TraceML Quickstart

Get from install to your first useful TraceML run in a few minutes.

This guide is for practitioners who want the fastest path to an answer.

TraceML is most useful when you want to know:

- why training is slow
- whether the bottleneck is input, compute, wait, or rank imbalance
- whether memory is drifting over time

If you are new to TraceML, start here.

---

## What you will do

1. Install TraceML
2. Wrap your training step with `trace_step(model)`
3. Run `traceml run train.py`
4. Read the diagnosis in the CLI
5. Optionally open the local UI

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.5+ |

TraceML works best with PyTorch training scripts that already run successfully on their own.

---

## 1) Install

```bash
pip install traceml-ai
```

Check that the CLI is available:

```bash
traceml --help
```

You should see commands such as:

- `watch`
- `run`
- `deep`

### Optional extras

For Hugging Face Trainer support:

```bash
pip install "traceml-ai[hf]"
```

For PyTorch Lightning support:

```bash
pip install "traceml-ai[lightning]"
```

If you want the PyTorch versions TraceML is tested against:

```bash
pip install "traceml-ai[torch]"
```

---

## 2) Minimal training script

Save this as `train.py`.

```python
import torch
import torch.nn as nn
import torch.optim as optim

from traceml.decorators import trace_step


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    model = MyModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for step in range(200):
        with trace_step(model):
            inputs = torch.randn(64, 128, device=device)
            labels = torch.randint(0, 10, (64,), device=device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if step % 50 == 0:
            print(f"Step {step} | loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
```

### The only required change

In a normal PyTorch loop, the only required code change is:

```python
with trace_step(model):
    ...
```

Wrap the full training step body, from `zero_grad(...)` through `optimizer.step()`.

---

## 3) Run TraceML

```bash
traceml run train.py
```

This is the default TraceML workflow.

During training, TraceML opens a live terminal view alongside your logs.

At the end of the run, it prints a compact summary you can review or share.

---

## 4) How to read the output

TraceML is built to answer one question quickly:

**Why is this training job slow?**

Common diagnoses:

### `INPUT-BOUND`

The dataloader or preprocessing path is taking a large share of the step.

Typical next steps:

- increase dataloader workers
- improve storage throughput
- reduce CPU preprocessing cost
- check host-to-device transfer delays

### `COMPUTE-BOUND`

Model compute dominates the step.

Typical next steps:

- reduce model step cost
- tune batch size / precision / kernels
- inspect forward / backward / optimizer cost
- use a deeper profiler only after identifying the hot path

### `INPUT STRAGGLER`

One rank is slower in the input path than the others.

Typical next steps:

- inspect dataloader imbalance
- check rank-local preprocessing
- check host-side jitter
- look at the worst rank called out in the summary

### `COMPUTE STRAGGLER`

One rank is slower in compute than the others.

Typical next steps:

- inspect forward / backward / optimizer imbalance
- check uneven data shapes or rank-local work
- inspect the worst rank called out in the summary

### `WAIT-HEAVY`

A meaningful part of the step is going into waiting rather than useful work.

Typical next steps:

- inspect synchronization points
- check CPU stalls
- check uneven rank progress
- compare wait share against input and compute shares

### `MEMORY CREEP`

Memory is rising over time instead of staying stable.

Typical next steps:

- inspect retained tensors
- inspect caches and per-step state
- compare early vs later steps
- look for graph-backed tensors kept alive across steps

---

## 5) Optional: local UI

If you want a richer view, run:

```bash
traceml run train.py --mode=dashboard
```

The local UI runs at:

```text
http://localhost:8765
```

Use the local UI when you want:

- a richer run review experience
- an easier browser-based layout
- local comparison of runs

If you just want the fastest path, stay with the default CLI mode.

---

## 6) Other run modes

### `traceml watch`

```bash
traceml watch train.py
```

Use this when you want:

- zero-code system and process visibility
- a quick look before adding step instrumentation

`watch` is lighter-weight than `run`, but it does not provide the same step-aware diagnosis.

### `traceml deep`

```bash
traceml deep train.py
```

Use this only for short diagnostic runs when you need deeper per-layer signals.

`deep` is more expensive than `run` and is best used after TraceML already showed you where to dig.

---

## 7) Single-node DDP

TraceML supports single-node multi-GPU DDP.

Keep `trace_step(...)` inside the training loop.

If you also enable model hooks, call `trace_model_instance(model)` before wrapping the model in `DistributedDataParallel`.

### Minimal DDP example

```python
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from traceml.decorators import trace_step


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


def main():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    model = MyModel().to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank] if use_cuda else None,
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for step in range(200):
        with trace_step(model.module):
            inputs = torch.randn(64, 128, device=device)
            labels = torch.randint(0, 10, (64,), device=device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

Launch with:

```bash
traceml run train.py --nproc-per-node=4
```

> Scope: multi-node distributed training is not yet supported.

---

## 8) Optional model hooks

If you want per-layer timing and memory signals, you can attach model hooks.

```python
from traceml.decorators import trace_model_instance

trace_model_instance(model)
```

Use this together with `trace_step(model)`.

Launch `deep` mode when you want a short diagnostic run with deeper layer-level signals:

```bash
traceml deep train.py
```

Use model hooks only when:

- you already know the step is slow
- you want more detail about where inside the model the time or memory is going
- you are okay with some extra overhead for diagnosis

---

## 9) Common launch patterns

Standard CLI:

```bash
traceml run train.py
```

Local UI:

```bash
traceml run train.py --mode=dashboard
```

Single-node DDP:

```bash
traceml run train.py --nproc-per-node=4
```

Zero-code first look:

```bash
traceml watch train.py
```

Run without telemetry for a baseline comparison:

```bash
traceml run train.py --disable-traceml
```

Pass arguments to your training script:

```bash
traceml run train.py --args -- --epochs 10 --lr 1e-3
```

---

## 10) Troubleshooting

### `torchrun: command not found`

TraceML launches your script through:

```bash
python -m torch.distributed.run
```

Check that this works:

```bash
python -m torch.distributed.run --help
```

If that works but your environment still fails to launch, check your Python environment and PATH.

### Output is messy in the terminal

If your own logger, progress bar, or framework output is fighting with the TraceML CLI:

- disable `tqdm`
- reduce extra terminal logging
- try `--mode=dashboard` for browser-based viewing

### I want the fastest path

Stay with:

```bash
traceml run train.py
```

Add the local UI only after you get value from the CLI.

---

## Next steps

- Hugging Face integration: `docs/huggingface.md`
- PyTorch Lightning integration: `docs/lightning.md`
- GitHub issues: `https://github.com/traceopt-ai/traceml/issues`

If TraceML helped you find a slowdown, please open an issue and include:

- hardware / CUDA / PyTorch versions
- single GPU or multi-GPU
- whether you used `run`, `watch`, or `deep`
- the end-of-run summary
- a minimal repro if possible
