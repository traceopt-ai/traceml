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
2. Initialize TraceML with `traceml.init(mode="auto")`
3. Wrap your training step with `traceml.trace_step(model)`
4. Run `traceml run train.py`
5. Read the diagnosis in the CLI
6. Optionally collect a structured final summary
7. Optionally compare two runs
8. Optionally open the local UI

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.5+ |

TraceML works best with PyTorch training scripts that already run successfully on their own.

---

## Pick your stack

Three minimal paths to a first TraceML run, depending on how your training code is structured. Pick the tab that matches your setup — each shows install + the single change + the run command. Deeper details for each stack live in their integration pages.

=== "Plain PyTorch"

    ```bash
    pip install "traceml-ai[torch]"
    ```

    Wrap the training step body:

    ```python
    import traceml

    for step in range(num_steps):
        with traceml.trace_step(model):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    ```

    Run:

    ```bash
    traceml run train.py
    ```

    !!! note
        For a full end-to-end example, see the [plain PyTorch walkthrough](#2-minimal-training-script) below.

=== "HF Trainer"

    ```bash
    pip install "traceml-ai[hf]"
    ```

    Replace `Trainer` with `TraceMLTrainer`:

    ```python
    from traceml.integrations.huggingface import TraceMLTrainer

    trainer = TraceMLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        traceml_enabled=True,
    )
    trainer.train()
    ```

    Run:

    ```bash
    traceml run fine_tune.py
    ```

    !!! note
        For full HF details, multi-GPU DDP, and deeper layer signals, see the [Hugging Face integration](integrations/huggingface.md).

=== "Lightning"

    ```bash
    pip install "traceml-ai[lightning]"
    ```

    Add `TraceMLCallback` to your `Trainer`:

    ```python
    import lightning as L
    from traceml.integrations.lightning import TraceMLCallback

    trainer = L.Trainer(
        max_steps=500,
        callbacks=[TraceMLCallback()],
    )
    trainer.fit(model, train_dataloaders=loader)
    ```

    Run:

    ```bash
    traceml run train.py
    ```

    !!! note
        For full Lightning details, see the [PyTorch Lightning integration](integrations/lightning.md).

Everything below this point applies to all three stacks — reading output, compare runs, DDP, troubleshooting.

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
- `compare`

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
import traceml
import torch
import torch.nn as nn
import torch.optim as optim


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

    traceml.init(mode="auto")

    model = MyModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for step in range(200):
        with traceml.trace_step(model):
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

In a normal PyTorch loop, the preferred minimal setup is:

```python
traceml.init(mode="auto")

with traceml.trace_step(model):
    ...
```

Call `traceml.init(mode="auto")` once near the start of the script, then wrap
the full training step body from `zero_grad(...)` through `optimizer.step()`.

Legacy imports from `traceml.decorators` still work for backward compatibility.
The preferred API is the top-level `traceml.*`. Legacy decorator
imports are planned for deprecation starting in `v0.3.0`.

If you need explicit wrappers or partial auto-instrumentation, use
`mode="manual"` or `mode="selective"`. Keep that as a second step after you
are comfortable with the default `auto` path.

---

## 3) Run TraceML

```bash
traceml run train.py
```

This is the default TraceML workflow and the best place to start.

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
- tune batch size, precision, or kernels
- inspect forward, backward, or optimizer cost
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

- inspect forward, backward, or optimizer imbalance
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

## 5) Optional: structured final summary

If you want a low-noise run and a structured end-of-run payload for W&B or
MLflow, launch in summary mode:

```bash
traceml run train.py --mode=summary
```

Then call `traceml.final_summary()` near the end of your script:

```python
summary = traceml.final_summary(print_text=True)
if summary is not None:
    print(summary["step_time"]["diagnosis"]["status"])
```

This returns a Python dict generated by the aggregator process. It is intended
for logging selected TraceML diagnosis fields into your existing tracking stack.

TraceML also writes canonical end-of-run summary artifacts, including:

- `final_summary.json`
- `final_summary.txt`

`final_summary.json` is the canonical machine-readable TraceML summary artifact
and the intended input for downstream logging and run comparison.

---

## 6) Optional: compare two runs

If you have `final_summary.json` from two runs, compare them with:

```bash
traceml compare run_a.json run_b.json
```
This writes:

- a structured compare JSON
- a compact text report

`traceml compare` is designed to consume TraceML `final_summary.json`
artifacts.

Use compare when you want to answer questions like:


- did the run get slower or faster?
- did the diagnosis change?
- did wait share increase?
- did memory behavior get worse?

See [Compare Runs](compare.md).

---

## 7) Optional: local UI

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

## 8) Other run modes

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

## 9) Single-node DDP

TraceML supports single-node multi-GPU DDP.

Keep `traceml.trace_step(...)` inside the training loop.

If you also enable model hooks, call `traceml.trace_model_instance(model)`
before wrapping the model in `DistributedDataParallel`.

### Minimal DDP example

```python
import os

import traceml
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim


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
        with traceml.trace_step(model.module):
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

## 10) Optional model hooks

If you want per-layer timing and memory signals, you can attach model hooks.

```python
import traceml

traceml.trace_model_instance(model)
```

Use this together with `traceml.trace_step(model)`.

Launch `deep` mode when you want a short diagnostic run with deeper layer-level signals:

```bash
traceml deep train.py
```

Use model hooks only when:

- you already know the step is slow
- you want more detail about where inside the model the time or memory is going
- you are okay with some extra overhead for diagnosis

---

## 11) Common launch patterns

Standard CLI:

```bash
traceml run train.py
```

Local UI:

```bash
traceml run train.py --mode=dashboard
```

Summary-only run:

```bash
traceml run train.py --mode=summary
```

Compare two TraceML summary artifacts:

```bash
traceml compare run_a.json run_b.json
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

## 12) Troubleshooting

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
- try `--mode=summary` if you only want the final summary artifact and output

### I want the fastest path

Stay with:

```bash
traceml run train.py
```

Add the local UI or compare workflow only after you get value from the default run path.

---

## Next steps

- [Compare Runs](compare.md)
- [Hugging Face integration](integrations/huggingface.md)
- [PyTorch Lightning integration](integrations/lightning.md)
- [GitHub issues](https://github.com/traceopt-ai/traceml/issues)

If TraceML helped you find a slowdown, please open an issue and include:

- hardware / CUDA / PyTorch versions
- single GPU or multi-GPU
- whether you used `run`, `watch`, or `deep`
- the end-of-run summary
- a minimal repro if possible
