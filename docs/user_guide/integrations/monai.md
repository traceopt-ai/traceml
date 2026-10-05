# MONAI Integration

## Install

This guide assumes PyTorch, MONAI, and its Ignite engine dependency are already
installed.

```bash
pip install traceml-ai
```

## Add the handler

Initialize the MONAI integration before building your `SupervisedTrainer`,
then add `TraceMLHandler()` to its `train_handlers`. Keep your existing
handlers alongside it.

```python
from monai.engines import SupervisedTrainer
from traceml_ai.integrations import monai as traceml_monai

traceml_monai.init()

trainer = SupervisedTrainer(
    device=device,
    max_epochs=5,
    train_data_loader=loader,
    network=network,
    optimizer=optimizer,
    loss_function=loss_function,
    train_handlers=[traceml_monai.TraceMLHandler()],
)
trainer.run()
```

Use both the integration's `init()` and its handler. Do not also call generic
`traceml.init()`: MONAI's handler measures batch fetching through engine events,
so it requires the generic DataLoader timing patch to stay disabled.

## Run

```bash
traceml run train.py
```

The command starts TraceML's collector and runs your script. MONAI requires the
handler setup above; it is not attached automatically by the launcher.

## Read the result

When training finishes, TraceML prints a diagnosis of the likely bottleneck
and saves the report for comparison or CI.

The timing breakdown shows input waiting, forward, backward, and optimizer
work. CUDA runs also show available host-to-device and memory measurements.
A large Input Wait points to batch availability; Residual includes work such
as loss computation and postprocessing that is outside the measured phases.

See [How to Read Output](../reading-output.md) for an example report and
explanations.

## How TraceML measures MONAI training

TraceML attaches to MONAI's `SupervisedTrainer` through its Ignite events and
wraps its batch preparation and inferer calls. MONAI continues to manage
training, metrics, and checkpoints.

```text
MONAI training path                 TraceML measurement
──────────────────────────────────────────────────────────────────
GET_BATCH_STARTED →                 Input Wait
GET_BATCH_COMPLETED
          ↓
prepare_batch()                     Open iteration + GPU transfer
          ↓
engine.inferer()                    Forward
          ↓
LOSS_COMPLETED → BACKWARD_COMPLETED  Backward
          ↓
optimizer.step()                    Optimizer
          ↓
MODEL_COMPLETED                     Close iteration timing;
                                    publish if update group is complete
```

One reported step covers one optimizer update attempt. With
`accumulation_steps=4`, four training iterations contribute to that step.
TraceML adds their timings and reports peak CUDA memory across the group's
measurement window. A loader with a known length can finish an epoch with a
shorter group.

Batch fetching appears separately as **Input Wait**. The traced iteration runs
from `prepare_batch()` through the handler's `MODEL_COMPLETED` callback.
Evaluation is not traced by this handler. For `ThreadDataLoader`, Input Wait
measures how long the training thread waits, not how long the background
thread spends preparing data.

## Advanced options

### Complete examples

The [minimal example](https://github.com/traceopt-ai/traceml/blob/main/examples/integrations/monai_minimal.py)
trains a small UNet on synthetic volumes and downloads nothing. From the
repository root, run:

```bash
traceml run examples/integrations/monai_minimal.py
```

<details markdown="1">
<summary>Compare input loading on a real workload</summary>

[`examples/integrations/monai_dataloading_bottleneck.py`](https://github.com/traceopt-ai/traceml/blob/main/examples/integrations/monai_dataloading_bottleneck.py)
trains a 3D UNet on patches from the Medical Segmentation Decathlon spleen task.
Each flag changes one setting: the dataset class, the worker count, the loader
class, or mixed precision. Compare two runs that differ in one setting, keeping the dataset and other
training options fixed:

```bash
traceml run --logs-dir logs --run-name spleen_1_baseline \
    examples/integrations/monai_dataloading_bottleneck.py --args --data-dir data
traceml run --logs-dir logs --run-name spleen_2_workers \
    examples/integrations/monai_dataloading_bottleneck.py \
    --args --data-dir data --num-workers 4
traceml compare logs/spleen_1_baseline/final_summary.json \
    logs/spleen_2_workers/final_summary.json
```

The notebook runs six settings on one GPU, compares each adjacent pair, and
shows where the bottleneck moves as each one changes:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/traceopt-ai/traceml/blob/main/notebooks/monai_dataloading_bottleneck.ipynb)

</details>

### Timing details

<details markdown="1">
<summary>Loss, postprocessing, schedulers, and mixed precision</summary>

`network.train()`, `optimizer.zero_grad()` and the loss run between
`prepare_batch` and the events above. They sit inside the step and in no phase.
TraceML has no loss stream, so loss time is not reported on its own.

MONAI registers decollation and postprocessing on `MODEL_COMPLETED`, ahead of
the handler that closes the step, so both are inside the step too. Handlers on
`ITERATION_COMPLETED`, such as metrics, schedulers and loggers, run after the
step closes and fall outside it.

A step the AMP scaler skips usually records no optimizer event, because the
hooks are on `optimizer.step()` and the scaler does not call it when it finds an
inf. The optimizer is an occurrence signal, so such a step contributes zero to
the average optimizer time rather than making the whole stream abstain. A run
whose scaler skips every step reports no optimizer time at all. A fused
optimizer is the exception: the scaler calls its `step()` either way and skips
the update inside the kernel, so the event is recorded.

</details>

## Limitations

- **Supported engine.** The handler supports `SupervisedTrainer` with MONAI's
  standard `_iteration`, including subclasses that inherit it. GAN trainers,
  evaluators, `iteration_update=` replacements, and overridden iterations
  receive a warning and are not traced.
- **Initialization and duplicates.** Generic initialization with the DataLoader
  patch enabled causes the handler to skip tracing with a warning. A duplicate
  handler on the same trainer is also refused to avoid double counting.
- **Unknown-length accumulation.** Unfinished groups are discarded when MONAI
  clears their gradients or when the run ends. See the accumulation note below.

- **Handler ordering.** Backward is measured between the handler's own two callbacks. Another handler
  registered between them falls inside that window.
- **Inferer proxy.** While a run is traced, `engine.inferer` is a timing proxy. Reads and writes
  both reach the real inferer, so state a handler stores on it survives, but
  the object is not MONAI's `Inferer` for an `isinstance` check, `repr` shows
  the proxy, and it does not deepcopy. The original is restored at every run
  exit, including after an exception.
- **Cleanup.** `engine.run` stays wrapped, and the engine keeps a marker naming its handler,
  for the life of the engine. Only `prepare_batch` and `inferer` are put back at
  a run exit.
- **Optimizer scope.** Optimizer time is `optimizer.step()` only. A step-interval scheduler, such as
  `LrScheduleHandler(epoch_level=False)`, runs on `ITERATION_COMPLETED`, so it
  is in neither the optimizer phase nor the step.
- **Hardware validation.** CUDA is a documented recipe, not CI tested: the spleen notebook linked above ran on
  one T4, and CI covers CPU single-process runs only. `multi_process` and
  `multi_node` are not claimed, because the handler has no per-rank branch.

<details markdown="1">
<summary>Accumulation with an unknown-length loader</summary>

With `accumulation_steps=N` and a loader of known length, the last group of an
epoch is published even if it holds fewer than N iterations, because MONAI
forces an optimizer step there. When the loader has no known length,
as with an `IterableDataset` that has no `__len__`, MONAI does not force
that step, and the window stays open past the epoch boundary. MONAI itself
zeroes that window's gradients once it learns the epoch length, without ever
stepping them, so TraceML drops it too rather than merging it into the group
that steps next. A group still open when the run ends is dropped the same way.

</details>

## Next Steps

- [How to Read Output](../reading-output.md)
- [Compare Runs](../compare.md)
- [Catch Regressions in CI](../regression-guard.md)
- [Public API](../public-api.md)
- [Open an issue](https://github.com/traceopt-ai/traceml/issues)
