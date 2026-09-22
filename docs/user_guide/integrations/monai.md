# MONAI

MONAI's `SupervisedTrainer` is an Ignite engine, so TraceML attaches to it as a
MONAI handler. Pass `TraceMLHandler()` in `train_handlers` and each training
step is timed. Some phases come from the engine's events and the rest from
wrapping what the trainer calls, which the Limitations below spell out. The
trainer keeps its own handlers, metrics and checkpoints.

## Install and initialize

MONAI is not a TraceML dependency. Install the extra:

```bash
pip install "traceml-ai[monai]"
```

Initialize TraceML before building the trainer:

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

Run the script with `traceml run train.py` to start the telemetry collector.

Use `traceml_monai.init()` as the only TraceML init in the process. A plain
`traceml.init()` installs the torch DataLoader patch, which would count every
fetch a second time. The handler declines to trace that run and says so on
stderr.

## What one step is

One TraceML step is one optimizer update, as in the Lightning callback. With
`accumulation_steps=N` a step spans N iterations.

When the loader has a length, the last group of an epoch is published even if it
holds fewer than N, because MONAI forces an optimizer step there. When it does
not, as with an `IterableDataset` that has no `__len__`, MONAI does not force
that step, and the window stays open past the epoch boundary. MONAI itself
zeroes that window's gradients once it learns the epoch length, without ever
stepping them, so TraceML drops it too rather than merging it into the group
that steps next. A group still open when the run ends is dropped the same way.

## What each phase measures

| Phase | Measured from |
| --- | --- |
| Input Wait | the engine's `GET_BATCH_STARTED` to `GET_BATCH_COMPLETED` bracket, on the training thread |
| Forward | the `engine.inferer(...)` call, which is where MONAI runs the model |
| Backward | `LOSS_COMPLETED` to `BACKWARD_COMPLETED` |
| Optimizer | step hooks on the trainer's own optimizer |
| Step memory | the step as a whole, on CUDA only |
| H2D | the `.to()` calls `prepare_batch` makes, on CUDA only |

The window opens when MONAI calls `prepare_batch`, so the batch transfer is
inside it, and closes at `MODEL_COMPLETED`. That happens once per iteration, so
under `accumulation_steps` a published step is the sum of its iterations'
windows. Ignite fetches the batch before the iteration starts, which is why
Input Wait sits outside the window rather than inside it.

## Where MONAI's own work lands

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

## Which engines are traced

Only a `SupervisedTrainer` that runs MONAI's own `_iteration`, including a
subclass that inherits it. `GanTrainer`, `AdversarialTrainer`, evaluators, a
trainer built with `iteration_update=`, and a subclass that overrides
`_iteration` each get one warning and nothing attached.

One handler traces one trainer. A second handler on the same trainer is refused
with a warning, so no number is silently doubled.

## A threaded loader measures what the loop waited

`ThreadDataLoader` fetches on a background thread. The bracket above measures
what the training loop waited for a batch, not what the producer spent building
it. A loader that keeps up reports a small Input Wait even while its thread
works through the whole step.

## Reading the measurements

The end-of-run summary reports one row per published step, so the step count
should equal the optimizer updates your trainer made. The example prints its own
update count for exactly this comparison.

Input Wait is the loop's idle time before a batch arrives, and forward, backward
and optimizer are the three compute phases. Whatever is left inside the step is
reported as residual. For a MONAI trainer that holds `zero_grad`, the loss,
decollation, postprocessing, any handler that runs before the step closes, and
on CPU the batch transfer as well.

A large Input Wait points at the input pipeline. A large residual points at work
around the model rather than in it, and the summary cannot say which part of it
without a profiler.

## Limitations

- Backward is measured between the handler's own two callbacks. Another handler
  registered between them falls inside that window.
- While a run is traced, `engine.inferer` is a timing proxy. Reads and writes
  both reach the real inferer, so state a handler stores on it survives, but
  the object is not MONAI's `Inferer` for an `isinstance` check, `repr` shows
  the proxy, and it does not deepcopy. The original is restored at every run
  exit, including after an exception.
- `engine.run` stays wrapped, and the engine keeps a marker naming its handler,
  for the life of the engine. Only `prepare_batch` and `inferer` are put back at
  a run exit.
- Optimizer time is `optimizer.step()` only. A step-interval scheduler, such as
  `LrScheduleHandler(epoch_level=False)`, runs on `ITERATION_COMPLETED`, so it
  is in neither the optimizer phase nor the step.
- CUDA is a documented recipe, not CI tested: the spleen notebook below ran on
  one T4, and CI covers CPU single-process runs only. `multi_process` and
  `multi_node` are not claimed, because the handler has no per-rank branch.

## Full example

[`examples/integrations/monai_minimal.py`](https://github.com/traceopt-ai/traceml/blob/main/examples/integrations/monai_minimal.py)
trains a small UNet on synthetic volumes and downloads nothing. Run it with:

```bash
traceml run examples/integrations/monai_minimal.py
```

The end-of-run summary reports Input Wait, forward, backward and optimizer time
for each published step.

## Try it on a real workload

[`examples/integrations/monai_dataloading_bottleneck.py`](https://github.com/traceopt-ai/traceml/blob/main/examples/integrations/monai_dataloading_bottleneck.py)
trains a 3D UNet on patches from the Medical Segmentation Decathlon spleen task.
Each flag changes one setting: the dataset class, the worker count, the loader
class, or mixed precision. Two runs that differ in one flag measure that setting
alone:

```bash
traceml run --mode summary --logs-dir logs --run-name spleen_1_baseline \
    examples/integrations/monai_dataloading_bottleneck.py --args --data-dir data
traceml run --mode summary --logs-dir logs --run-name spleen_2_workers \
    examples/integrations/monai_dataloading_bottleneck.py \
    --args --data-dir data --num-workers 4
traceml compare logs/spleen_1_baseline/final_summary.json \
    logs/spleen_2_workers/final_summary.json
```

The notebook runs six settings on one GPU, compares each adjacent pair, and
shows where the bottleneck moves as each one changes:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/traceopt-ai/traceml/blob/main/notebooks/monai_dataloading_bottleneck.ipynb)
