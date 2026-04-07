# Use TraceML with W&B, MLflow, or TensorBoard

You do **not** need to replace your existing tracking stack to use TraceML.

TraceML is designed to work **alongside** tools like:

- Weights & Biases (W&B)
- MLflow
- TensorBoard
- Lightning loggers
- other experiment or metric tracking tools

A good mental model is:

- your existing stack tracks runs, configs, metrics, and artifacts
- TraceML explains **why training is slow**

---

## The short version

Use your existing tools for:

- experiment tracking
- loss / accuracy curves
- artifacts and checkpoints
- long-term dashboards
- team reporting

Use TraceML for:

- input bottlenecks
- compute-bound steps
- DDP stragglers
- wait-heavy behavior
- memory creep
- step-aware bottleneck diagnosis

TraceML is the **training bottleneck finder** in the stack.

---

## A common setup

A common setup looks like this:

- Hugging Face / Lightning / PyTorch loop for training
- W&B or MLflow for experiment tracking
- TraceML for bottleneck diagnosis during runs

That means:

- keep your current tracking workflow
- add TraceML when you need to understand throughput problems

---

## Why this setup makes sense

Tracking tools answer questions like:

- what config did this run use?
- what was the loss curve?
- which checkpoint belongs to this run?
- how did this run compare to the last one?

TraceML answers questions like:

- is the job input-bound or compute-bound?
- is one rank slower than the others?
- is the job spending time waiting?
- is memory drifting upward over time?
- which phase of the step is the real bottleneck?

These are different jobs.

---

## Example: W&B + TraceML

You can keep using W&B logging in your training script and launch the run through TraceML.

Example:

```python
import wandb

wandb.init(project="my-project")

for batch in dataloader:
    with trace_step(model):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["x"])
        loss = criterion(outputs, batch["y"])
        loss.backward()
        optimizer.step()

    wandb.log({"loss": loss.item()})
```

Run:

```bash
traceml run train.py
```

This gives you:

- W&B for run tracking
- TraceML for live bottleneck diagnosis

---

## Example: MLflow + TraceML

You can do the same with MLflow.

```python
import mlflow

mlflow.start_run()

for batch in dataloader:
    with trace_step(model):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["x"])
        loss = criterion(outputs, batch["y"])
        loss.backward()
        optimizer.step()

    mlflow.log_metric("loss", float(loss.item()), step=step)
```

Run:

```bash
traceml run train.py
```

This gives you:

- MLflow for experiment tracking
- TraceML for bottleneck diagnosis

---

## Keeping terminal output clean

If you use TraceML in CLI mode together with other loggers, the terminal can get noisy.

Good ways to reduce that:

- disable `tqdm` progress bars
- reduce extra console logging
- use the local UI if the terminal feels crowded

Launch the local UI with:

```bash
traceml run train.py --mode=dashboard
```

This is often the cleanest option when you want:

- existing loggers to keep running
- TraceML visible at the same time
- less terminal clutter

---

## Do I need to disable W&B or MLflow?

No.

For the cleanest local demo experience, you *may* temporarily disable external reporting, but it is not required.

For example, in some Hugging Face runs you might set:

```python
report_to="none"
disable_tqdm=True
```

That is just to keep the terminal cleaner during local diagnosis runs.

It is not a TraceML requirement.

---

## Best way to adopt TraceML

The easiest path is:

1. keep your existing tracking setup
2. add `trace_step(model)` or a supported integration
3. launch the run with `traceml run ...`
4. use TraceML when training feels slower than expected

This is much easier than trying to replace your tracking stack.

---

## When TraceML is most useful alongside other tools

TraceML is especially useful when:

- dashboards look normal but throughput is still poor
- one distributed rank is slowing the job down
- memory grows over time but you do not know why
- you want a fast answer before using a heavyweight profiler

---

## What TraceML is not trying to replace

TraceML is not trying to replace:

- experiment tracking
- artifact storage
- model registry
- team dashboards
- long-term run management

Its job is narrower:

- help you find the training bottleneck quickly

---

## Recommended workflow

A practical workflow is:

1. run training normally with your existing stack
2. if training is slower than expected, launch with TraceML
3. read the diagnosis
4. inspect the called-out bottleneck
5. only then move to deeper profiling if needed

This keeps TraceML in the part of the workflow where it adds the most value.

---

## Related docs

- [Quickstart](quickstart.md)
- [How to Read TraceML Output](how-to-read-output)
- [Hugging Face Trainer](huggingface.md)
- [PyTorch Lightning](lightning.md)
