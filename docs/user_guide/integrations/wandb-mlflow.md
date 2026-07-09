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
- loss and accuracy curves
- artifacts and checkpoints
- long-term dashboards
- team reporting

Use TraceML for:

- input bottlenecks
- compute-bound steps
- DDP stragglers
- residual-heavy behavior
- memory creep
- step-aware bottleneck diagnosis
- compact run-to-run comparison from saved TraceML summary JSON files

TraceML is the training bottleneck finder in the stack.

---

## A common setup

A common setup looks like this:

- Hugging Face, Lightning, or a normal PyTorch loop for training
- W&B or MLflow for experiment tracking
- TraceML for bottleneck diagnosis during runs and end-of-run summaries

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
import traceml_ai as traceml
import wandb

wandb.init(project="my-project")

for batch in dataloader:
    with traceml.trace_step(model):
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
- TraceML for the final bottleneck summary

---

## Log the final TraceML summary to W&B

For quiet tracker runs, launch TraceML in summary mode:

```bash
traceml run train.py --mode=summary
```

Then request TraceML's compact tracker summary near the end of your script and
log it into W&B:

```python
import traceml_ai as traceml
import wandb

wandb.init(project="my-project")

# training loop ...

summary = traceml.summary(print_text=True)
if summary is not None:
    wandb.log(summary)
```

This lets W&B stay your experiment system of record while TraceML contributes a
clean bottleneck diagnosis at the end of the run.

`traceml.summary()` returns a flat dict of diagnosis statuses and global
average metrics. It is derived from the canonical `final_summary.json` artifact,
which TraceML generates once per run and reuses on later calls. Keep that JSON
as a run artifact when you want to use `traceml compare`.

---

## Example: MLflow + TraceML

You can do the same with MLflow.

```python
import traceml_ai as traceml
import mlflow

mlflow.start_run()

for batch in dataloader:
    with traceml.trace_step(model):
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

## Log the final TraceML summary to MLflow

TraceML can also return a compact tracker summary for MLflow logging:

```python
import traceml_ai as traceml
import mlflow

mlflow.start_run()

# training loop ...

summary = traceml.summary(print_text=True)
if summary is not None:
    numeric = {
        k: v for k, v in summary.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    tags = {
        k.replace("/", "."): v for k, v in summary.items()
        if isinstance(v, str)
    }

    mlflow.log_metrics(numeric)
    mlflow.set_tags(tags)
```

This is a good fit when you want a compact diagnosis in your run metadata.

If you also want the full TraceML artifact, call `traceml.final_summary()` and
attach that JSON to the run:

```python
full = traceml.final_summary()
if full is not None:
    mlflow.log_dict(full, "traceml/final_summary.json")
```

That attached JSON is a good input for `traceml compare` later.

---

## Compare runs later

Once you have TraceML final summary JSON files for two runs, compare them with:

```bash
traceml compare run_a.json run_b.json
```

This is useful when:

- a training change may have made runs slower
- a dataloader or preprocessing change may have shifted the bottleneck
- a model or optimizer change may have moved time into a different phase
- dashboards look similar but throughput still feels worse

A practical workflow is:

1. run training with TraceML
2. keep the TraceML final summary JSON as a W&B or MLflow artifact
3. compare two saved summaries locally with `traceml compare`

See [Compare Runs](../compare.md).

---

## Keeping terminal output clean

If you use TraceML in CLI mode together with other loggers, the terminal can get noisy.

Good ways to reduce that:

- disable `tqdm` progress bars
- reduce extra console logging
- use `--mode=summary` if you only want the final TraceML summary
- use the local UI on a single-node run if the terminal feels crowded

Launch the local UI with:

```bash
pip install "traceml-ai[dashboard]"
traceml run train.py --mode=dashboard
```

Dashboard mode is intended for single-node runs. Multi-node runs default to
summary mode; log the final summary artifact from that run.

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

A clean adoption path is:

1. start with `traceml run train.py`
2. use `--mode=cli` or `--mode=dashboard` when you want live feedback on a single-node run
3. log selected TraceML summary fields into W&B or MLflow if useful
4. keep the TraceML final summary JSON for important runs
5. compare two saved runs later with `traceml compare`

This usually gives the best balance between:

- low overhead
- clear diagnosis
- compatibility with your current stack
- simple run-to-run review

---

## Related docs

- [Quickstart](../quickstart.md)
- [Compare Runs](../compare.md)
- [How to Read TraceML Output](../reading-output.md)
- [FAQ](../faq.md)
