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
- wait-heavy behavior
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
import traceml
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
- TraceML for live bottleneck diagnosis

---

## Log the final TraceML summary to W&B

If you want a quieter run and a structured TraceML summary you can log at the
end, launch in summary mode:

```bash
traceml run train.py --mode=summary
```

Then request the finalized TraceML summary near the end of your script and log
selected fields into W&B:

```python
import traceml
import wandb

wandb.init(project="my-project")

# training loop ...

summary = traceml.final_summary(print_text=True)
if summary is not None:
    wandb.summary["traceml/step_time_status"] = summary["step_time"][
        "diagnosis"
    ]["status"]
    wandb.summary["traceml/step_memory_status"] = summary["step_memory"][
        "diagnosis"
    ]["status"]
    wandb.summary["traceml/duration_s"] = summary.get("duration_s")
```

This lets W&B stay your experiment system of record while TraceML contributes a
clean bottleneck diagnosis at the end of the run.

If you also keep the TraceML final summary JSON as a run artifact, you can
compare runs later with `traceml compare`.

---

## Example: MLflow + TraceML

You can do the same with MLflow.

```python
import traceml
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

TraceML can also return a structured final summary for MLflow logging:

```python
import traceml
import mlflow

mlflow.start_run()

# training loop ...

summary = traceml.final_summary(print_text=True)
if summary is not None:
    mlflow.log_param(
        "traceml_step_time_status",
        summary["step_time"]["diagnosis"]["status"],
    )
    mlflow.log_param(
        "traceml_step_memory_status",
        summary["step_memory"]["diagnosis"]["status"],
    )
    mlflow.log_dict(summary, "traceml/final_summary.json")
```

This is a good fit when you want both a compact diagnosis in your run metadata
and the full TraceML summary JSON attached to the run.

That attached JSON is also a good input for `traceml compare` later.

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

1. run training with TraceML summary mode
2. keep the TraceML final summary JSON as a W&B or MLflow artifact
3. compare two saved summaries locally with `traceml compare`

See [Compare Runs](compare.md).

---

## Keeping terminal output clean

If you use TraceML in CLI mode together with other loggers, the terminal can get noisy.

Good ways to reduce that:

- disable `tqdm` progress bars
- reduce extra console logging
- use `--mode=summary` if you only want the final TraceML summary
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

A clean adoption path is:

1. start with `traceml run train.py`
2. use `--mode=summary` when you want a quieter run and a structured final summary
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

- [Quickstart](quickstart.md)
- [Compare Runs](compare.md)
- [How to Read TraceML Output](how-to-read-output.md)
- [FAQ](faq.md)
