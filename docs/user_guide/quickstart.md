# TraceML Quickstart

Diagnose slow training with Hugging Face Trainer, PyTorch Lightning, RF-DETR, or a custom PyTorch loop.

This guide assumes Python 3.10+ and a working training environment with your
framework already installed.

## 1. Install TraceML

Add TraceML to the environment where you run training:

```bash
pip install traceml-ai
```

## 2. Run Your Training

### Hugging Face Trainer, PyTorch Lightning, and RF-DETR

Launch your existing script through TraceML:

```diff
- python train.py
+ traceml run train.py
```

TraceML detects the framework and attaches instrumentation automatically.
No code changes are required for these standard training paths.


### Custom PyTorch Loop

Initialize TraceML once and mark the optimizer-step boundary:

```diff
+   import traceml_ai as traceml

+   traceml.init(mode="auto")

    for batch in dataloader:
+       with traceml.trace_step(model):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["x"])
            loss = criterion(outputs, batch["y"])
            loss.backward()
            optimizer.step()
```

Then run:

```bash
traceml run train.py
```

See the [core API](public-api.md) for custom input sources and more explicit
control.

## 3. Read the Diagnosis

When training finishes, TraceML identifies the likely bottleneck, explains the
evidence, and suggests what to inspect next. For example:

```text
TraceML Run Summary

Verdict: INPUT-BOUND (CRITICAL)
Why: Input Wait took 64% of Step Time.
Next: Increase workers, prefetch, or storage throughput.

Step Time       200.4 ms  100%
├─ Input Wait   128.0 ms   64%  ◀ cause
├─ Forward       24.0 ms   12%
├─ Backward      38.0 ms   19%
├─ Optimizer      6.0 ms    3%
├─ H2D            0.4 ms   <1%
└─ Residual        3.6 ms    2%
```

Your result depends on the workload and hardware. TraceML also reports system,
process, memory, and distributed-rank evidence when those signals are
available.

Reports are saved under:

```text
logs/<run_name>/final_summary.json
logs/<run_name>/final_summary.txt
```

Use [How to Read TraceML Output](reading-output.md) for the full report and
verdict reference.

## Try It Without Your Own Script

In an environment with Hugging Face Trainer and TraceML installed, try this
small example. It contains no TraceML code and downloads no model or dataset.
The examples are not included in the PyPI wheel, so run it from a repository
checkout:

```bash
git clone https://github.com/traceopt-ai/traceml.git
cd traceml
traceml run examples/integrations/huggingface_trainer_minimal.py \
  --args --steps 20
```

## Next Steps

- Understand every field in the [saved and terminal output](reading-output.md).
- Compare a baseline and candidate with [Compare Runs](compare.md).
- Add a threshold check with the [CI regression guard](regression-guard.md).
- Run DDP, FSDP, Slurm, or multi-node jobs with
  [Distributed Training](distributed-training.md).
- Use terminal or browser views through the documented
  [`traceml run` options](public-api.md#cli).
- Check the full list of [training integrations](integrations.md).
