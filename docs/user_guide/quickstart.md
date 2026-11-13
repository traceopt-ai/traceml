# TraceML Quickstart

Get a first saved report from [your training script](#use-your-training-script)
or [the included example](#try-the-included-example). TraceML requires Python
3.10 or later.

## Use your training script

Install TraceML in the Python environment where you run training. It should
already contain PyTorch and any trainer your script uses:

```bash
pip install traceml-ai
```

### Hugging Face Trainer, PyTorch Lightning, and RF-DETR

Standard [Hugging Face Trainer](integrations/huggingface.md),
[PyTorch Lightning](integrations/lightning.md), and
[RF-DETR object detection](integrations/rfdetr.md) scripts need no TraceML code
changes. Launch your existing script through TraceML:

```diff
- python train.py
+ traceml run train.py
```

### Custom PyTorch Loop

For a plain PyTorch loop, initialize TraceML once and wrap each training step,
including its optimizer update:

```diff
+ import traceml_ai as traceml

+ traceml.init(mode="auto")

  for batch in dataloader:
+     with traceml.trace_step(model):
          optimizer.zero_grad(set_to_none=True)
          outputs = model(batch["x"])
          loss = criterion(outputs, batch["y"])
          loss.backward()
          optimizer.step()
```

Then launch the instrumented script:

```bash
traceml run train.py
```

See the [core API](public-api.md) for custom input sources and more explicit
control, or the [integration guides](integrations.md) for other training setups.

## Try the included example

The repository's `examples/quickstart.py` trains on generated data, so it needs
no dataset download. It runs on CPU when CUDA is unavailable and already has
the custom-loop instrumentation shown above. Use a repository checkout of this
documentation version; examples are not included in the PyPI package:

```bash
git clone --branch version_0.5.2 https://github.com/traceopt-ai/traceml.git
cd traceml
python -m pip install -e ".[torch]"
traceml run examples/quickstart.py --args --steps 20
```

## Read the diagnosis and saved report

When training finishes, TraceML prints a verdict, the supporting measurements,
and a suggested next investigation. The terminal shows the report path; the
files are saved under the generated run name:

```text
logs/<run-name>/final_summary.json
logs/<run-name>/final_summary.txt
```

The included example checks the first-run workflow, not a particular
bottleneck. This excerpt illustrates how to read a diagnosis; your verdict and
measurements may differ:

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
available. See [How to Read TraceML Output](reading-output.md#start-with-the-diagnosis)
for the full report and verdict reference.

## Next Steps

- Understand every field in the [saved and terminal output](reading-output.md).
- Compare a baseline and candidate with [Compare Runs](compare.md).
- Add a threshold check with the [CI regression guard](regression-guard.md).
- Run DDP, FSDP, Slurm, or multi-node jobs with
  [Distributed Training](distributed-training.md).
- Use terminal or browser views through the documented
  [`traceml run` options](public-api.md#cli).
- Check the full list of [training integrations](integrations.md).
