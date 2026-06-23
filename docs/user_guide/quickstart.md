# TraceML Quickstart

Get from install to your first TraceML diagnosis in a few minutes.

TraceML runs with your existing PyTorch script and writes a structured
`final_summary.json` plus a human-readable `final_summary.txt` at the end of
the run.

## 1. Install

```bash
pip install traceml-ai
```

Using Hugging Face, Lightning, Ray, W&B, or MLflow? See
[Use With Your Stack](integrations.md).

## 2. Instrument Your Training Step

Add TraceML initialization once, then wrap the training step body:

```python
import traceml_ai as traceml

traceml.init(mode="auto")

for batch in dataloader:
    with traceml.trace_step(model):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["x"])
        loss = criterion(outputs, batch["y"])
        loss.backward()
        optimizer.step()
```

Wrap the work from `zero_grad(...)` through `optimizer.step()`.

## 3. Run Your Script

```bash
traceml run train.py
```

To try the same flow with a checked-in example first:

```bash
traceml run examples/quickstart.py --mode=summary
```

TraceML writes:

```text
logs/<run_name>/final_summary.json
logs/<run_name>/final_summary.txt
```

Add `--html-report` (`traceml run train.py --html-report`) to also write a
shareable `final_summary.html`. See
[Reading the output](reading-output.md#shareable-html-report).

For DDP, FSDP, and multi-node launches, see
[Distributed Training](distributed-training.md).

## 4. Read Your Diagnosis

```text
+----------------------------------------------------------------------------+
|  Step Time                                                                 |
|  - Diagnosis: INPUT STRAGGLER                                              |
|  - Stats: total 303.7ms | input 254.5ms | compute 259.5ms             |
|  - Residual: 40.5ms                                                       |
|  - Why: r0 input was slower than median global rank (254.5/3.8ms).         |
+----------------------------------------------------------------------------+
```

In this example, rank 0 is the slow input rank, which can hold back the aligned
distributed step.

## Useful Next Commands

Live terminal view:

```bash
traceml run train.py --mode=cli
```

Browser view:

```bash
pip install "traceml-ai[dashboard]"
traceml run train.py --mode=dashboard
```

Compare two runs:

```bash
traceml compare run_a/final_summary.json run_b/final_summary.json
```

## Next Steps

- [How to Read Output](reading-output.md)
- [Compare Runs](compare.md)
- [Distributed Training](distributed-training.md)
- [Use With Your Stack](integrations.md)
- [FAQ](faq.md)
