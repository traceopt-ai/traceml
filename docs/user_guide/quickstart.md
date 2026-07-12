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

### Alternative: launch your script directly

If you would rather launch training yourself with `python` or `torchrun`, start
the TraceML aggregator once, then run your script directly. `traceml.init(...)`
connects to the running aggregator:

```bash
# terminal 1
traceml serve --aggregator-host 127.0.0.1 --aggregator-port 29765

# terminal 2
python train.py
```

For torchrun and multi-node, bind the aggregator so other nodes can connect:

```bash
traceml serve --aggregator-bind-host 0.0.0.0 --aggregator-host <node0-ip> --aggregator-port 29765
torchrun ... train.py
```

In the direct-launch path you cannot pass `traceml run` flags, so runtime
settings come from `traceml.init(...)` arguments, then `TRACEML_*` environment
variables, then `traceml.yaml`, then built-in defaults. The aggregator endpoint
is set with `traceml serve` flags. If the aggregator is not reachable,
`traceml.init(...)` fails with a clear message after a short bounded retry. Pass
`traceml.init(disabled=True)` to run without tracing. See
[Public API](public-api.md#direct-launch-with-traceml-serve).

## 4. Read Your Diagnosis

```text
+----------------------------------------------------------------------------+
|  Step Time                                                                 |
|  - Diagnosis: INPUT STRAGGLER                                              |
|  - Stats: total 303.7ms | input 254.5ms | compute 259.5ms | wait 40.5ms    |
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
