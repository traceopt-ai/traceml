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

This starts the live browser dashboard and prints the local URL, usually
`http://127.0.0.1:8765`. Open it to see live bottleneck diagnostics while the
job runs.

<details>
<summary>Running on a remote server?</summary>

SSH into the server and start TraceML there:

```bash
traceml run train.py
```

TraceML prints a tunnel command like this:

```bash
ssh -L 8765:127.0.0.1:8765 user@remote-host
```

Copy that tunnel command into a local terminal on your laptop. Leave the
training command running on the server, then open `http://127.0.0.1:8765`
locally.

</details>

If you want a live view without a browser or SSH tunnel, use
`traceml run train.py --mode=cli`. Use `traceml run train.py --mode=summary`
for headless jobs, CI, DDP, FSDP, Slurm, or multi-node runs.

To try the same flow with a checked-in example first:

```bash
traceml run examples/quickstart.py
```

TraceML writes:

```text
logs/<run_name>/final_summary.json
logs/<run_name>/final_summary.txt
```

Add `--html-report` (`traceml run train.py --html-report`) to also write a
shareable `final_summary.html`. See
[Reading the output](reading-output.md#shareable-html-report).

TraceML finalizes summaries after training by settling late telemetry, closing
SQLite cleanly, and checkpointing WAL. Large distributed jobs can raise that
end-of-run budget with `--finalize-timeout-sec <seconds>`.

In `--mode=summary`, if training finishes but TraceML cannot produce
`final_summary.json`, `traceml run` exits non-zero, so a silently missing
summary fails loudly instead of passing. (Reused history databases keep any
rows written by older releases; new runs only write the structured projection
tables.)

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
# aggregator node: --nnodes x --nproc-per-node = total workers, so the
# aggregator waits for every rank before finalizing
traceml serve --aggregator-bind-host 0.0.0.0 --aggregator-host <node0-ip> \
  --aggregator-port 29765 --nnodes <N> --nproc-per-node <M>

# each training node: point workers at node 0's aggregator, then launch
TRACEML_AGGREGATOR_HOST=<node0-ip> TRACEML_AGGREGATOR_PORT=29765 \
  torchrun ... train.py
```

In the direct-launch path you cannot pass `traceml run` flags, so runtime
settings come from `traceml.init(...)` arguments, then `TRACEML_*` environment
variables, then `traceml.yaml`, then built-in defaults. The aggregator endpoint
is set with `traceml serve` flags. If the aggregator is not reachable,
`traceml.init(...)` prints one warning and continues without tracing (a no-op),
so instrumentation never crashes your training run. Pass
`traceml.init(on_missing_aggregator="raise")` to fail hard instead, or
`traceml.init(disabled=True)` to silence it entirely. See
[Public API](public-api.md#direct-launch-with-traceml-serve).

## 4. Read Your Diagnosis

```text
+----------------------------------------------------------------------------+
|  TraceML Run Summary | duration 40.1s                                      |
+----------------------------------------------------------------------------+
|                                                                            |
|  TraceML Verdict: INPUT STRAGGLER / CRITICAL                               |
|  Why: Rank r0 input wait was 254.5ms vs median rank r1 at 3.8ms.           |
|  Next: Inspect dataloader, collate_fn, preprocessing, and storage on the   |
|  slow rank.                                                                |
|                                                                            |
|  Section Status                                                            |
|  Section       Status                  Severity                            |
|  ------------------------------------------------                          |
|  Step Time     INPUT STRAGGLER         CRITICAL                            |
|  System        LOW GPU UTIL            INFO                                |
|  Process       NORMAL                  INFO                                |
|  Step Memory   BALANCED                INFO                                |
|                                                                            |
|  System Evidence                                                           |
|  Metric          Median        Worst         Skew        Scope             |
|  --------------------------------------------------------------------------|
|  CPU Util        18.4%         71.2%         52.8pp      node=n1           |
|  GPU Util        14.0%         0.0%          14.0pp      node=n0           |
|  GPU Memory      6.20GB        8.90GB        43.5%       node=n1           |
|  GPU Temp        42C           58C           16C         node=n1           |
|                                                                            |
|  Step Time Evidence                                                        |
|  Phase           Median        Worst         Skew        Scope             |
|  --------------------------------------------------------------------------|
|  Total           303.7ms       304.1ms       0.1%        rank=r0 node=n0   |
|  Input Wait      3.8ms         254.5ms       6597.4%     rank=r0 node=n0   |
|  Compute         259.5ms       261.0ms       0.6%        rank=r2 node=n1   |
+----------------------------------------------------------------------------+
```

In this example, rank 0 is the slow input rank, which can hold back the aligned
distributed step.

## Useful Next Commands

Live terminal view:

```bash
traceml run train.py --mode=cli
```

Browser view, explicit:

```bash
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
