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

On your laptop or workstation, this starts the live browser dashboard and
prints the local URL, usually `http://127.0.0.1:8765`.

If training runs on a remote server, SSH into the server and run the same
command there. TraceML will print a tunnel command like this:

```bash
ssh -L 8765:127.0.0.1:8765 user@remote-host
```

Copy that tunnel command into a terminal on your laptop, leave the training
command running on the server, then open `http://127.0.0.1:8765` locally.

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
