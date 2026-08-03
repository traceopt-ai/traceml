# TraceML Quickstart

Get from install to your first TraceML diagnosis in a few minutes.

TraceML diagnoses input, compute, waiting, memory, and distributed-rank
bottlenecks in PyTorch training. It complements experiment trackers and deeper
operator- or kernel-level profilers by showing you where to investigate first.

TraceML runs with your existing PyTorch script and writes a structured
`final_summary.json` plus a human-readable `final_summary.txt` at the end of
the run.

## 1. Install

With pip:

```bash
pip install traceml-ai
```

Or in a project managed by [uv](https://docs.astral.sh/uv/):

```bash
uv add traceml-ai
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

By default, TraceML runs without a live UI, prints a compact final diagnosis
when training ends, and writes `final_summary.json` and `final_summary.txt`.

To try the same flow with a checked-in example first:

```bash
traceml run examples/quickstart.py
```

TraceML writes:

```text
logs/<run_name>/final_summary.json
logs/<run_name>/final_summary.txt
```

<details>
<summary><strong>Want live diagnostics during training?</strong></summary>

Use terminal mode for a live view over a local shell or SSH session:

```bash
traceml run train.py --mode=cli
```

Use dashboard mode for the live browser view:

```bash
traceml run train.py --mode=dashboard
```

When training runs on a remote server, TraceML prints an SSH tunnel command
like this:

```bash
ssh -L 8765:127.0.0.1:8765 user@remote-host
```

Run that command on your local machine, leave training running remotely, and
open `http://127.0.0.1:8765`. Live views do not change the saved end-of-run
artifacts. See [How to Read TraceML Output](reading-output.md#what-the-summary-cli-and-local-ui-show).

</details>

<details>
<summary><strong>Try TraceML in Colab</strong></summary>

- Any PyTorch loop: data-loading bottleneck before and after [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/traceopt-ai/traceml/blob/main/notebooks/data_loading_bottleneck.ipynb)
- Hugging Face Trainer: data-loading bottleneck before and after [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/traceopt-ai/traceml/blob/main/notebooks/huggingface_dataloading_bottleneck.ipynb)

</details>

<details>
<summary><strong>Try TraceML with Docker</strong></summary>

From a repository checkout, build the image and run the default CPU demo:

```bash
docker build -t traceml-demo .
docker run --rm traceml-demo
```

The self-contained slow-DataLoader demo requires no external dataset or GPU
and should finish with an input-bound diagnosis. If the host has the NVIDIA
Container Toolkit and a CUDA-compatible PyTorch build, optionally expose the
GPU:

```bash
docker run --rm --gpus all traceml-demo
```

</details>

<details>
<summary><strong>Launching with <code>python</code> or <code>torchrun</code> directly?</strong></summary>

Start one TraceML aggregator, then launch the instrumented script yourself:

```bash
# terminal 1
traceml serve --aggregator-host 127.0.0.1 --aggregator-port 29765

# terminal 2
python train.py
```

See [Direct launch with `traceml serve`](public-api.md#direct-launch-with-traceml-serve)
for `torchrun`, multi-node networking, configuration precedence, and behavior
when the aggregator cannot be reached.

</details>

Summary mode is also the default for DDP, FSDP, Slurm, and multi-node runs.
Follow [Distributed Training](distributed-training.md) or the
[Slurm guide](slurm.md) for the supported launch patterns and current
limitations.

<details>
<summary><strong>Advanced finalization behavior</strong></summary>

TraceML waits for late telemetry before writing the final artifacts. Large
distributed jobs can raise that end-of-run budget with
`--finalize-timeout-sec <seconds>`.

In `--mode=summary`, if training finishes but TraceML cannot produce
`final_summary.json`, `traceml run` exits non-zero, so a silently missing
summary fails loudly instead of passing. The
[Distributed Training guide](distributed-training.md) explains timeout tuning
for larger jobs.

</details>

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

TraceML also triages input-pipeline delays, compute-heavy steps, waiting and
residual time, memory growth, and run regressions. Start with
[How to Read TraceML Output](reading-output.md), then choose the matching
troubleshooting guide below when you need a focused investigation.

## Useful Next Commands

Reprint a saved summary:

```bash
traceml view logs/<run_name>/final_summary.json
```

Create a self-contained HTML report during the run or from a saved summary:

```bash
traceml run train.py --html-report
traceml view logs/<run_name>/final_summary.json --html
```

Compare two runs:

```bash
traceml compare run_a/final_summary.json run_b/final_summary.json
```

## Next Steps

- **Understand the result:** [How to Read Output](reading-output.md) and
  [Compare Runs](compare.md).
- **Investigate a bottleneck:** [slow training](../guides/slow-pytorch-training.md),
  [input pipeline](../guides/pytorch-input-pipeline-bottleneck.md),
  [low GPU utilization](../guides/low-gpu-utilization-pytorch.md),
  [DDP rank stragglers](../guides/ddp-slow-training-rank-straggler.md), or
  [memory creep](../guides/pytorch-memory-creep.md).
- **Use your training stack:** [Hugging Face, Lightning, Ray, DeepSpeed,
  W&B, and MLflow](integrations.md).
- **Run at scale:** [Distributed Training](distributed-training.md) and
  [Slurm](slurm.md).
- **Check behavior and limitations:** [FAQ](faq.md) and
  [Public API](public-api.md).
