# TraceML Quickstart

Get from install to your first TraceML diagnosis in a few minutes.

TraceML diagnoses input, compute, waiting, memory, and distributed-rank
bottlenecks in PyTorch training. It complements experiment trackers and deeper
operator- or kernel-level profilers by showing you where to investigate first.
TraceML runs with your existing PyTorch script and writes a structured
`final_summary.json` plus a human-readable `final_summary.txt` at the end of
the run.

## 1. Install

TraceML requires Python 3.10+. This path assumes your project already has
PyTorch. For a fresh source checkout, use the optional example path below;
`.[torch]` installs the supported PyTorch dependencies (PyTorch 2.5+).

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

```diff
    import traceml_ai as traceml

+   traceml.init(mode="auto")

    for batch in dataloader:
+       with traceml.trace_step(model):
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

## Or Run the Checked-In Example

To try TraceML before modifying your own script, clone the repository. The
examples are not included in the PyPI wheel:

```bash
git clone https://github.com/traceopt-ai/traceml.git
cd traceml
pip install ".[torch]"
traceml run examples/quickstart.py --mode=summary
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

Docker also builds from a repository checkout, rather than from the PyPI
package. From the cloned `traceml` directory, build and run the default CPU
demo:

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

Here is an illustrative single-process result where a slow DataLoader leaves
the GPU waiting:

```text
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|  TraceML Run Summary                                                                                                                                     |
|  bert_finetune · 1 rank · 1 GPU observed · 256 common steps · 52.4s                                                                                      |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                                                          |
|  Verdict: INPUT-BOUND  (CRITICAL)                                                                                                                        |
|  Why: Input Wait took 64% of Step Time.                                                                                                                  |
|  Next: Increase workers, prefetch, or storage throughput.                                                                                                |
|                                                                                                                                                          |
|  STEP TIMING (Window Average), GPU Clock                      ||  STEP MEMORY: BALANCED                                                                  |
|  Step Time           200.4 ms  100%                           ||  Evidence: None                                                                         |
|  ├─ Input Wait       128.0 ms   64%  ◀  cause                 ||                                                                                         |
|  ├─ Compute           68.0 ms   34%                           ||  avg per-step peak           avg                                                        |
|  │  ├─ Forward        24.0 ms   12%                           ||  Allocated                   2.9 GB                                                     |
|  │  ├─ Backward       38.0 ms   19%                           ||  Reserved                    3.2 GB                                                     |
|  │  └─ Optimizer       6.0 ms    3%                           ||                                                                                         |
|  ├─ H2D                0.4 ms   <1%                           ||                                                                                         |
|  └─ Residual           3.6 ms    2%                           ||                                                                                         |
|  DataLoader fetch: 120.0 ms (CPU, supplemental)               ||                                                                                         |
|                                                                                                                                                          |
|  SYSTEM METRICS: LOW GPU UTIL                                 ||  PROCESS METRICS: NORMAL                                                                |
|  Evidence: GPU utilization averaged 24%.                      ||  Evidence: None                                                                         |
|                                                               ||                                                                                         |
|                         avg                                   ||                       avg                                                               |
|  CPU                    18%                                   ||  CPU capacity         14%                                                               |
|  RAM used               6.2 GB (19%)                          ||  RSS used             3.1 GB (10%)                                                      |
|  GPU util               24%                                   ||  CUDA used            2.9 GB                                                            |
|  GPU memory/device      3.3 GB (21%)                          ||  CUDA reserved        3.2 GB (20%)                                                      |
|  GPU temperature        42C                                   ||                                                                                         |
|  GPU power              58W                                   ||                                                                                         |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Full evidence: logs/bert_finetune/final_summary.json  (--html-report)                                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
```

Your verdict and measurements depend on your workload and hardware.

TraceML can surface input-bound, H2D-bound, compute-bound, and residual-heavy
patterns; distributed jobs can also identify rank stragglers. Its other
sections report memory growth, and [Compare Runs](compare.md) identifies
regressions between saved summaries.

<details>
<summary><strong>Running DDP or multi-node training? See a rank-straggler diagnosis</strong></summary>

```text
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|  TraceML Run Summary                                                                                                                                     |
|  ddp_pretrain · 4/4 ranks · 4 GPUs observed · 2/2 nodes · 250 common steps · 40.1s                                                                       |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                                                                                          |
|  Verdict: INPUT STRAGGLER  (CRITICAL)                                                                                                                    |
|  Why: R0/N0 waited 254.5 ms for input; R1/N0 waited 3.8 ms for input.                                                                                    |
|  Next: Inspect input wait on the slow rank.                                                                                                              |
|  Scope: N = node · R = global rank · G = GPU index                                                                                                       |
|                                                                                                                                                          |
|  STEP TIMING (Median R1/N0), GPU Clock                        ||  STEP MEMORY: BALANCED · 4/4 ranks                                                      |
|  Step Time           303.7 ms  100%                           ||  Evidence: None                                                                         |
|  ├─ Input Wait         3.8 ms    1%                           ||                                                                                         |
|  ├─ Compute          259.5 ms   85%                           ||  avg per-step peak           median rank avg     worst rank avg                         |
|  │  ├─ Forward        80.0 ms   26%                           ||  Allocated                   8.5 GB              9.4 GB, R2/N1                          |
|  │  ├─ Backward      169.5 ms   56%                           ||  Reserved                    8.9 GB              9.8 GB, R2/N1                          |
|  │  └─ Optimizer      10.0 ms    3%                           ||                                                                                         |
|  ├─ H2D                1.1 ms   <1%                           ||                                                                                         |
|  └─ Residual          39.3 ms   13%                           ||                                                                                         |
|  DataLoader fetch: 3.7 ms (CPU, supplemental)                 ||                                                                                         |
|                                                                                                                                                          |
|  SYSTEM METRICS: LOW GPU UTIL · 2/2 nodes                     ||  PROCESS METRICS: NORMAL · 4/4 ranks                                                    |
|  Evidence: GPU utilization averaged 14%.                      ||  Evidence: None                                                                         |
|                                                               ||                                                                                         |
|                         median node avg   worst node avg      ||                       median rank avg   worst rank avg                                  |
|  CPU                    18%               26%, N1             ||  CPU capacity         12%               81%, R2/N1                                      |
|  RAM used               16.0 GB (27%)     20.8 GB (35%), N1   ||  RSS used             3.1 GB (10%)      5.4 GB (17%), R1/N0                             |
|  GPU util               9%                9%, N1              ||  CUDA used            2.9 GB            4.6 GB, R3/N1                                   |
|  GPU memory/device      5.0 GB (31%)      7.0 GB (44%), N1    ||  CUDA reserved        3.2 GB (20%)      6.8 GB (43%), R3/N1                             |
|  GPU temperature        58C               70C, N1             ||                                                                                         |
|  GPU power              220W              280W, N1            ||                                                                                         |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Full evidence: logs/ddp_pretrain/final_summary.json  (--html-report)                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
```

In this example, rank 0 is the slow input rank, which can hold back the aligned
distributed step.

</details>

For the full verdict reference and recommended next actions, start with
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
