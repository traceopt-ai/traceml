<div align="center">

# TraceML

**Find out why your PyTorch training is slow—before it wastes GPU hours.**

[![PyPI version](https://img.shields.io/pypi/v/traceml-ai.svg)](https://pypi.org/project/traceml-ai/)
[![CI](https://github.com/traceopt-ai/traceml/actions/workflows/ci.yml/badge.svg)](https://github.com/traceopt-ai/traceml/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/traceopt-ai/traceml/blob/main/LICENSE)
[![GitHub stars](https://badgen.net/github/stars/traceopt-ai/traceml?icon=github)](https://github.com/traceopt-ai/traceml)

[**Quickstart**](#quickstart) •
[**Try in Colab**](https://colab.research.google.com/github/traceopt-ai/traceml/blob/main/notebooks/data_loading_bottleneck.ipynb) •
[**Integrations**](https://traceopt-ai.github.io/traceml/user_guide/integrations/) •
[**Documentation**](https://traceopt-ai.github.io/traceml/) •
[**GitHub Issues**](https://github.com/traceopt-ai/traceml/issues)

</div>

**TraceML is an open-source tool that explains why PyTorch training is slow.**
At the end of a run, it gives you:

- **A diagnosis:** data loading, GPU compute, waiting, memory growth, or a slow
  distributed worker.
- **The evidence:** timing, CPU/GPU usage, memory, and per-worker measurements.
- **The next step:** what part of your training setup to investigate first.

### Example diagnosis

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
|  Step Time           200.4 ms  100%                           ||                                                                                         |
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
|  Evidence: GPU utilization averaged 24%.                      ||                                                                                         |
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

TraceML produces this diagnosis at the end of the instrumented training run.

Want the complete evidence? Jump to the
[single-run and distributed example reports](#example-reports).

## Quickstart

### 1. Install

TraceML expects an existing PyTorch project. Install the TraceML package with:

```bash
pip install traceml-ai
```

Using [uv](https://docs.astral.sh/uv/) instead? Run `uv add traceml-ai`.

### 2. Instrument the training step

Add TraceML around the core step in your existing PyTorch training script:

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

### 3. Run

```bash
traceml run train.py
```

Summary mode is the default. TraceML prints the final diagnosis and writes
`final_summary.json` and `final_summary.txt` under `logs/<run_name>/`.

No training script ready? [Try the Colab example](https://colab.research.google.com/github/traceopt-ai/traceml/blob/main/notebooks/data_loading_bottleneck.ipynb).

## Example Reports

<details>
<summary><strong>See the complete single-run report</strong></summary>

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
|  Step Time           200.4 ms  100%                           ||                                                                                         |
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
|  Evidence: GPU utilization averaged 24%.                      ||                                                                                         |
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

</details>

<details>
<summary><strong>Running distributed training? See a rank-straggler diagnosis</strong></summary>

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
|  Step Time           303.7 ms  100%                           ||                                                                                         |
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
|  Evidence: GPU utilization averaged 14%.                      ||                                                                                         |
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

</details>

Read [How to Read TraceML Output](https://traceopt-ai.github.io/traceml/user_guide/reading-output/)
for the complete field definitions, diagnosis rules, evidence, and recommended
actions.

## What TraceML Diagnoses

| Diagnosis | Where to investigate |
|---|---|
| Input-bound | DataLoader workers, transforms, tokenization, collation, or storage |
| H2D-bound | Pinned memory, non-blocking copies, batch size, or transfer overlap |
| Compute-bound | Model compute, mixed precision, batch size, or deeper profiling |
| Residual-heavy | Work outside traced phases, CPU stalls, logging, checkpointing, validation, or unobserved transfers |
| Rank straggler | Rank-local input, data imbalance, node variance, or networking |
| Memory creep | Retained tensors, logging references, or cached activations |

## Compare Runs

After fixing a bottleneck, compare two summaries to see whether training
improved and what changed:

```bash
traceml compare before/final_summary.json after/final_summary.json
```

For example, reducing the DataLoader bottleneck shown above changes the
diagnosis and cuts step time:

```text
+--------------------------------------------------------------------------------------+
|  TraceML Compare                                                                     |
+--------------------------------------------------------------------------------------+
|                                                                                      |
|  A: before_dataloader_fix                                                            |
|  B: after_dataloader_fix                                                             |
|  Delta: B - A                                                                        |
|  Primary diagnosis: INPUT-BOUND -> COMPUTE-BOUND (changed)                           |
|                                                                                      |
|  Verdict: IMPROVEMENT                                                                |
|  Why: GPU Step Time decreased by 59.9%.                                              |
+--------------------------------------------------------------------------------------+
```

<details>
<summary><strong>See the full comparison</strong></summary>

```text
+--------------------------------------------------------------------------------------+
|  TraceML Compare                                                                     |
+--------------------------------------------------------------------------------------+
|                                                                                      |
|  A: before_dataloader_fix                                                            |
|  B: after_dataloader_fix                                                             |
|  Delta: B - A                                                                        |
|  Primary diagnosis: INPUT-BOUND -> COMPUTE-BOUND (changed)                           |
|                                                                                      |
|  Verdict: IMPROVEMENT                                                                |
|  Why: GPU Step Time decreased by 59.9%.                                              |
|                                                                                      |
|  Step Time (GPU comparison clock)                                                    |
|  Metric                       A                 B                 Delta              |
|  Step time diagnosis          INPUT-BOUND       COMPUTE-BOUND     changed            |
|  GPU Step Time                200.4 ms          80.4 ms           -120.0 ms (-59.9%) |
|  Input                        128.0 ms          8.0 ms            -120.0 ms (-93.8%) |
|  H2D                          0.4 ms            0.4 ms            +0.0 ms (+0.0%)    |
|  Compute                      68.0 ms           68.0 ms           +0.0 ms (+0.0%)    |
|  Residual                     3.6 ms            3.6 ms            +0.0 ms (+0.0%)    |
|                                                                                      |
|  Step Memory                                                                         |
|  Metric                       A                 B                 Delta              |
|  Step memory diagnosis        BALANCED          BALANCED          same               |
|  Peak reserved                3.1 GB            3.1 GB            0 B (+0.0%)        |
|  Memory skew                  0.0%              0.0%              +0.0 pp            |
|                                                                                      |
|  Process                                                                             |
|  Metric                       A                 B                 Delta              |
|  Process diagnosis            NORMAL            NORMAL            same               |
|  Process CPU avg              95.0%             110.0%            +15.0 pp           |
|  Process RSS avg              1.4 GB            1.6 GB            +0.2 GB (+14.3%)   |
|                                                                                      |
|  System                                                                              |
|  Metric                       A                 B                 Delta              |
|  System diagnosis             LOW GPU UTIL      NORMAL            changed            |
|  System CPU avg               18.4%             32.0%             +13.6 pp           |
|  System RAM avg               12.0 GB           13.5 GB           +1.5 GB (+12.5%)   |
|  GPU util avg                 24.0%             88.0%             +64.0 pp           |
|  GPU memory avg               18.0%             18.0%             +0.0 pp            |
+--------------------------------------------------------------------------------------+
```

</details>

See [Compare Runs](https://traceopt-ai.github.io/traceml/user_guide/compare/)
for the complete workflow and artifact format.

## Save the Result

Send the compact diagnosis to an existing W&B run:

```python
import traceml_ai as traceml
import wandb

...

summary = traceml.summary(print_text=True)
if summary is not None:
    wandb.log(summary)
```

The same result can be stored in MLflow. See
[W&B and MLflow](https://traceopt-ai.github.io/traceml/user_guide/integrations/wandb-mlflow/)
for complete examples.

<details>
<summary><strong>Want live diagnostics during training?</strong></summary>

Use the live terminal view locally or over SSH:

```bash
traceml run train.py --mode=cli
```

Use the browser dashboard on a single node:

```bash
traceml run train.py --mode=dashboard
```

For remote browser access and SSH tunneling, see the
[full quickstart](https://traceopt-ai.github.io/traceml/user_guide/quickstart/).

</details>

## Distributed Training and Integrations

- **Distributed:** [DDP, FSDP, and multi-node](https://traceopt-ai.github.io/traceml/user_guide/distributed-training/)
  or [Slurm](https://traceopt-ai.github.io/traceml/user_guide/slurm/)
- **Frameworks:** [Hugging Face](https://traceopt-ai.github.io/traceml/user_guide/integrations/huggingface/),
  [PyTorch Lightning](https://traceopt-ai.github.io/traceml/user_guide/integrations/lightning/),
  [Ray Train](https://traceopt-ai.github.io/traceml/user_guide/integrations/ray/),
  and [DeepSpeed](https://traceopt-ai.github.io/traceml/user_guide/integrations/deepspeed/)
- **Trackers:** [W&B and MLflow](https://traceopt-ai.github.io/traceml/user_guide/integrations/wandb-mlflow/)

Summary mode is the documented path for single-node and multi-node runs. Live
terminal and dashboard modes are explicit single-node options. See the
[FAQ](https://traceopt-ai.github.io/traceml/user_guide/faq/) for current
support and limitations.

Distributed GPU analysis currently assumes homogeneous GPU hardware across
ranks. Heterogeneous GPU configurations may produce inaccurate cross-rank
analysis and diagnoses.

## Learn More

- [Complete quickstart](https://traceopt-ai.github.io/traceml/user_guide/quickstart/)
- [Measured case studies](https://github.com/traceopt-ai/traceml/blob/main/examples/case_studies/README.md)
- [Examples](https://github.com/traceopt-ai/traceml/blob/main/examples/README.md)
- [Troubleshoot slow training](https://traceopt-ai.github.io/traceml/guides/slow-pytorch-training/)
- [Public API](https://traceopt-ai.github.io/traceml/user_guide/public-api/)
- [FAQ](https://traceopt-ai.github.io/traceml/user_guide/faq/)

## Community

If TraceML helps you find a bottleneck, consider
[starring the repository](https://github.com/traceopt-ai/traceml).
Contributions and real-world slowdown reports are welcome:

- [Contributing guide](https://github.com/traceopt-ai/traceml/blob/main/CONTRIBUTING.md)
- [Open an issue](https://github.com/traceopt-ai/traceml/issues)
- [Security policy](https://github.com/traceopt-ai/traceml/blob/main/SECURITY.md)
- [Discord](https://discord.gg/rY3EQguZAN)

## License

Apache 2.0. See [LICENSE](https://github.com/traceopt-ai/traceml/blob/main/LICENSE).
