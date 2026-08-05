<div align="center">

# TraceML

**Find out why your PyTorch training is slow—before it wastes GPU hours.**

[![PyPI version](https://img.shields.io/pypi/v/traceml-ai.svg)](https://pypi.org/project/traceml-ai/)
[![CI](https://github.com/traceopt-ai/traceml/actions/workflows/ci.yml/badge.svg)](https://github.com/traceopt-ai/traceml/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE)
[![GitHub stars](https://badgen.net/github/stars/traceopt-ai/traceml?icon=github)](https://github.com/traceopt-ai/traceml/stargazers)

[**Quickstart**](#quickstart) •
[**Try in Colab**](https://colab.research.google.com/github/traceopt-ai/traceml/blob/main/notebooks/data_loading_bottleneck.ipynb) •
[**Integrations**](docs/user_guide/integrations.md) •
[**Documentation**](https://traceopt-ai.github.io/traceml/) •
[**Discord**](https://discord.gg/rY3EQguZAN)

</div>

**TraceML is an open-source tool that explains why PyTorch training is slow.**
It analyzes the full training run and gives you:

- **A diagnosis:** data loading, GPU compute, waiting, memory growth, or a slow
  distributed worker.
- **The evidence:** timing, CPU/GPU usage, memory, and per-worker measurements.
- **The next step:** what part of your training setup to investigate first.

Here is an example where TraceML finds that a slow DataLoader is leaving the
GPU waiting:

```text
+----------------------------------------------------------------------------+
|  TraceML Run Summary | duration 52.4s                                      |
+----------------------------------------------------------------------------+
|                                                                            |
|  TraceML Verdict: INPUT-BOUND / CRITICAL                                   |
|  Why: Input Wait is 64.0% of the typical GPU Step Time.                    |
|  Next: Increase workers, prefetch, or storage throughput.                  |
|                                                                            |
|  Section Status                                                            |
|  Section       Status                  Severity                            |
|  ------------------------------------------------                          |
|  Step Time     INPUT-BOUND             CRITICAL                            |
|  System        LOW GPU UTIL            INFO                                |
|  Process       NORMAL                  INFO                                |
|  Step Memory   BALANCED                INFO                                |
|                                                                            |
|  System Evidence                                                           |
|  Metric            Average                                                 |
|  ----------------------------------                                        |
|  CPU Util          18.4%                                                   |
|  GPU Util          24.0%                                                   |
|  GPU Memory        3.33GB                                                  |
|  GPU Temp          42C                                                     |
|                                                                            |
|  Step Time Evidence                                                        |
|  Phase             Average           Share                                 |
|  ------------------------------------------------                          |
|  Step Time         200.4ms           100.0%                                |
|  Input Wait        128.0ms           64.0%                                 |
|  Traced Step Time  72.0ms            supplemental                          |
|  Compute           68.0ms            34.0%                                 |
|  Residual          3.6ms             1.8%                                  |
|  H2D               0.4ms             0.2%                                  |
+----------------------------------------------------------------------------+
```

<details>
<summary><strong>Running distributed training? See a rank-straggler diagnosis</strong></summary>

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
|  Step Time       303.7ms       304.1ms       0.1%        rank=r0 node=n0   |
|  Input Wait      3.8ms         254.5ms       6597.4%     rank=r0 node=n0   |
|  Compute         259.5ms       261.0ms       0.6%        rank=r2 node=n1   |
+----------------------------------------------------------------------------+
```

</details>

## Quickstart

### 1. Install

```bash
pip install traceml-ai
```

Or, in a project managed by [uv](https://docs.astral.sh/uv/):

```bash
uv add traceml-ai
```

### 2. Instrument the training step

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

Summary mode is the default. TraceML prints the final diagnosis and writes:

```text
logs/<run_name>/final_summary.json
logs/<run_name>/final_summary.txt
```

See the
[full quickstart](docs/user_guide/quickstart.md) for source-only examples,
Docker (both require a repository checkout), Colab, direct `python`/`torchrun`
launches, HTML reports, and advanced options.

## What TraceML Diagnoses

| Diagnosis | Where to investigate |
|---|---|
| Input-bound | DataLoader workers, transforms, tokenization, collation, or storage |
| Compute-bound | Model compute, mixed precision, batch size, or deeper profiling |
| Residual-heavy | Work outside traced phases, CPU stalls, logging, checkpointing, validation, or unobserved transfers |
| Rank straggler | Rank-local input, data imbalance, node variance, or networking |
| Memory creep | Retained tensors, logging references, or cached activations |
| Run regression | Code, data, environment, hardware, or infrastructure changes |

Read [How to Read TraceML Output](docs/user_guide/reading-output.md) for the
diagnosis rules, evidence fields, and recommended next actions.

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

See [Compare Runs](docs/user_guide/compare.md) for the complete workflow and
artifact format.

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
[W&B and MLflow](docs/user_guide/integrations/wandb-mlflow.md) for complete
examples.

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
[full quickstart](docs/user_guide/quickstart.md).

</details>

## Distributed Training and Integrations

- **Distributed:** [DDP, FSDP, and multi-node](docs/user_guide/distributed-training.md)
  or [Slurm](docs/user_guide/slurm.md)
- **Frameworks:** [Hugging Face](docs/user_guide/integrations/huggingface.md),
  [PyTorch Lightning](docs/user_guide/integrations/lightning.md),
  [Ray Train](docs/user_guide/integrations/ray.md), and
  [DeepSpeed](docs/user_guide/integrations/deepspeed.md)
- **Trackers:** [W&B and MLflow](docs/user_guide/integrations/wandb-mlflow.md)

Summary mode supports single-node and multi-node runs. Live terminal and
dashboard modes are explicit single-node options. See the
[FAQ](docs/user_guide/faq.md) for current support and limitations.

## Learn More

- [Complete quickstart](docs/user_guide/quickstart.md)
- [Examples](examples/README.md)
- [Troubleshoot slow training](docs/guides/slow-pytorch-training.md)
- [Public API](docs/user_guide/public-api.md)
- [FAQ](docs/user_guide/faq.md)

## Community

If TraceML helps you find a bottleneck, consider
[starring the repository](https://github.com/traceopt-ai/traceml).
Contributions and real-world slowdown reports are welcome:

- [Contributing guide](CONTRIBUTING.md)
- [Open an issue](https://github.com/traceopt-ai/traceml/issues)
- [Security policy](SECURITY.md)
- [Discord](https://discord.gg/rY3EQguZAN)

## License

Apache 2.0. See [LICENSE](LICENSE).
