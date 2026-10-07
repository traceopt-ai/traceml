<div align="center">

# TraceML

**Diagnose slow PyTorch training with zero-code instrumentation. Catch regressions in CI.**

**Works automatically with:**
[Hugging Face Trainer](https://traceopt-ai.github.io/traceml/user_guide/integrations/huggingface/) ·
[PyTorch Lightning](https://traceopt-ai.github.io/traceml/user_guide/integrations/lightning/) ·
[RF-DETR](https://traceopt-ai.github.io/traceml/user_guide/integrations/rfdetr/)

[![PyPI version](https://img.shields.io/pypi/v/traceml-ai.svg)](https://pypi.org/project/traceml-ai/)
[![CI](https://github.com/traceopt-ai/traceml/actions/workflows/ci.yml/badge.svg)](https://github.com/traceopt-ai/traceml/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/traceopt-ai/traceml/blob/main/LICENSE)
[![GitHub stars](https://badgen.net/github/stars/traceopt-ai/traceml?icon=github)](https://github.com/traceopt-ai/traceml)

[**Quickstart**](#quickstart) ·
[**What you get**](#what-you-get) ·
[**Compare runs**](#compare-runs) ·
[**Regression checks**](#performance-regression-checks) ·
[**Integrations**](#training-integrations) ·
[**Documentation**](https://traceopt-ai.github.io/traceml/)

⭐ If TraceML helps you find a bottleneck, please
[star the repository](https://github.com/traceopt-ai/traceml).

</div>

TraceML shows where each training step goes—input loading, data transfer, forward, backward, and optimizer work—then identifies the bottleneck and saves evidence for local comparison or CI.

## Quickstart

### 1. Install

If your training framework is already installed:

```bash
pip install traceml-ai
```

### 2. Run your existing script

```diff
- python train.py
+ traceml run train.py
```

For standard Hugging Face Trainer, PyTorch Lightning, and RF-DETR training,
TraceML instruments the run without changes to the training script. It prints a
diagnosis when training finishes and writes `final_summary.json` and
`final_summary.txt` under `logs/<run-name>/`.

No training script ready? Try the
[interactive demo](https://huggingface.co/spaces/abhinavsriva/traceml-training-diagnosis)
or [Colab example](https://colab.research.google.com/github/traceopt-ai/traceml/blob/main/notebooks/data_loading_bottleneck.ipynb).

## What you get

The terminal report identifies the bottleneck, shows the timing and resource
evidence behind it, and suggests the next investigation.

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
|  GPU util               24%                                   ||  CUDA allocated       2.9 GB                                                            |
|  GPU memory/device      3.3 GB (21%)                          ||  CUDA reserved        3.2 GB (20%)                                                      |
|  GPU temperature        42C                                   ||                                                                                         |
|  GPU power              58W                                   ||                                                                                         |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Full evidence: logs/bert_finetune/final_summary.json  (--html-report)                                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
```

<details>
<summary><strong>Training with multiple ranks? See a rank-straggler diagnosis</strong></summary>

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
|  GPU util               9%                9%, N1              ||  CUDA allocated       2.9 GB            4.6 GB, R3/N1                                   |
|  GPU memory/device      5.0 GB (31%)      7.0 GB (44%), N1    ||  CUDA reserved        3.2 GB (20%)      6.8 GB (43%), R3/N1                             |
|  GPU temperature        58C               70C, N1             ||                                                                                         |
|  GPU power              220W              280W, N1            ||                                                                                         |
|                                                                                                                                                          |
|                                                                                                                                                          |
|  Full evidence: logs/ddp_pretrain/final_summary.json  (--html-report)                                                                                    |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+
```

</details>

TraceML is designed for lightweight, always-on training diagnosis. It shows
enough evidence to choose the next investigation; use a kernel profiler when
the result points inside GPU compute.

## What TraceML diagnoses

| Diagnosis | Where to investigate |
|---|---|
| Input-bound | DataLoader workers, transforms, tokenization, collation, or storage |
| H2D-bound | Pinned memory, non-blocking copies, batch size, or transfer overlap |
| Compute-bound | Model compute, mixed precision, batch size, or deeper profiling |
| Residual-heavy | Logging, checkpointing, validation, CPU stalls, or unobserved work |
| Rank straggler | Rank-local input, data imbalance, node variance, or networking |
| Memory creep | Retained tensors, logging references, or cached activations |

Read [How to Read TraceML Output](https://traceopt-ai.github.io/traceml/user_guide/reading-output/)
for definitions, diagnosis rules, and evidence limits.

## Compare runs

After changing the DataLoader, batch size, model, or infrastructure, compare two
completed runs:

```bash
traceml compare before/final_summary.json after/final_summary.json
```

```text
+--------------------------------------------------------------------------------------+
|  TraceML Compare                                                                     |
+--------------------------------------------------------------------------------------+
|  A: before_dataloader_fix                                                            |
|  B: after_dataloader_fix                                                             |
|  Primary diagnosis: INPUT-BOUND -> COMPUTE-BOUND (changed)                           |
|                                                                                      |
|  Verdict: IMPROVEMENT                                                                |
|  Why: GPU Step Time decreased by 59.9%.                                              |
+--------------------------------------------------------------------------------------+
```

TraceML keeps the comparison evidence in JSON and text so the result is easy to
review, archive, or use in automation. See
[Compare Runs](https://traceopt-ai.github.io/traceml/user_guide/compare/).

## Performance regression checks

Use the same comparison as a local or CI gate for compatible runs:

```bash
traceml compare \
  logs/reference/final_summary.json \
  logs/candidate/final_summary.json \
  --max-step-time-regression-pct 5 \
  --output compare/reference-vs-candidate
```

TraceML writes the evidence before returning the CI exit code. The regression
guard is an experimental pilot; read the
[Regression Guard](https://traceopt-ai.github.io/traceml/user_guide/regression-guard/)
for comparability requirements and supported environments.

## Distributed training

Launch one TraceML process per training process:

```diff
- torchrun --nproc-per-node=4 train.py
+ traceml run train.py --nproc-per-node=4
```

The final summary aligns common steps across ranks and can identify the worker
most likely to be holding up the run. See
[Distributed Training](https://traceopt-ai.github.io/traceml/user_guide/distributed-training/),
[DDP rank stragglers](https://traceopt-ai.github.io/traceml/guides/ddp-slow-training-rank-straggler/),
and [Slurm](https://traceopt-ai.github.io/traceml/user_guide/slurm/).

TraceML is currently focused on single-device training and DDP on one machine.
You can launch multi-node runs, but that path remains experimental. Distributed
GPU comparisons assume homogeneous hardware across ranks. See the
[integration support matrix](https://traceopt-ai.github.io/traceml/user_guide/integrations/#integration-support-matrix)
for framework-specific details, including FSDP.

## Featured case studies

| Investigation | Result |
|---|---|
| [ResNet-18 input pipeline](examples/case_studies/resnet18_input_bound/README.md) | See an input-bound run become compute-bound after changing only DataLoader settings. |
| [RF-DETR Nano training](examples/case_studies/rfdetr_nano_training/README.md) | Examine single-GPU phase timing and four-GPU DDP scaling on real COCO batches. |
| [RF-DETR release regression](examples/case_studies/rfdetr_input_pipeline_regression/README.md) | Attribute a release-to-release slowdown to input waiting and verify the fixed release. |

[Browse all case studies →](https://traceopt-ai.github.io/traceml/case-studies/)

## Training integrations

The zero-code command works with these standard trainer APIs:

| Framework | Training API |
|---|---|
| [Hugging Face Trainer](https://traceopt-ai.github.io/traceml/user_guide/integrations/huggingface/) | `trainer.train()` |
| [PyTorch Lightning](https://traceopt-ai.github.io/traceml/user_guide/integrations/lightning/) | `trainer.fit(...)` |
| [RF-DETR](https://traceopt-ai.github.io/traceml/user_guide/integrations/rfdetr/) | `model.train(...)` |

### Manual and explicit integrations

Custom loops and other training paths use their existing TraceML integration:

| Training path | Setup |
|---|---|
| Plain PyTorch or a custom loop | [`traceml.init()` and `trace_step(...)`](https://traceopt-ai.github.io/traceml/user_guide/quickstart/) |
| Hugging Face Accelerate | [Explicit step instrumentation](https://traceopt-ai.github.io/traceml/user_guide/integrations/accelerate/) |
| Ray Train and Ray Data | [TraceML Trainer/config wrappers](https://traceopt-ai.github.io/traceml/user_guide/integrations/ray/) |
| DeepSpeed | [Explicit step instrumentation](https://traceopt-ai.github.io/traceml/user_guide/integrations/deepspeed/) |
| MONAI | [TraceML handler setup](https://traceopt-ai.github.io/traceml/user_guide/integrations/monai/) |

<details>
<summary><strong>Plain PyTorch example</strong></summary>

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

Run it with `traceml run train.py`.

</details>

Manual APIs remain available for custom loops and advanced timing boundaries.
See the [Public API](https://traceopt-ai.github.io/traceml/user_guide/public-api/)
and [integration support matrix](https://traceopt-ai.github.io/traceml/user_guide/integrations/).

## Reports and experiment trackers

Summary mode is the default. For live diagnostics in the terminal, use:

```bash
traceml run train.py --mode=cli
```

For the browser dashboard, install the optional dashboard dependencies:

```bash
pip install "traceml-ai[dashboard]"
traceml run train.py --mode=dashboard
```

TraceML can also export a self-contained HTML report and send its compact
summary to an existing W&B or MLflow run. See
[W&B and MLflow](https://traceopt-ai.github.io/traceml/user_guide/integrations/wandb-mlflow/)
and the [complete quickstart](https://traceopt-ai.github.io/traceml/user_guide/quickstart/).

## Learn more

- [Documentation](https://traceopt-ai.github.io/traceml/)
- [Examples](https://github.com/traceopt-ai/traceml/blob/main/examples/README.md)
- [Troubleshoot slow training](https://traceopt-ai.github.io/traceml/guides/slow-pytorch-training/)
- [Integration support matrix](https://traceopt-ai.github.io/traceml/user_guide/integrations/)
- [FAQ](https://traceopt-ai.github.io/traceml/user_guide/faq/)

## Community

Questions, contributions, and real-world slowdown reports are welcome:

- [Open an issue](https://github.com/traceopt-ai/traceml/issues)
- [Contributing guide](https://github.com/traceopt-ai/traceml/blob/main/CONTRIBUTING.md)
- [Discord](https://discord.gg/rY3EQguZAN)
- [Security policy](https://github.com/traceopt-ai/traceml/blob/main/SECURITY.md)

## License

Apache 2.0. See [LICENSE](https://github.com/traceopt-ai/traceml/blob/main/LICENSE).
