<div align="center">

# TraceML

**A lightweight runtime health check for PyTorch training runs.**

[![PyPI version](https://img.shields.io/pypi/v/traceml-ai.svg)](https://pypi.org/project/traceml-ai/)
[![CI](https://github.com/traceopt-ai/traceml/actions/workflows/ci.yml/badge.svg)](https://github.com/traceopt-ai/traceml/actions/workflows/ci.yml)
[![CodeQL](https://github.com/traceopt-ai/traceml/actions/workflows/codeql.yml/badge.svg)](https://github.com/traceopt-ai/traceml/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE)
[![GitHub stars](https://badgen.net/github/stars/traceopt-ai/traceml?icon=github)](https://github.com/traceopt-ai/traceml/stargazers)
[![Discord](https://img.shields.io/badge/Discord-Join%20chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/rY3EQguZAN)

[**Quickstart**](docs/user_guide/quickstart.md) •
[**Compare Runs**](docs/user_guide/compare.md) •
[**Read Output**](docs/user_guide/reading-output.md) •
[**Use With Your Stack**](docs/user_guide/integrations.md) •
[**FAQ**](docs/user_guide/faq.md)

</div>

### Is your GPU training, waiting on data, or blocked by one slow rank?

PyTorch training can look normal while step time is lost to a slow input
pipeline, low GPU utilization, memory growth, distributed rank skew, or a
run-to-run regression.

TraceML runs alongside your PyTorch training loop and writes a compact
runtime performance report at the end of each run. With low overhead
(<2% in our current benchmark runs), it helps you see where step time and
memory went before opening heavier tools like `torch.profiler` or Nsight.
It helps answer:

- Are my GPUs waiting on a slow dataloader?
- Is one distributed rank consistently slower than the others?
- Is memory usage silently creeping upward during the run?
- Did a recent code, data, or infrastructure change slow training down?

Here, **health** means runtime performance health: step time, input/compute/wait
balance, memory behavior, distributed rank skew, and run-to-run change.

---

## 3-Minute Quickstart

### 1. Install the package

```bash
pip install traceml-ai
```

Using Hugging Face Trainer, PyTorch Lightning, Ray Train, W&B, or MLflow?
Start with the native integration path in
[Use With Your Stack](docs/user_guide/integrations.md).

### 2. Wrap your training step

Add TraceML around the core training step. You do not need to change your model,
optimizer, loss function, or dataloader.

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

### 3. Run your script

```bash
traceml run train.py
```

For DDP, FSDP, and multi-node runs, see
[Distributed Training](docs/user_guide/distributed-training.md).

---

## What You Get

TraceML writes two end-of-run artifacts:

```text
logs/<run_name>/final_summary.json
logs/<run_name>/final_summary.txt
```

You can re-print a saved summary later without rerunning training:

```bash
traceml view logs/<run_name>/final_summary.json
```

Instead of guessing why training feels slow, you get a compact diagnosis of
where step time and memory went.

Example TraceML output:

```text
+----------------------------------------------------------------------------+
|  Step Time                                                                 |
|  - Diagnosis: INPUT STRAGGLER                                              |
|  - Scope: compared over last 460 aligned steps across 4 global ranks       |
|  - Stats: total 303.7ms | input 254.5ms | compute 259.5ms | wait 40.5ms    |
|  - Why: r0 input was slower than median global rank (254.5/3.8ms).         |
+----------------------------------------------------------------------------+
```

In this example, rank 0 is the slow input rank, which can hold back the aligned
distributed step.

For experiment trackers, call `traceml.summary()` near the end of your script
to get a flat dict of diagnosis statuses and average metrics. Keep
`final_summary.json` when you want the full run artifact or an input for
`traceml compare`.

---

## What TraceML Helps You Triage

TraceML is meant to be the first check when a training run is slower than
expected. It points you to the likely area before you decide whether to open a
heavier profiler.

| Area | What TraceML surfaces | What to inspect next |
|---|---|---|
| Input pipeline | High input time or slow input rank | `num_workers`, `pin_memory`, transforms, tokenization, `collate_fn`, dataset/storage latency |
| GPU utilization / wait | Step time split across input, compute, and wait | input pipeline, CPU/GPU handoff, synchronization, distributed coordination |
| Distributed skew | One DDP/FSDP rank slower than the others | rank-local dataloading, data imbalance, node variance, storage/network differences |
| Memory creep | Memory usage growing during the run | retained tensors, logging references, loss accumulation, cached activations |
| Run regression | Changed metrics versus a known-good run | code changes, data changes, batch size, container, driver, hardware, infrastructure |
| Compute-heavy runs | Most time is spent in compute | open `torch.profiler` or Nsight for operator/kernel-level detail |

---

## Catching Regressions with Compare Mode

Compare a slow run against a known good baseline to identify which metrics
changed:

```bash
traceml compare input_slow/final_summary.json input_fixed/final_summary.json
```

```text
+--------------------------------------------------------------------------------------+
|  TraceML Compare                                                                     |
+--------------------------------------------------------------------------------------+
|  Verdict: IMPROVEMENT                                                                |
|  Why: Step time decreased by 95.6%.                                                  |
|                                                                                      |
|  Metric                         A                B                Delta              |
|  Total step                     294.0 ms         13.0 ms          -280.9 ms (-95.6%) |
|  Input                          66.4 ms          2.7 ms           -63.7 ms (-95.9%)  |
+--------------------------------------------------------------------------------------+
```

See [Compare Runs](docs/user_guide/compare.md) for the full report format.

---

## Display Modes

TraceML controls what you see during training with the `--mode` flag, without
changing the final saved artifacts.

| Mode flag | Experience during training | Supported topology |
|---|---|---|
| `--mode=summary` (default) | Silent execution | Single-node and multi-node multi-GPU |
| `--mode=cli` | Live terminal display | Single-node, including multi-GPU |
| `--mode=dashboard` | Live browser display | Single-node; requires `pip install "traceml-ai[dashboard]"` |

---

## Current Support

**Works today:**

- Single GPU training
- Single-node multi-GPU DDP / FSDP
- Multi-node DDP summary reports
- Multi-node runs on Slurm (sbatch template + guide)
- Run-to-run comparison from `final_summary.json`
- Custom PyTorch loops, Hugging Face, PyTorch Lightning, and Ray Train

**On the roadmap:**

- Multi-node live CLI / browser dashboard
- Explicit collective / NCCL timing

---

## Where TraceML Fits in the Stack

TraceML does not replace `torch.profiler`. It is the low-overhead first pass that
helps you decide where to aim heavier profiling tools.

| Tool | Best used for | Output | Cost / overhead |
|---|---|---|---|
| TraceML | Classifying high-level bottlenecks: input, compute, wait, memory, rank skew | JSON fingerprint, text summary, live views | <2% in current benchmark runs; small code wrapper |
| `torch.profiler` | Inspecting expensive ops, kernels, and CUDA activity | Profiler trace | Higher overhead; requires profiler context |
| Nsight Systems | Debugging low-level CUDA and kernel behavior | GPU timeline | Separate profiler run |
| W&B / MLflow | Tracking training metrics and experiment history | Metrics dashboard / run history | Logging integration |
| `nvidia-smi` | Checking machine-level GPU health and utilization | Terminal metrics | No code changes |

---

## Overhead

In our benchmark runs, TraceML adds:

- <2% overhead on single GPU at default settings
- <1% overhead on single-node multi-GPU at default settings

---

## Troubleshooting Guides

These guides cover the common bottlenecks TraceML is designed to identify:

- [Find why PyTorch training is slow](docs/guides/slow-pytorch-training.md)
- [Find DataLoader Bottlenecks](docs/guides/pytorch-dataloader-bottleneck.md)
- [Debug Low GPU Utilization](docs/guides/low-gpu-utilization-pytorch.md)
- [Debug DDP Rank Stragglers](docs/guides/ddp-slow-training-rank-straggler.md)
- [Find PyTorch Memory Creep](docs/guides/pytorch-memory-creep.md)
- [Distributed Training](docs/user_guide/distributed-training.md)
- [Running on Slurm](docs/user_guide/slurm.md)
- [Use With Your Stack](docs/user_guide/integrations.md)
- [Compare Runs](docs/user_guide/compare.md)
- [How to Read Output](docs/user_guide/reading-output.md)
- [FAQ](docs/user_guide/faq.md)

---

## Feedback

For bugs, unexpected results, or feature requests, open a GitHub issue and use
the matching issue template. The templates ask for the details we need to
reproduce training-environment problems, including hardware, topology, launch
command, TraceML version, PyTorch/CUDA versions, and redacted summary output.

GitHub issues: [open an issue](https://github.com/traceopt-ai/traceml/issues)

If TraceML helped you find a real bottleneck, use the "I found a bottleneck"
issue template. These reports help other training teams recognize similar
problems.

Security reports: see [SECURITY.md](SECURITY.md)

Email: [support@traceopt.ai](mailto:support@traceopt.ai)

---

## Contributing

Contributions are welcome, especially:

- real slowdown examples and repros
- distributed training edge cases
- docs improvements
- framework integrations

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

TraceOpt is a trademark of OptAI UG (haftungsbeschränkt).
