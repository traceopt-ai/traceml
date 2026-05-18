<div align="center">

# TraceML

**Runtime bottleneck detection for PyTorch training jobs.**

[![PyPI version](https://img.shields.io/pypi/v/traceml-ai.svg)](https://pypi.org/project/traceml-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/traceopt-ai/traceml?style=social)](https://github.com/traceopt-ai/traceml)

[**Quickstart**](docs/user_guide/quickstart.md) • [**Compare Runs**](docs/user_guide/compare.md) • [**How to Read Output**](docs/user_guide/reading-output.md) • [**W&B / MLflow**](docs/user_guide/integrations/wandb-mlflow.md) • [**FAQ**](docs/user_guide/faq.md) • [**Issues**](https://github.com/traceopt-ai/traceml/issues)

</div>

TraceML records lightweight signals during a PyTorch training run and produces a structured end-of-run summary. It answers the questions that usually come before deep profiling:

- Is the run input-bound, compute-bound, wait-heavy, or memory-constrained?
- Where is time going across dataloader, forward, backward, and optimizer?
- Are some distributed ranks consistently slower than others?
- Did memory usage drift upward during the run?
- Did a recent change cause a regression?

---

## Quickstart

Install:

```bash
pip install traceml-ai
```

Initialize TraceML and wrap your training step:

```python
import traceml

traceml.init()

for batch in dataloader:
    with traceml.trace_step(model):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["x"])
        loss = criterion(outputs, batch["y"])
        loss.backward()
        optimizer.step()
```

Run your script with TraceML:

```bash
traceml run train.py
```

At the end of the run, TraceML prints a compact summary and writes two artifacts:

```text
logs/<session_id>/final_summary.json
logs/<session_id>/final_summary.txt
```

```
+----------------------------------------------------------------------------+
|  TraceML Run Summary | duration 94.2s                                      |
+----------------------------------------------------------------------------+
|                                                                            |
|  System                                                                    |
|  - Diagnosis: NORMAL                                                       |
|  - Scope: nodes 1/1 | samples 847                                          |
|  - Stats: CPU 31% | RAM 28% | GPU util 96% | GPU mem 71% | GPU temp 68.4C  |
|  - Why: CPU and RAM showed no system pressure.                             |
|                                                                            |
|  Process                                                                   |
|  - Diagnosis: NORMAL                                                       |
|  - Stats: global ranks 1 | CPU avg 29% | RSS peak 8.2 / 31.3 GB | GPU mem  |
|  11.4 / 16.0 GB (71%)                                                      |
|  - Why: Process CPU, RSS, and GPU memory showed no pressure.               |
|                                                                            |
|  Step Time                                                                 |
|  - Diagnosis: INPUT-BOUND                                                  |
|  - Scope: last 128 aligned steps on global rank r0                         |
|  - Stats: total 471.2ms | model 101.1ms | compute 94.8ms | wait 6.3ms |    |
|  input 370.1ms                                                             |
|  - Why: Input loading took a large share (370.1ms/471.2ms).                |
|                                                                            |
|  Step Memory                                                               |
|  - Diagnosis: BALANCED                                                     |
|  - Scope: last 128 aligned steps                                           |
|  - Stats: peak reserved peak 11.4 GB                                       |
|  - Why: Memory usage is stable.                                            |
+----------------------------------------------------------------------------+
```

The `final_summary.json` is machine-readable and designed for logging to W&B or MLflow, storing as a run artifact, or comparing against another run.

---

## Compare Runs

TraceML is most useful when you compare a slow or suspicious run against a
baseline or fixed run.

```bash
traceml compare input_slow/final_summary.json input_fixed/final_summary.json
```

Changed your model, dataloader, or batch size? The compare output shows which
phase moved and by how much.

```
+--------------------------------------------------------------------------------------+
|  TraceML Compare                                                                     |
+--------------------------------------------------------------------------------------+
|                                                                                      |
|  A: input_slow                                                                       |
|  B: input_fixed                                                                      |
|  Delta: B - A                                                                        |
|                                                                                      |
|  Verdict: IMPROVEMENT                                                                |
|  Why: Step time decreased by 71.9%.                                                  |
|                                                                                      |
|  Step Time                                                                           |
|  Metric                         A                B                Delta              |
|  Step time diagnosis            INPUT-BOUND      BALANCED         changed            |
|  Total step                     471.2 ms         132.4 ms         -338.8 ms (-71.9%) |
|  Model step                     101.1 ms         101.1 ms         +0.0 ms (+0.0%)    |
|  Compute                        94.8 ms          94.8 ms          +0.0 ms (+0.0%)    |
|  Wait                           6.3 ms           6.3 ms           +0.0 ms (+0.0%)    |
|  Wait share                     6.2%             6.2%             +0.0 pp            |
|  Input                          370.1 ms         31.3 ms          -338.8 ms (-91.5%) |
|                                                                                      |
|  Step Memory                                                                         |
|  Metric                         A                B                Delta              |
|  Step memory diagnosis          BALANCED         BALANCED         same               |
|  Peak reserved                  11.4 GB          11.6 GB          +205 MB (+1.8%)    |
|  Memory skew                    11.1%            11.1%            -0.0 pp            |
|                                                                                      |
|  Process                                                                             |
|  Metric                         A                B                Delta              |
|  Process diagnosis              NORMAL           NORMAL           same               |
|  Process CPU avg                29.0%            31.0%            +2.0 pp            |
|  Process RSS avg                0.8 GB           0.8 GB           +0.0 GB (+0.0%)    |
|                                                                                      |
|  System                                                                              |
|  Metric                         A                B                Delta              |
|  System diagnosis               NORMAL           NORMAL           same               |
|  System CPU avg                 31.0%            30.0%            -1.0 pp            |
|  System RAM avg                 8.0 GB           8.1 GB           +0.1 GB (+1.2%)    |
|  GPU util avg                   96.0%            97.0%            +1.0 pp            |
|  GPU memory avg                 71.0%            72.0%            +1.0 pp            |
+--------------------------------------------------------------------------------------+
```

TraceML writes both a structured compare JSON and a compact text report.

See [Compare Runs](docs/user_guide/compare.md).

---

## Modes

All modes write `final_summary.json` and `final_summary.txt` at the end of the run. The mode controls only what you see during training.

| Mode | During training | Topology |
|------|----------------|----------|
| `--mode=summary` | Silent | single-node and multi-node multi-GPU |
| `--mode=cli` | Live terminal display | single-node, including multi-GPU |
| `--mode=dashboard` | Live browser display | single-node, including multi-GPU |

Summary mode is the default and works across all topologies. Use `--mode=cli` or `--mode=dashboard` when you want live feedback on a single-node job.

Multi-node live views are on the roadmap.

For very long jobs, tune the final-summary window with
`--summary-window-rows N`. TraceML analyzes the latest `N` rows per node or
rank and retains a small alignment buffer internally.

---

## Common Workflows

### Diagnose one run

```bash
traceml run train.py
```

### Multi-node distributed run

On node 0:

```bash
traceml run train.py \
  --nnodes=2 \
  --node-rank=0 \
  --nproc-per-node=4 \
  --master-addr=<node0-ip> \
  --session-id=my-run
```

On node 1:

```bash
traceml run train.py \
  --nnodes=2 \
  --node-rank=1 \
  --nproc-per-node=4 \
  --master-addr=<node0-ip> \
  --session-id=my-run
```

Use the same `--session-id`, `--nnodes`, `--nproc-per-node`, and `--master-addr` on every node. Node 0 starts the TraceML aggregator; the other nodes connect to it. If the TraceML aggregator should use a different reachable address than `--master-addr`, pass `--aggregator-host=<host>`.

### Zero-code first look

```bash
traceml watch train.py
```

System and process telemetry only. No step instrumentation needed.

### Compare two runs

```bash
traceml compare before/final_summary.json after/final_summary.json
```

---

## What TraceML measures

| Signal | What it means |
|--------|--------------|
| Input-bound | Dataloader is the bottleneck — GPU is waiting on data |
| Compute-bound | GPU is saturated — expected in a healthy run |
| Wait-heavy | Unattributed step time outside the traced phases |
| Rank imbalance | One rank consistently slower — straggler or uneven data |
| Memory creep | Peak allocation growing step-over-step |
| High pressure | Memory near capacity — risk of OOM |

`wait` is residual step time, not direct NCCL or all-reduce timing. In DDP, communication may overlap with backward. Use PyTorch Profiler or Nsight when you need explicit collective or kernel-level timing.

---

## When to use TraceML

Use TraceML when you want a lightweight performance fingerprint for a PyTorch training run:

- keep a small `final_summary.json` you can share, store, diff, or log
- see where step time went across dataloader, forward, backward, optimizer, and wait time
- compare a new run against a previous baseline
- check whether ranks, nodes, process memory, or system resources look imbalanced
- collect enough evidence before opening PyTorch Profiler or Nsight

**When not to use TraceML:** If you already need operator, kernel, or collective-level timing, go straight to `torch.profiler` or Nsight. TraceML is the cheap first pass that tells you where to look.

---

## How it fits with your stack

TraceML sits between experiment tracking and heavyweight profiling.

```text
Run PyTorch training with TraceML
        ↓
Save final_summary.json as a lightweight performance fingerprint
        ↓
Review final_summary.txt for the likely bottleneck
        ↓
Compare against a previous summary when behavior changes
        ↓
Open torch.profiler or Nsight only if you need operator/kernel detail
```

Use W&B, MLflow, or TensorBoard for experiment tracking, metrics, and dashboards. Use TraceML for bottleneck diagnosis, distributed run summaries, and run-to-run performance comparison.

See [Use TraceML with W&B / MLflow](docs/user_guide/integrations/wandb-mlflow.md).

---

## Current support

**Works today:**

- Single GPU and single-node multi-GPU training
- Multi-node DDP / FSDP summary reports
- Step Time, Step Memory, System, and Process diagnostics
- Run-to-run comparison from `final_summary.json`
- Custom PyTorch loops, Hugging Face, and PyTorch Lightning

**Next:**

- Ray Train integration
- Slurm launch examples
- Multi-node live CLI / dashboard
- Explicit collective / NCCL timing

---

## Overhead

TraceML adds fixed per-step instrumentation overhead. Relative overhead is highest when training steps are very short. In larger jobs the fixed cost is amortized over longer step time.

In our early DDP benchmarks, TraceML did not produce a measurable slowdown beyond normal run-to-run variation.

---

## Learn more

- [Quickstart](docs/user_guide/quickstart.md)
- [Compare Runs](docs/user_guide/compare.md)
- [How to Read TraceML Output](docs/user_guide/reading-output.md)
- [Examples](examples/README.md)
- [FAQ](docs/user_guide/faq.md)
- [Use TraceML with W&B / MLflow](docs/user_guide/integrations/wandb-mlflow.md)
- [Hugging Face integration](docs/user_guide/integrations/huggingface.md)
- [PyTorch Lightning integration](docs/user_guide/integrations/lightning.md)

---

## Feedback

If TraceML helped you catch a slowdown, please open an issue and include:

- hardware / CUDA / PyTorch versions
- single GPU or multi-GPU setup
- training framework
- the end-of-run summary
- a minimal repro if possible

GitHub issues: https://github.com/traceopt-ai/traceml/issues

Email: support@traceopt.ai

---

## Contributing

If TraceML helped you catch a slowdown, a GitHub star helps others find it.

Contributions are welcome, especially:

- real slowdown examples and repros
- distributed training edge cases
- docs improvements
- framework integrations

---

## License

Apache 2.0. See [LICENSE](LICENSE).

TraceOpt is a trademark of OptAI UG (haftungsbeschränkt).

> **Upcoming rename:** `traceml-ai` will be renamed to `traceopt-ai` in a future release.
> Python imports will change from `traceml` to `traceopt`. The active package today remains `traceml-ai`.
