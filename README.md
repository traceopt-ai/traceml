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

TraceML writes two end-of-run artifacts:

```text
logs/<session_id>/final_summary.json
logs/<session_id>/final_summary.txt
```

## Example Output

### End-of-run summary

At the end of training, TraceML prints the same compact text report written to
`final_summary.txt`.

Example from a 4-rank DDP run configured as 2 nodes x 2 GPUs:

```
+----------------------------------------------------------------------------+
|  TraceML Run Summary | duration 122.5s                                     |
+----------------------------------------------------------------------------+
|                                                                            |
|  System                                                                    |
|  - Diagnosis: NORMAL                                                       |
|  - Scope: nodes 2/2 | samples 124                                          |
|  - Stats: CPU med/worst 3%/3% n0 | RAM med/worst 4%/4% n1 | GPU util       |
|  med/worst 74%/74% n0 | GPU temp med/worst 47.9C/47.9C n1                  |
|  - Why: CPU, RAM, and GPU showed no system pressure.                       |
|                                                                            |
|  Process                                                                   |
|  - Diagnosis: GPU MEMORY RESERVED OVERHANG                                 |
|  - Stats: global ranks 4 | CPU avg 75% | RSS peak 1.3 / 540.7 GB | GPU     |
|  reserved peak 1%                                                          |
|  - Why: Reserved GPU memory was 1.70x active use.                          |
|                                                                            |
|  Step Time                                                                 |
|  - Diagnosis: INPUT STRAGGLER                                              |
|  - Scope: compared over last 460 aligned steps across 4 global ranks       |
|  - Stats: median/worst | total 303.7/303.7ms | input 3.8/254.5ms |         |
|  compute 259.5/259.5ms | wait 40.5/40.5ms                                  |
|  - Ranks: median/worst | total r3/r2 | input r2/r0 | compute r3/r1 | wait  |
|  r2/r1                                                                      |
|  - Why: r0 input was slower than median global rank (254.5/3.8ms).         |
|                                                                            |
|  Step Memory                                                               |
|  - Diagnosis: BALANCED                                                     |
|  - Scope: last 460 aligned steps                                           |
|  - Stats: peak reserved worst 192 MB on r0 | skew 0.0%                     |
|  - Why: No clear pressure, imbalance, or creep signal.                     |
+----------------------------------------------------------------------------+
```

The `final_summary.json` is machine-readable and designed for logging to W&B or MLflow, storing as a run artifact, or comparing against another run.

---

### Compare two runs

Compare a slow or suspicious run against a baseline or fixed run:

```bash
traceml compare input_slow/final_summary.json input_fixed/final_summary.json
```

The compact text report shows the verdict first, then the changed metrics:

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
|  Why: Step time decreased by 95.6%.                                                  |
|                                                                                      |
|  Step Time                                                                           |
|  Metric                         A                B                Delta              |
|  Step time diagnosis            INPUT STRAGGLER  BALANCED         changed            |
|  Total step                     294.0 ms         13.0 ms          -280.9 ms (-95.6%) |
|  Input                          66.4 ms          2.7 ms           -63.7 ms (-95.9%)  |
|  Compute                        197.2 ms         8.6 ms           -188.6 ms (-95.6%) |
|  Wait                           30.4 ms          1.7 ms           -28.6 ms (-94.3%)  |
|  Forward                        45.0 ms          2.1 ms           -42.9 ms (-95.3%)  |
|  Backward                       130.0 ms         5.4 ms           -124.6 ms (-95.8%) |
|  Optimizer                      22.2 ms          1.1 ms           -21.1 ms (-95.0%)  |
+--------------------------------------------------------------------------------------+
```

The full compare report also includes Step Memory, Process, and System sections
when those signals are available. TraceML writes both a structured compare JSON
and a compact text report.

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

Use the same `--session-id`, `--nnodes`, `--nproc-per-node`, and
`--master-addr` on every node. Node 0 starts the TraceML aggregator. Other
nodes connect to `<node0-ip>:29765` by default. If workers need a different
reachable address or port for TraceML telemetry, add
`--aggregator-host=<host>` or `--aggregator-port=<port>` on every node. For
multi-node runs, node 0 binds the aggregator to `0.0.0.0` by default; override
that only when needed with `--aggregator-bind-host=<bind-host>`.

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

- Single GPU training
- Single-node multi-GPU DDP / FSDP training
- Multi-node DDP summary reports
- Step Time, Step Memory, System, and Process diagnostics
- Run-to-run comparison from `final_summary.json`
- Custom PyTorch loops, Hugging Face, and PyTorch Lightning

**Next:**

- Ray Train integration
- Slurm launch examples
- Broader multi-node FSDP validation
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

GitHub issues: [open an issue](https://github.com/traceopt-ai/traceml/issues)

Email: [support@traceopt.ai](mailto:support@traceopt.ai)

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
