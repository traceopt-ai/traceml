<div align="center">

# TraceML

**Runtime bottleneck detection for PyTorch training jobs.**

[![PyPI version](https://img.shields.io/pypi/v/traceml-ai.svg)](https://pypi.org/project/traceml-ai/)
[![CI](https://github.com/traceopt-ai/traceml/actions/workflows/ci.yml/badge.svg)](https://github.com/traceopt-ai/traceml/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/traceopt-ai/traceml?style=social)](https://github.com/traceopt-ai/traceml)

[**Quickstart**](docs/user_guide/quickstart.md) •
[**Compare Runs**](docs/user_guide/compare.md) •
[**How to Read Output**](docs/user_guide/reading-output.md) •
[**Ray Train**](docs/user_guide/integrations/ray.md) •
[**W&B / MLflow**](docs/user_guide/integrations/wandb-mlflow.md) •
[**FAQ**](docs/user_guide/faq.md) •
[**Security**](SECURITY.md) •
[**Issues**](https://github.com/traceopt-ai/traceml/issues) •
[**Discussions**](https://github.com/traceopt-ai/traceml/discussions)

</div>

TraceML gives every PyTorch training run a structured performance fingerprint: where time went, whether ranks skewed, and whether memory drifted. It answers the questions that usually come before operator-level profiling:

- Is the run input-bound, compute-bound, wait-heavy, or memory-constrained?
- How much time is spent in dataloader, forward, backward, and optimizer?
- Are some distributed ranks consistently slower than others?
- Did memory usage drift upward during the run?
- Did a recent change cause a regression?

## How TraceML Fits

TraceML is the first pass before heavyweight profiling. It tells you which
class of bottleneck to investigate next.

| Tool                              | Use it for | Output | Cost |
|-----------------------------------|---|---|---|
| **TraceML**                       | Classify training bottlenecks: input, compute, wait, memory, rank skew | `final_summary.json`, text summary, live views | Small code wrapper |
| `torch.profiler`                  | Inspect expensive ops, kernels, and CUDA activity | Profiler trace | Profiler schedule/context |
| Nsight Systems / Compute          | Debug low-level CUDA and kernel behavior | GPU timeline and kernel detail | Separate profiler run |
| W&B / MLflow / TensorBoard        | Track experiments and metrics over time | Dashboards and metric history | Logging integration |
| `nvidia-smi` / cluster dashboards | Check machine health and utilization | System-level metrics | No code changes |

TraceML does not replace these tools. It is the cheap first pass that tells you where to look.

---

## Quickstart

Install:

```bash
pip install traceml-ai
```

Initialize TraceML and wrap your training step:

```python
import traceml_ai as traceml

traceml.init()

for batch in dataloader:
    with traceml.trace_step(model):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["x"])
        loss = criterion(outputs, batch["y"])
        loss.backward()
        optimizer.step()
```

Run your script with the `traceml` CLI:

```bash
traceml run train.py
```

> The CLI command is `traceml`. New Python code should use
> `import traceml_ai as traceml`. The old `import traceml` path still works for
> now, but emits a `FutureWarning` and will be removed in a future release.

TraceML writes two end-of-run artifacts:

```text
logs/<run_name>/final_summary.json
logs/<run_name>/final_summary.txt
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
|  r2/r1                                                                     |
|  - Why: r0 input was slower than median global rank (254.5/3.8ms).         |
|                                                                            |
|  Step Memory                                                               |
|  - Diagnosis: BALANCED                                                     |
|  - Scope: last 460 aligned steps                                           |
|  - Stats: peak reserved worst 192 MB on r0 | skew 0.0%                     |
|  - Why: No clear pressure, imbalance, or creep signal.                     |
+----------------------------------------------------------------------------+
```

For experiment trackers, call `traceml.summary()` near the end of your script
to get a flat dict of diagnosis statuses and average metrics. Keep
`final_summary.json` when you want the full run artifact or an input for
`traceml compare`.

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

### Live CLI view

![TraceML live CLI view](docs/assets/cli_demo_v1.png)

Live CLI view while TraceML collects the same signals used for `final_summary.json`.

---

## Modes

All modes write `final_summary.json` and `final_summary.txt` at the end of the run. The mode controls only what you see during training.

| Mode | During training | Topology |
|------|----------------|----------|
| `--mode=summary` | Silent | single-node and multi-node multi-GPU |
| `--mode=cli` | Live terminal display | single-node, including multi-GPU |
| `--mode=dashboard` | Live browser display | single-node, including multi-GPU |

Summary mode is the default and works across all topologies. Use `--mode=cli` or `--mode=dashboard` when you want live feedback on a single-node job.
Dashboard mode requires the optional dashboard extra:

```bash
pip install "traceml-ai[dashboard]"
```

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
  --run-name=my-run
```

On node 1:

```bash
traceml run train.py \
  --nnodes=2 \
  --node-rank=1 \
  --nproc-per-node=4 \
  --master-addr=<node0-ip> \
  --run-name=my-run
```

Use the same `--run-name`, `--nnodes`, `--nproc-per-node`, and
`--master-addr` on every node. Node 0 starts the TraceML aggregator. Other
nodes connect to `<node0-ip>:29765` by default. If workers need a different
reachable address or port for TraceML telemetry, add
`--aggregator-host=<host>` or `--aggregator-port=<port>` on every node. For
multi-node runs, node 0 binds the aggregator to `0.0.0.0` by default; override
that only when needed with `--aggregator-bind-host=<bind-host>`.

`--session-id` remains accepted as a backward-compatible alias for
`--run-name`.

### Watch mode (no code changes)

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

## Current support

**Works today:**

- Single GPU training
- Single-node multi-GPU DDP / FSDP training
- Multi-node DDP summary reports
- Ray Train through a thin `TorchTrainer` wrapper
- Step Time, Step Memory, System, and Process diagnostics
- Run-to-run comparison from `final_summary.json`
- Custom PyTorch loops, Hugging Face, and PyTorch Lightning

**Next:**

- Slurm launch examples
- Broader multi-node FSDP validation
- Multi-node live CLI / dashboard
- Explicit collective / NCCL timing

---

## Overhead

**Overhead:** In our benchmark runs, TraceML adds <2% overhead on single GPU and <1% on single-node multi-GPU at default settings.

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
- [Ray Train integration](docs/user_guide/integrations/ray.md)

---

## Feedback

If TraceML helped, a GitHub star helps others find it.

If you hit a problem or unexpected result, open an issue and include:

- hardware / CUDA / PyTorch versions
- single GPU or multi-GPU setup
- training framework
- the end-of-run summary
- a minimal repro if possible

GitHub issues: [open an issue](https://github.com/traceopt-ai/traceml/issues)

Security reports: see [SECURITY.md](SECURITY.md)

Email: [support@traceopt.ai](mailto:support@traceopt.ai)

---

## Contributing

Contributions are welcome, especially:

- real slowdown examples and repros
- distributed training edge cases
- docs improvements
- framework integrations

---

## License

Apache 2.0. See [LICENSE](LICENSE).

TraceOpt is a trademark of OptAI UG (haftungsbeschränkt).
