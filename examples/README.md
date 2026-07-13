# Examples

This folder contains the easiest ways to try TraceML without reading the full codebase.

If you are new to TraceML, start here.

---

## Start here

These are the main user-facing examples.

| Example | What it shows | Works on | Notes |
|---|---|---|---|
| `quickstart.py` | Minimal plain PyTorch loop with `traceml.init(mode="auto")`, `traceml.trace_step(...)`, and `traceml.summary(...)` | CPU / CUDA | Best first example |
| `summary_logging_minimal.py` | Minimal tracker-friendly `traceml.summary()` output for W&B or MLflow logging | CPU / CUDA | Best summary API example |
| `manual_custom_minimal.py` | Manual TraceML instrumentation with a custom batch source and explicit wrappers | CPU / CUDA | Best starting point for `mode="manual"` |
| `distributed/ddp_minimal.py` | Minimal single-node DDP example | CPU / CUDA | Best distributed starter |
| `ray/torchtrainer_minimal.py` | Minimal Ray Train example with Ray Data input timing | CPU / CUDA | Uses `TraceMLTorchTrainer` |
| `ray/lightning_text_classifier.py` | Ray Train + Lightning text classifier | CPU / CUDA | Uses Ray Data, `TraceMLCallback`, and optional input/H2D demo knobs |
| `integrations/huggingface_trainer_minimal.py` | Minimal Hugging Face `TraceMLTrainer` example | CPU / CUDA | No model download required |
| `integrations/lightning_minimal.py` | Minimal Lightning integration init + `TraceMLCallback` example | CPU / CUDA | No dataset download required |

If you only try one example first, use:

```bash
traceml run examples/quickstart.py --mode=summary
```

Then keep the TraceML final summary JSON if you want to compare runs later with `traceml compare`.

---

## Diagnosis demos

These examples are still user-facing, but they are more about showing specific TraceML diagnoses than showing the smallest integration.

| Example | What it demonstrates | Works on | Notes |
|---|---|---|---|
| `diagnosis/dataloader_bottleneck_demo.py` | Slow input pipeline or input-bound training | CPU / CUDA | Simulates dataloader delay |
| `distributed/ddp_rank_straggler_demo.py` | Rank stragglers in DDP | CPU / CUDA | Simulates balanced, input-straggler, and compute-straggler runs |

These are useful when you want to see how TraceML behaves on a known bottleneck.

To contrast a normal input path with a synthetic input pipeline bottleneck:

```bash
traceml run examples/diagnosis/dataloader_bottleneck_demo.py --args --scenario fast
traceml run examples/diagnosis/dataloader_bottleneck_demo.py --args --scenario slow --sleep-ms 8
```

Use `--num-workers` on the same demo to test whether adding DataLoader workers
reduces the input wait.

On a fast GPU, increase model compute while keeping the same fast/slow shape:

```bash
traceml run examples/diagnosis/dataloader_bottleneck_demo.py --args --scenario fast --hidden-dim 4096 --depth 4
```

To contrast balanced DDP with rank-local input and compute stragglers:

```bash
traceml run examples/distributed/ddp_rank_straggler_demo.py --mode=summary --nproc-per-node=2 --run-name ddp_balanced --args --scenario balanced
traceml run examples/distributed/ddp_rank_straggler_demo.py --mode=summary --nproc-per-node=2 --run-name ddp_input_straggler --args --scenario input-straggler --straggler-rank 0 --input-sleep-ms 200
traceml run examples/distributed/ddp_rank_straggler_demo.py --mode=summary --nproc-per-node=2 --run-name ddp_compute_straggler --args --scenario compute-straggler --straggler-rank 0 --compute-extra-matmuls 8
```

The default DDP demo uses precomputed tensors plus a compute-heavy MLP so the
balanced run is not dominated by tiny batches or synthetic input overhead on
GPUs such as T4 or L4.

---

## Advanced workloads

These are real or heavier workloads intended for focused investigations, not
first-run examples.

| Example | What it demonstrates | Works on | Notes |
|---|---|---|---|
| `advanced/bert_single_gpu_compare.py` | Run the same fixed BERT workload on different single-GPU machines, then compare TraceML summaries | CUDA | Use the same batch size, sequence length, precision, and step count on each machine |

Example hardware comparison run:

```bash
traceml run examples/advanced/bert_single_gpu_compare.py --mode=summary --run-name bert_l4_bs16_seq128 --args --batch-size 16 --max-length 128 --max-steps 200
```

---

## How to run examples

Standard run:

```bash
traceml run examples/quickstart.py
```

Local UI:

```bash
traceml run examples/quickstart.py --mode=dashboard
```

Summary mode:

```bash
traceml run examples/quickstart.py --mode=summary
```

Single-node DDP:

```bash
traceml run examples/distributed/ddp_minimal.py --nproc-per-node=4
```

Multi-node on Slurm:

```bash
sbatch examples/slurm/traceml_ddp.sbatch
```

See [`examples/slurm/`](slurm/README.md) and the
[Slurm guide](../docs/user_guide/slurm.md) for the template and the
network/aggregator model.

Run without TraceML telemetry for a baseline:

```bash
traceml run examples/quickstart.py --disable-traceml
```

Compare two saved TraceML final summary JSON files:

```bash
traceml compare run_a.json run_b.json
```

Starter examples now prefer the top-level public API:

- `traceml.init(mode="auto")`
- `traceml.trace_step(...)`
- `traceml.summary()`
- `traceml.final_summary()`

Lightning examples use `traceml_ai.integrations.lightning.init()` with
`TraceMLCallback()` so Lightning can keep owning the training loop while
TraceML records input fetch, transfer, step, phase, and memory timing.

Ray Data examples wrap `iter_torch_batches(...)` with
`traceml.wrap_dataloader_fetch(...)` because Ray Data iterators are not PyTorch
`DataLoader` objects.

Ray + Lightning can use `--input-delay-ms` / `--input-delay-rank` for input
stragglers, `--delay-ms` / `--delay-rank` for compute stragglers, and
`--transfer-dim` to make Lightning H2D timing visible.

For explicit manual instrumentation, see:

- `traceml.init(mode="manual")`
- `traceml.wrap_dataloader_fetch(...)`
- `traceml.wrap_forward(...)`
- `traceml.wrap_backward(...)`
- `traceml.wrap_optimizer(...)`

Examples use the top-level `traceml.*` API from
`import traceml_ai as traceml`. The old `import traceml` path remains available
for compatibility, but emits a deprecation warning. Do not import from
decorator compatibility paths.

---

## Which example should I use?

Use:

- `quickstart.py` if you have a normal PyTorch loop
- `manual_custom_minimal.py` if you use a custom input pipeline or want full explicit control
- `distributed/ddp_minimal.py` if you want single-node distributed training
- `integrations/huggingface_trainer_minimal.py` if you use Hugging Face `Trainer`
- `integrations/lightning_minimal.py` if you use PyTorch Lightning
- `ray/torchtrainer_minimal.py` if you use Ray Train

Use the diagnosis demos when you want to see:

- an input bottleneck
- an input straggler in DDP

---

## What is not in this folder

Heavier development and stress scenarios are kept separately from these starter examples so this folder stays easy to understand.

That includes things like:

- large BERT DDP runs
- memory-creep stress scripts
- FSDP experiments
- heavy vision or LLM demos

---

## Related docs

- [Quickstart](../docs/user_guide/quickstart.md)
- [Distributed Training](../docs/user_guide/distributed-training.md)
- [Running on Slurm](../docs/user_guide/slurm.md)
- [Compare Runs](../docs/user_guide/compare.md)
- [How to Read TraceML Output](../docs/user_guide/reading-output.md)
- [Use With Your Stack](../docs/user_guide/integrations.md)
- [FAQ](../docs/user_guide/faq.md)
