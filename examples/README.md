# Examples

This folder contains the easiest ways to try TraceML without reading the full codebase.

If you are new to TraceML, start here.

---

## Start here

These are the main user-facing examples.

| Example | What it shows | Works on | Notes |
|---|---|---|---|
| `pytorch_minimal.py` | Minimal plain PyTorch loop with `traceml.trace_step(...)` and `traceml.final_summary()` | CPU / CUDA | Best first example |
| `ddp_minimal.py` | Minimal single-node DDP example | CPU / CUDA | Best distributed starter |
| `huggingface_trainer_minimal.py` | Minimal Hugging Face `TraceMLTrainer` example | CPU / CUDA | No model download required |
| `lightning_minimal.py` | Minimal Lightning `TraceMLCallback` example | CPU / CUDA | No dataset download required |

If you only try one example first, use:

```bash
traceml run examples/pytorch_minimal.py
```

---

## Diagnosis demos

These examples are still user-facing, but they are more about showing specific TraceML diagnoses than showing the smallest integration.

| Example | What it demonstrates | Works on | Notes |
|---|---|---|---|
| `input_bound_demo.py` | Slow input pipeline / input-bound training | CPU / CUDA | Simulates dataloader delay |
| `input_straggler_ddp_demo.py` | Input straggler in single-node DDP | CPU / CUDA | One rank is deliberately slower in the input path |

These are useful when you want to see how TraceML behaves on a known bottleneck.

---

## How to run examples

Standard run:

```bash
traceml run examples/pytorch_minimal.py
```

Local UI:

```bash
traceml run examples/pytorch_minimal.py --mode=dashboard
```

Single-node DDP:

```bash
traceml run examples/ddp_minimal.py --nproc-per-node=4
```

Run without TraceML telemetry for a baseline:

```bash
traceml run examples/pytorch_minimal.py --disable-traceml
```

Starter examples now prefer the top-level public API:

- `traceml.trace_step(...)`
- `traceml.trace_model_instance(...)`
- `traceml.final_summary()`

Legacy imports from `traceml.decorators` still work for backward
compatibility, but new examples use the top-level `traceml.*` API.

---

## Which example should I use?

Use:

- `pytorch_minimal.py` if you have a normal PyTorch loop
- `ddp_minimal.py` if you want single-node distributed training
- `huggingface_trainer_minimal.py` if you use Hugging Face `Trainer`
- `lightning_minimal.py` if you use PyTorch Lightning

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

- [Quickstart](../docs/quickstart.md)
- [How to Read TraceML Output](../docs/how-to-read-output.md)
- [FAQ](../docs/faq.md)
- [Use TraceML with W&B / MLflow](../docs/use-with-wandb-mlflow.md)
- [Hugging Face Trainer](../docs/huggingface.md)
- [PyTorch Lightning](../docs/lightning.md)
