# Examples

This folder contains the easiest ways to try TraceML without reading the full codebase.

If you are new to TraceML, start here.

The scripts in this folder are available only in a repository checkout; they
are not included in the PyPI wheel. From the repository root, install the
dependencies for the zero-code starter example:

```bash
pip install ".[torch,hf]"
```

All commands below assume that checkout directory.

---

## Start here

```bash
traceml run examples/integrations/huggingface_trainer_minimal.py \
  --args --steps 20
```

This is a standard Hugging Face `Trainer` script with no TraceML code, model
download, or dataset download. TraceML detects the Trainer, prints the final
diagnosis, and writes JSON/TXT artifacts. Keep the final summary JSON to
compare runs later with `traceml compare`.

Prefer Colab? Browse the [runnable notebooks](../notebooks/README.md).

## Basic examples

| Example | What it shows | Works on |
|---|---|---|
| [Plain PyTorch quickstart](quickstart.py) | Plain PyTorch loop with an explicit step boundary and a final summary | CPU / CUDA |
| [Summary logging](summary_logging_minimal.py) | Export `traceml.summary()` for W&B or MLflow | CPU / CUDA |
| [Custom instrumentation](manual_custom_minimal.py) | Custom batch source and explicit wrappers in manual mode | CPU / CUDA |

## Framework integrations

For the Hugging Face examples, install `pip install ".[torch,hf]"`; the ViT
example also needs `datasets`. Launch them with `traceml run` for automatic
instrumentation.

| Example | What it shows | Requirements |
|---|---|---|
| [Hugging Face Trainer](integrations/huggingface_trainer_minimal.py) | Standard Trainer script with no TraceML code | CPU / CUDA; no model download |
| [Hugging Face ViT](integrations/huggingface_vision_vit.py) | Standard Trainer image classification on CIFAR-10 | CPU / CUDA; downloads model and data |
| [Accelerate](integrations/accelerate_minimal.py) | Accelerator loop with `trace_step` | CPU / CUDA; no model download |
| [Lightning](integrations/lightning_minimal.py) | Standard Trainer script with no TraceML code | CPU / CUDA; no dataset download |
| [Lightning loader comparison](integrations/lightning_dataloading_bottleneck.py) | Compare DataLoader profiles on ResNet-18 and 320px Imagenette | CUDA; 326 MB download; CPU `--smoke`; companion Colab notebook |
| [MONAI](integrations/monai_minimal.py) | `SupervisedTrainer` with `TraceMLHandler` | CPU / CUDA; synthetic volumes, no download; [guide](../docs/user_guide/integrations/monai.md) |
| [MONAI pipeline comparison](integrations/monai_dataloading_bottleneck.py) | Compare loading, caching and compute settings on a 3D UNet | CUDA; 1.6 GB spleen dataset (CC BY-SA 4.0), `nibabel`; CPU `--smoke`; companion Colab notebook |
| [RF-DETR](integrations/rfdetr_minimal.py) | Try Nano training with generated sample data or your own COCO export | CPU / CUDA; `rfdetr[train]==1.10.1`, `--demo` needs no dataset download; pretrained weights download; [guide](../docs/user_guide/integrations/rfdetr.md) |
| [DeepSpeed](integrations/deepspeed_minimal.py) | Engine loop with `trace_step` | CUDA; requires `deepspeed`, exits cleanly without it |
| [Ray Train](integrations/ray/torchtrainer_minimal.py) | `TraceMLTorchTrainer` with Ray Data input timing | CPU / CUDA |
| [Ray + Lightning](integrations/ray/lightning_text_classifier.py) | Text classifier with Ray Data, `TraceMLCallback`, and input/H2D controls | CPU / CUDA |

To try RF-DETR without preparing a dataset:

```bash
pip install "traceml-ai==0.5.0" "rfdetr[train]==1.10.1"
traceml run examples/integrations/rfdetr_minimal.py --args \
  --demo --output-dir checkpoints/rfdetr-demo --epochs 1
```

Run from this checkout; no editable install is needed. The demo generates
temporary sample images and uses CUDA when available, otherwise CPU. RF-DETR
downloads pretrained weights on first use; CPU training can be slow. Use a new
checkpoint directory for each attempt. For real data, replace `--demo` with
`--dataset-dir data/coco`. Sample-data timings are not benchmark results.

## Distributed training

| Example | What it shows | Works on |
|---|---|---|
| [DDP](distributed/ddp_minimal.py) | Minimal single-node distributed loop | CPU / CUDA |
| [FSDP](distributed/fsdp_minimal_cuda.py) | Sharded training with `trace_step` | CUDA |
| [Slurm](distributed/slurm/README.md) | Multi-node launch templates | Slurm cluster |

---

## Diagnosis demos

Controlled examples for understanding TraceML's timing signals and diagnoses.

| Example | What it demonstrates | Works on | Notes |
|---|---|---|---|
| [diagnosis/dataloader_bottleneck_demo.py](diagnosis/dataloader_bottleneck_demo.py) | Slow input pipeline or input-bound training | CPU / CUDA | Simulates dataloader delay |
| [distributed/ddp_rank_straggler_demo.py](distributed/ddp_rank_straggler_demo.py) | Rank stragglers in DDP | CPU / CUDA | Simulates balanced, input-straggler, and compute-straggler runs |
| [diagnosis/step_memory_creep_demo.py](diagnosis/step_memory_creep_demo.py) | Step memory creep (`MEMORY CREEP`) | CUDA for the verdict; runs on CPU | Retains 8 MiB of CUDA memory per step on every rank; on CPU it runs without leaking and Step Memory reports `NO GPU` |
| [H2D timing](diagnosis/h2d_timing_demo.py) | Inspect host-to-device transfer timings per step | CUDA for timing; runs on CPU | CPU-only moves are reported as absent |
| [diagnosis/incomplete_signals_demo.py](diagnosis/incomplete_signals_demo.py) | Missing signals reported as absent, not zero (`INCOMPLETE DATA`) | CPU / CUDA | Calls `model.forward(...)` directly, so forward timing is never recorded |

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

To see a uniform step-memory leak reported as creep (requires CUDA):

```bash
traceml run examples/diagnosis/step_memory_creep_demo.py --args --steps 300
```

To see Step Time refuse a verdict when the forward signal is missing:

```bash
traceml run examples/diagnosis/incomplete_signals_demo.py
```

---

## Workload comparisons

Run controlled workloads to investigate hardware or batch configuration.

| Example | What it demonstrates | Works on | Notes |
|---|---|---|---|
| [workloads/bert_single_gpu_compare.py](workloads/bert_single_gpu_compare.py) | Run the same fixed BERT workload on different single-GPU machines, then compare TraceML summaries | CUDA | Use the same batch size, sequence length, precision, and step count on each machine |
| [`workloads/qwen3_8b_lora_ga`](workloads/qwen3_8b_lora_ga/) | Measure physical batch size and gradient accumulation with Qwen3-8B TRL LoRA while holding effective batch and packed-token capacity constant | CUDA | Production-shaped single-L40S workload; includes a 500-step matrix runner |
| [BERT gradient accumulation](workloads/bert_gradient_accum.py) | Group microbatches into optimizer updates in a plain PyTorch BERT loop | CPU / CUDA | Downloads BERT and AG News; batch and accumulation settings are in the script |

Example hardware comparison run:

```bash
traceml run examples/workloads/bert_single_gpu_compare.py --mode=summary --run-name bert_l40s_bs32_seq256 --args --model-name bert-large-uncased --batch-size 32 --max-length 256 --max-steps 350 --warmup-steps 50 --num-workers 4 --precision fp16
```

---

## Case studies

[Browse case studies](case_studies/README.md) for measured investigations and
reproduction packages, including ResNet, LeRobot, and RF-DETR.

---

## How to run examples

Standard run with the default summary:

```bash
traceml run examples/quickstart.py
```

For the live browser dashboard, install the dashboard extra and select
dashboard mode. It listens on `http://127.0.0.1:8765` by default:

```bash
pip install "traceml-ai[dashboard]"
traceml run examples/quickstart.py --mode=dashboard
```

Choose another local browser port with `--dashboard-port`:

```bash
traceml run examples/quickstart.py --mode=dashboard --dashboard-port=9000
```

On a remote machine, forward that dashboard port before opening the browser on
your laptop:

```bash
ssh -L 8765:127.0.0.1:8765 user@remote-host
```

Then open `http://127.0.0.1:8765` locally. The launcher also prints this URL
and SSH tunnel command in a boxed message after the aggregator and training
process have launched.

Terminal UI:

```bash
traceml run examples/quickstart.py --mode=cli
```

Summary mode:

```bash
traceml run examples/quickstart.py --mode=summary
```

Single-node DDP:

```bash
traceml run examples/distributed/ddp_minimal.py --nproc-per-node=4
```

For a short smoke run, cap the number of optimizer steps completed by each
rank:

```bash
traceml run examples/distributed/ddp_minimal.py \
  --nproc-per-node=2 --args --steps 20
```

The same `--steps` option sets the run length of `quickstart.py`,
`summary_logging_minimal.py`, `manual_custom_minimal.py`,
`integrations/huggingface_trainer_minimal.py`,
`integrations/accelerate_minimal.py`, `integrations/deepspeed_minimal.py`,
`distributed/fsdp_minimal_cuda.py`, `diagnosis/h2d_timing_demo.py`,
`diagnosis/step_memory_creep_demo.py`, and
`diagnosis/incomplete_signals_demo.py`. `accelerate_minimal.py`,
`deepspeed_minimal.py`, `fsdp_minimal_cuda.py`, `step_memory_creep_demo.py`
and `incomplete_signals_demo.py` also accept `--epochs`. Pass `--args --help`
to see each default.

DeepSpeed (single or multi-GPU; requires `deepspeed` + a CUDA GPU):

```bash
traceml run examples/integrations/deepspeed_minimal.py --mode=summary
traceml run examples/integrations/deepspeed_minimal.py --nproc-per-node=2 --mode=summary
```

Multi-node on Slurm:

```bash
sbatch examples/distributed/slurm/traceml_ddp.sbatch
```

See [`examples/distributed/slurm/`](distributed/slurm/README.md) and the
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

The minimal Lightning example is an unchanged Trainer script: `traceml run`
initializes timing and attaches the callback. The existing
`integrations/lightning_dataloading_bottleneck.py` uses the advanced manual API
and remains valid. Run it twice with `--profile baseline` and
`--profile optimized`, then `traceml compare` the two summaries (the module
docstring carries the exact commands).

Ray Data examples wrap `iter_torch_batches(...)` with
`traceml.wrap_dataloader_fetch(...)` because Ray Data iterators are not PyTorch
`DataLoader` objects. `TraceMLTorchTrainer` initializes each worker before the
training function creates this wrapper.

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
`import traceml_ai as traceml`.

---

## Related docs

- [Quickstart](../docs/user_guide/quickstart.md)
- [Distributed Training](../docs/user_guide/distributed-training.md)
- [Running on Slurm](../docs/user_guide/slurm.md)
- [Compare Runs](../docs/user_guide/compare.md)
- [How to Read TraceML Output](../docs/user_guide/reading-output.md)
- [Use With Your Stack](../docs/user_guide/integrations.md)
- [FAQ](../docs/user_guide/faq.md)
