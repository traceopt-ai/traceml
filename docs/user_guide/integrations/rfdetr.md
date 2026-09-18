# RF-DETR

Use TraceML to see where RF-DETR training time goes and compare configuration
changes. Keep RF-DETR's normal `model.train()` API; TraceML attaches its callback
without replacing RF-DETR's EMA, checkpoints, evaluation, or loggers.

## Install and initialize

Install RF-DETR separately; it is not a TraceML dependency:

```bash
pip install "traceml-ai[torch]" "rfdetr[train]==1.10.1"
```

Add initialization before training, in every worker:

```python
from rfdetr import RFDETRNano
from traceml_ai.integrations import rfdetr as traceml_rfdetr

traceml_rfdetr.init()
model = RFDETRNano(device="cpu")
model.train(
    dataset_dir="data/coco", output_dir="checkpoints/run_a", device="cpu"
)
```

Use `device="cuda"` in both calls for GPU training. Run the script with
`traceml run train.py` to start the telemetry collector.
RF-DETR may download pretrained weights on the first run. Do a separate smoke
run first so both measured runs reuse cached weights. TraceML does not change
checkpoint or weight licensing.

## First workflow: compare input loading

The [runnable example](https://github.com/traceopt-ai/traceml/blob/main/examples/integrations/rfdetr_minimal.py)
uses Nano, seed 42, 384px inputs, no multi-scale resizing, and no gradient
accumulation. It accepts a Roboflow COCO export with images and
`_annotations.coco.json` in each of `train/`, `valid/`, and `test/`.
Use a small dataset for the first smoke run; CPU Nano training can be slow.

From the repository root, run the same experiment with zero and two loader
workers per training process:

```bash
traceml run --mode summary --logs-dir logs --run-name rfdetr_workers0 \
  examples/integrations/rfdetr_minimal.py \
  --args --dataset-dir data/coco --output-dir checkpoints/workers0 \
  --epochs 2 --batch-size 2 --num-workers 0 --accelerator cuda

traceml run --mode summary --logs-dir logs --run-name rfdetr_workers2 \
  examples/integrations/rfdetr_minimal.py \
  --args --dataset-dir data/coco --output-dir checkpoints/workers2 \
  --epochs 2 --batch-size 2 --num-workers 2 --accelerator cuda

traceml compare logs/rfdetr_workers0/final_summary.json \
  logs/rfdetr_workers2/final_summary.json --output comparisons/rfdetr_workers
```

Use `--accelerator cpu` in both runs for CPU training. The example also accepts
`auto`, which selects CUDA when available and otherwise CPU. Use fresh run names
and checkpoint directories for each attempt; the example refuses an existing
checkpoint directory. Final summaries live under `logs/<run-name>/`; the
comparison writes JSON and text files under `comparisons/`. RF-DETR writes its
checkpoints and training configuration to `--output-dir`.

Look for changes in Input Wait, Step Time and rank imbalance. More workers may
help, have no effect, or slow training. Keep the dataset, batch size, topology,
precision, seed and epochs fixed, and repeat meaningful comparisons to check
variance. Loader worker changes can change random augmentation sequences even
with the same seed, so this is a throughput experiment, not an accuracy claim.
See [Compare Runs](../compare.md) for report details.

## CPU and CUDA DDP

The example derives RF-DETR's `devices` and `num_nodes` from the launcher and
initializes TraceML on every rank. For two CPU/Gloo processes:

```bash
traceml run --nproc-per-node=2 --run-name rfdetr_cpu_ddp \
  examples/integrations/rfdetr_minimal.py \
  --args --dataset-dir data/coco --output-dir checkpoints/cpu_ddp \
  --epochs 1 --batch-size 2 --num-workers 0 --accelerator cpu
```

For two CUDA/NCCL processes, use `--accelerator cuda`, a new run name and a new
checkpoint directory. Launch one process per GPU. `--batch-size` is per rank;
effective batch size in this example is `batch_size * WORLD_SIZE`.

For multi-node runs, use the same script with the node-specific launch flags in
[Distributed Training](../distributed-training.md#multi-node-ddp). Every node
needs the same dataset contents and environment; RF-DETR checkpoint output must
be accessible to all ranks, normally on a shared filesystem. Node 0 writes the
TraceML final summary. Use homogeneous GPU hardware when comparing rank timing.

## Reading the measurements

- **Input Wait:** observed time waiting for the next PyTorch DataLoader batch,
  not total preprocessing time in background workers.
- **Forward:** RF-DETR's inner detection model. Loss computation and Hungarian
  matching can appear in **Residual**; residual is not automatically wasted time.
- **Backward:** includes distributed synchronization that occurs during backward.
- **Optimizer:** the update region, which can include scheduler work, EMA updates
  and batch-end callbacks that run before TraceML's callback.

One TraceML step is one optimizer-update attempt. With gradient accumulation,
micro-batch measurements are combined, including a shorter final group.
Validation, sanity checks, dataset previews and final evaluation are excluded
from training-step measurements. Whole-process duration still includes startup,
evaluation and checkpoint work. Short runs include warm-up effects; compare
enough steps to avoid treating startup as steady-state performance.
`--trace-max-steps` caps recording, **not training**; use `--epochs` to limit this
example's training duration.

## Limitations and validation

The first integration covers eager object detection with automatic optimization
on CPU/CUDA, single process or ordinary DDP. It does not support segmentation,
keypoints, `torch.compile`, CUDA graphs, FSDP/DeepSpeed, TPU/MPS, notebook process
spawning, or automatic attachment to the separate `rfdetr fit` CLI. Known
unsupported configurations and adapter setup failures warn to stderr and skip
the RF-DETR callback. Training and existing callbacks continue unchanged;
native RF-DETR errors still propagate. Untested versions also warn.
Ordinary DDP includes `ddp_find_unused_parameters_true` and `_false` aliases.
The CUDA-graph guard reads `ModelConfig.cuda_graphs` on newer revisions; that
option is absent in 1.10.1 and the development revision tested below.

The dedicated CPU CI job pins RF-DETR 1.10.1 and exercises numerical parity,
EMA independence, timing boundaries and two-process Gloo collection. The Nano
smoke uses real model kernels, reduced image/query counts and random weights
without downloads; Gloo tests use a tiny detector with RF-DETR's real training
components. These checks also passed locally against RF-DETR development commit
[`0ed5be8`](https://github.com/roboflow/rf-detr/commit/0ed5be8e8d6762c4978a11671cbf34cfc0595e25)
with PyTorch 2.11 and Lightning 2.6.1. This is not a broad version matrix.
CUDA/NCCL and physical multi-node recipes remain unverified until those hardware
runs are completed; CPU tests do not establish GPU coverage. See the
[support matrix](../integrations.md#integration-support-matrix) for evidence.

For an uninstrumented control, add `--disable-traceml` to `traceml run` and choose
a fresh checkpoint directory. This produces no TraceML telemetry. Initialization
is idempotent, and `TRACEML_DISABLED=1` also disables the integration.
