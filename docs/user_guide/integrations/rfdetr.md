# RF-DETR

Use TraceML to see where RF-DETR training time goes and compare configuration
changes. Keep RF-DETR's normal `model.train()` API; TraceML attaches its callback
without replacing RF-DETR's EMA, checkpoints, evaluation, or loggers.

## Install

Install RF-DETR separately; it is not a TraceML dependency:

```bash
pip install "traceml-ai[torch]" "rfdetr[train]==1.10.1"
```

## Run

```bash
traceml run train.py
```

No TraceML code is needed in a standard RF-DETR `model.train()` script.
TraceML initializes timing when RF-DETR creates its training Trainer and adds
its specialized callback. Compatible existing `rfdetr.init()` calls are reused.
At training start, global rank zero prints `RF-DETR Trainer detected` once.
Importing RF-DETR alone does not initialize timing.

RF-DETR may download pretrained weights on the first run. Do a separate smoke
run first so measured runs reuse cached weights. TraceML does not change
checkpoint or weight licensing.

## Read the result

TraceML shows the available phase timings and saves the usual run artifacts.
See [How to Read Output](../reading-output.md) for their meanings.

- **Input Wait:** observed time waiting for the next PyTorch DataLoader batch,
  not total preprocessing time in background workers.
- **Forward:** RF-DETR's inner detection model. Loss computation and Hungarian
  matching can appear in **Residual**; residual is not automatically wasted time.
- **Backward:** includes distributed synchronization that occurs during backward.
- **Optimizer:** the update region, which can include scheduler work, EMA updates
  and batch-end callbacks that run before TraceML's callback.

## What one step includes

One TraceML step is one optimizer-update attempt. With `grad_accum_steps=4`,
four microbatches share one capture and one step number; phase times are added
across that group, including a shorter final group. CUDA step memory reports a
peak rather than a sum.
Validation, sanity checks, dataset previews and final evaluation are excluded
from training-step measurements. Whole-process duration still includes startup,
evaluation and checkpoint work. Short runs include warm-up effects, so compare
enough steps to avoid treating startup as steady-state performance.
`--trace-max-steps` caps recording, **not training**; use `--epochs` to limit this
example's training duration.

Checkpoint resume uses RF-DETR's normal `resume` argument. TraceML step IDs
remain local to the process and can differ from Lightning's restored
`global_step`. Traced Step Time covers the observed training regions; Input Wait
is separate from Traced Step Time.

## Advanced launch options

### Compare input loading

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

### CPU and CUDA DDP

The example derives RF-DETR's `devices` and `num_nodes` from the launcher.
TraceML activates the adapter on every rank. For two CPU/Gloo processes:

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

## Limitations

The integration supports eager object detection with automatic optimization on
CPU or CUDA, using either one process or ordinary DDP. Segmentation, keypoints,
`torch.compile`, CUDA graphs, FSDP, DeepSpeed, TPU/MPS, notebook process spawning
and the separate `rfdetr fit` CLI are not supported. If the adapter encounters
an unsupported configuration or cannot attach its callback, it reports the
reason and RF-DETR continues with its existing callbacks. Native RF-DETR errors
still propagate.

`RFDETR.evaluate()` constructs an evaluation-only trainer with
`include_training_callbacks=False`. TraceML intentionally leaves that trainer
uninstrumented and does not emit RF-DETR step telemetry. A custom trainer built
with the same flag also remains uninstrumented if it is later used with
`fit()`.

RF-DETR 1.10.1 is pinned in CI for single-process training and two-process Gloo
coverage. The [RF-DETR Nano case study](https://github.com/traceopt-ai/traceml/tree/main/examples/case_studies/rfdetr_nano_training)
also exercises eager CUDA training on one and four T4 GPUs against development
commit
[`0ed5be8`](https://github.com/roboflow/rf-detr/commit/0ed5be8e8d6762c4978a11671cbf34cfc0595e25).
The [RF-DETR release-regression case study](https://github.com/traceopt-ai/traceml/tree/main/examples/case_studies/rfdetr_input_pipeline_regression)
compares releases 1.10.1, 1.11.0 and 1.11.1 on a controlled non-JPEG workload
and attributes the observed slowdown to increased input waiting.
This evidence is not a broad compatibility matrix. CUDA CI and physical
multi-node execution have not yet been validated. See the
[support matrix](../integrations.md#integration-support-matrix) for the current
coverage.

## Advanced: manual setup

For direct Python/torchrun launches, initialize the integration before training
in every worker. The adapter adds its specialized callback; do not add a generic
Lightning TraceML callback separately.

```python
from rfdetr import RFDETRNano
from traceml_ai.integrations import rfdetr as traceml_rfdetr

traceml_rfdetr.init()
model = RFDETRNano(device="cpu")
model.train(
    dataset_dir="data/coco", output_dir="checkpoints/run_a", device="cpu"
)
```

Use `device="cuda"` in both model creation and training for GPU runs.
A direct launch needs a running TraceML aggregator; see
[Direct Launch](../public-api.md#direct-launch-with-traceml-serve).
Compatible repeated initialization is a no-op. Explicit incompatible
initialization retains its existing error behavior. Under automatic launch,
an incompatible pre-existing configuration is left intact with a warning.
