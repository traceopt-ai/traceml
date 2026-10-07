# RF-DETR Integration

## Install

This guide assumes PyTorch and RF-DETR's training dependencies are already
installed.

```bash
pip install traceml-ai
```

## Run

```bash
traceml run train.py
```

Run your existing script with TraceML. Standard RF-DETR `model.train()`
training is instrumented automatically, with no code changes required.
For direct `python`/`torchrun` launches, see
[Advanced: manual setup](#advanced-manual-setup).

## Read the result

When training finishes, TraceML prints a diagnosis of the likely bottleneck
and saves the report for comparison or CI.

The timing breakdown shows input waiting, forward, backward, and optimizer
work. CUDA runs also show available host-to-device and memory measurements.

See [How to Read Output](../reading-output.md) for an example report and
explanations.

## How TraceML measures RF-DETR training

TraceML hooks RF-DETR's `build_trainer()` factory and attaches its specialized
Lightning callback when a training Trainer is created. RF-DETR continues to
manage its training loop, EMA, checkpoints, evaluation, and loggers.
Importing RF-DETR alone does not initialize timing.

```text
RF-DETR training path                TraceML measurement
──────────────────────────────────────────────────────────────────
model.train() → build_trainer()      Attach RF-DETR callback
          ↓
Training DataLoader fetch            Input Wait
          ↓
Lightning batch transfer             Open capture + GPU transfer
          ↓
Inner detection model                Forward
          ↓
Lightning backward hooks             Backward
(repeat for accumulating microbatches)
          ↓
Optimizer → batch-end callback       Complete step when update is due
```

One reported step covers one optimizer update attempt. With
`grad_accum_steps=4`, four microbatches contribute to that step. TraceML adds
their timings and reports peak CUDA memory within the group's memory window.
The final group can contain fewer microbatches.

**Forward** measures the inner detection model. Loss computation and Hungarian
matching can appear in **Residual**, which is not automatically wasted time.
Input waiting is separate from traced training time. Dataset previews, sanity
checks, validation, and final evaluation are excluded from training-step
measurements.

**Checkpoint resume:** Continue using RF-DETR's normal `resume` argument.
TraceML records resumed training with step numbering local to the process,
which may differ from Lightning's restored `global_step`. See the
[Lightning step-time contract](../../developer_guide/step-time-pipeline-contract.md#lightning-steps)
for the callback's timing boundaries.

## Multi-GPU training

For single-node multi-GPU DDP:

```bash
traceml run train.py --nproc-per-node=4
```

Use the matching RF-DETR training configuration (`devices=4`, `num_nodes=1`).
TraceML launches one worker per GPU. The
[minimal example](https://github.com/traceopt-ai/traceml/blob/main/examples/integrations/rfdetr_minimal.py)
derives these values from the launcher environment. Its batch size is per rank.
For multi-node launch commands, see
[Distributed Training](../distributed-training.md#multi-node-ddp). All nodes
need the same environment and dataset, with checkpoint output accessible to
all ranks.

## Try the example

From a repository checkout, install the training dependencies and run Nano
with generated sample data:

```bash
pip install "traceml-ai==0.5.0" "rfdetr[train]==1.10.1"
traceml run examples/integrations/rfdetr_minimal.py --args \
  --demo --output-dir checkpoints/rfdetr-demo --epochs 1
```

The demo creates 32 training images and four images each for validation and
test in a temporary directory, removed when training finishes. No dataset
download or Roboflow account is needed. RF-DETR downloads pretrained weights
on first use. The example selects CUDA when available, otherwise CPU; CPU
training can be slow. Use a new checkpoint directory for each attempt.
Synthetic data verifies the integration; its timings are not benchmark evidence.

For your own data, replace `--demo` with `--dataset-dir data/coco`. The example
follows RF-DETR's [standard training API](https://rfdetr.roboflow.com/learn/train/)
and needs no TraceML imports or callbacks.

## Advanced: manual setup

Existing scripts using the RF-DETR integration's `init()` also work with
`traceml run`; TraceML reuses compatible setup without adding another callback.

Use this path for direct `python`/`torchrun` launches. Initialize before training
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
For a direct launch, start an aggregator with `traceml serve` first; see
[Direct Launch](../public-api.md#direct-launch-with-traceml-serve).
Compatible repeated initialization is a no-op. Explicit incompatible
initialization retains its existing error behavior. Under automatic launch,
an incompatible pre-existing configuration is left intact with a warning.

### Input-pipeline comparison example

<details markdown="1">
<summary>Compare DataLoader worker counts</summary>

The [runnable example](https://github.com/traceopt-ai/traceml/blob/main/examples/integrations/rfdetr_minimal.py)
uses Nano, seed 42, 384px inputs, no multi-scale resizing, and no gradient
accumulation. It accepts a Roboflow COCO export with images and
`_annotations.coco.json` in each of `train/`, `valid/`, and `test/`.
Use a small dataset for the first smoke run; CPU Nano training can be slow.

From the repository root, run the same experiment with zero and two loader
workers per training process:

```bash
traceml run --logs-dir logs --run-name rfdetr_workers0 \
  examples/integrations/rfdetr_minimal.py \
  --args --dataset-dir data/coco --output-dir checkpoints/workers0 \
  --epochs 2 --batch-size 2 --num-workers 0 --accelerator cuda

traceml run --logs-dir logs --run-name rfdetr_workers2 \
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

</details>

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

- **Timing boundaries.** Input Wait measures batch fetching in the training
  process, not total preprocessing in background workers. Backward includes
  distributed synchronization performed there. Optimizer timing can include
  scheduler work, EMA updates, and earlier batch-end callbacks.
- **Memory window.** Peak CUDA memory follows the Lightning callback's window,
  which starts after batch transfer. Earlier allocation peaks are excluded;
  validation between accumulating microbatches can affect the group's peak.
- **Run duration.** Whole-process duration includes startup, evaluation, and
  checkpoint work. RF-DETR may download weights on the first run; cache them
  before comparing steady-state training. Repeat sufficiently long runs to
  account for warm-up. `--trace-max-steps` limits recording, not training; use
  the example's `--epochs` to limit its training duration.
- **Versions.** The adapter warns once on rank zero for versions other than
  its CI pin, 1.10.1. The case studies below provide additional evidence,
  rather than a compatibility guarantee.

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

## Next Steps

- [How to Read Output](../reading-output.md)
- [Compare Runs](../compare.md)
- [Catch Regressions in CI](../regression-guard.md)
- [Distributed Training](../distributed-training.md)
- [W&B / MLflow](wandb-mlflow.md)
- [Open an issue](https://github.com/traceopt-ai/traceml/issues)
