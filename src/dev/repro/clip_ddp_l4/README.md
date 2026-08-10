# CLIP DDP GPU-utilization reproduction

This development workload is a configurable reproduction based on the
training and launch code in [Hugging Face Transformers issue #41615][issue].
The issue reports bursty GPU utilization while training CLIP from scratch with
Hugging Face Trainer, DDP/NCCL, BF16, fused AdamW, `torch.compile`, and
Flash Attention 2 on eight A100 GPUs.

The reproduction follows the same training structure while making the
hardware count, dataset source, DataLoader settings, attention backend, and
compile behavior configurable. It can run first on four L4 GPUs and later on
eight A100 GPUs. Results from either environment are recorded separately.

[issue]: https://github.com/huggingface/transformers/issues/41615

## What this reproduces

The goal is to reproduce the same **class of workload and symptom** in a
controlled environment:

- CLIP training from a randomly initialized configuration
- Hugging Face Trainer with single-node DDP
- Raw image and text preprocessing in the DataLoader path
- BF16, fused AdamW, TF32, `torch.compile`, and configurable attention
- Low, bursty, or uneven GPU utilization that requires phase-level diagnosis

The default hardware target is 4×L4. The same code can later run on 8×A100 by
changing the process count. A result from the smaller environment is a
representative reproduction and does not establish the cause of the original
run.

## Quick reproduction

From the TraceML repository root, install the project and Hugging Face
dependencies in an existing CUDA-enabled PyTorch environment:

```bash
python -m pip install -e '.[hf]'
python -m pip install datasets pillow
```

The commands intentionally use the `hf` extra without the `torch` extra so an
existing CUDA-enabled PyTorch installation is not replaced. Install PyTorch
separately only when the GPU environment does not already provide it.

Flash Attention must be installed separately with a build compatible with the
machine's PyTorch and CUDA versions. To verify the command without launching
workers:

```bash
DRY_RUN=1 RUN_NAME=clip-ddp-command-check \
  ./src/dev/repro/clip_ddp_l4/launch.sh
```

Run the issue-derived configuration on four GPUs:

```bash
RUN_NAME=clip-ddp-4l4-baseline \
  ./src/dev/repro/clip_ddp_l4/launch.sh
```

Run the same workload without TraceML for the overhead control:

```bash
DISABLE_TRACEML=1 RUN_NAME=clip-ddp-4l4-native \
  ./src/dev/repro/clip_ddp_l4/launch.sh
```

The TraceML run writes:

```text
logs/clip-ddp-4l4-baseline/final_summary.json
logs/clip-ddp-4l4-baseline/final_summary.txt
```

See [Controlled experiment sequence](#controlled-experiment-sequence) for the
input, compilation, attention, worker, and rank-straggler comparisons.

## Repository location

- `train_clip_ddp.py` contains the workload and experiment controls.
- `launch.sh` provides portable 4-GPU and 8-GPU commands.
- This README records the reproducible experiment protocol.

The workload lives under `src/dev/repro` because it is an experimental
reproduction rather than a supported end-user example. The directory can
still be linked directly from a technical report or blog post.

## Core setup retained from the issue

- `CLIPConfig` plus a randomly initialized `CLIPModel`
- `CLIPProcessor(use_fast=False)` for raw image/text batches
- Target global batch size 1024 and per-device batch size 128
- Learning rate `1e-3`, weight decay `0.01`, max gradient norm `4.0`
- Fused AdamW, cosine schedule, BF16, and TF32
- Configurable DataLoader workers, pinned memory, prefetching, persistent
  workers, and dropped incomplete batches
- DDP with NCCL, 25 MB buckets, no unused-parameter search, and no broadcast
  buffers
- `torch.compile=True` and `flash_attention_2`
- `OMP_NUM_THREADS=1` and `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1`

## Adaptations in this reproduction

- The default run uses 300 diagnostic steps for a shorter experiment.
- Checkpointing and W&B are disabled by default for the diagnostic run.
- The collator explicitly requests the CLIP contrastive training loss expected
  by `Trainer`.
- Gradient accumulation is calculated from world size to preserve the target
  global batch of 1024 on both 4-GPU and 8-GPU runs.
- Generated, preprocessed, ImageFolder, named Hugging Face, and local JSONL
  dataset sources are supported through a common interface.
- Worker count, prefetch factor, persistent workers, pinned memory, batch size,
  attention backend, compilation, and rank delay are configurable.

## Environment and prerequisites

For the issue-matching attention configuration, use a Flash Attention build
compatible with the host's PyTorch and CUDA versions. If it is not available,
begin with `ATTENTION_IMPL=sdpa` and record the selected backend with the
result.

Before collecting results, record `nvidia-smi -q`, GPU topology, CPU count,
storage type, PyTorch, CUDA, Transformers, Accelerate, Datasets, and TraceML
versions. The script prints the software versions into the run log.

## Dataset modes

### `generated` (default)

Creates deterministic PIL images and captions in each DataLoader worker, then
runs the same `CLIPProcessor` path as the issue. It requires no dataset
download and exercises CPU preprocessing, but not file decoding or storage.

### `preprocessed`

Returns ready-to-batch `pixel_values`, `input_ids`, and `attention_mask` CPU
tensors. Compare this with `generated` or real images to determine whether the
raw input pipeline is responsible for lost time.

### `imagefolder`

Uses the Hugging Face `imagefolder` loader. Point `DATA_PATH` at a directory
containing images and `metadata.jsonl` with `file_name` and `text` fields:

```bash
DATASET_SOURCE=imagefolder DATA_PATH=/datasets/clip-images \
  ./src/dev/repro/clip_ddp_l4/launch.sh \
  --data-path=/datasets/clip-images
```

### `hf`

Loads a named, non-streaming Hugging Face dataset. Pass the dataset name and
the image/text columns as forwarded training arguments:

```bash
DATASET_SOURCE=hf ./src/dev/repro/clip_ddp_l4/launch.sh \
  --dataset-name=<organization/dataset> \
  --dataset-split=train \
  --image-column=image \
  --text-column=text
```

### `jsonl`

Loads a local CC/LAION-like JSONL manifest. Each record must contain an image
path under `image`, `jpg`, `file_name`, or `path`, plus `text`, `txt`, or the
conversation shape from the issue. Relative image paths resolve under
`--data-path`.

```bash
DATASET_SOURCE=jsonl ./src/dev/repro/clip_ddp_l4/launch.sh \
  --data-path=/datasets/cc12m/images \
  --metadata-file=/datasets/cc12m/metadata.jsonl
```

## First 4×L4 run

The default launcher targets four GPUs and the issue's main configuration:

```bash
./src/dev/repro/clip_ddp_l4/launch.sh
```

If batch size 128 does not fit a 24 GB L4 with the installed software stack,
record the OOM and retry with 64. The global batch remains 1024 through
automatic gradient accumulation:

```bash
PER_DEVICE_BATCH_SIZE=64 ./src/dev/repro/clip_ddp_l4/launch.sh
```

Do not compare a 64-sample run with a 128-sample run without calling out the
batch-size change.

## Later 8×A100 run

Changing only the process count makes gradient accumulation one, matching the
issue:

```bash
NPROC_PER_NODE=8 RUN_NAME=clip-ddp-8a100-baseline \
  ./src/dev/repro/clip_ddp_l4/launch.sh
```

## Controlled experiment sequence

Use unique run names and change one factor at a time.

1. **Launcher baseline without telemetry**

   ```bash
   DISABLE_TRACEML=1 RUN_NAME=clip-ddp-native \
     ./src/dev/repro/clip_ddp_l4/launch.sh
   ```

2. **Matching TraceML run**

   ```bash
   RUN_NAME=clip-ddp-traceml \
     ./src/dev/repro/clip_ddp_l4/launch.sh
   ```

3. **Preprocessed-input control**

   ```bash
   DATASET_SOURCE=preprocessed RUN_NAME=clip-ddp-preprocessed \
     ./src/dev/repro/clip_ddp_l4/launch.sh
   ```

4. **Compile control**

   ```bash
   TORCH_COMPILE=0 RUN_NAME=clip-ddp-no-compile \
     ./src/dev/repro/clip_ddp_l4/launch.sh
   ```

5. **Attention control**

   ```bash
   ATTENTION_IMPL=sdpa RUN_NAME=clip-ddp-sdpa \
     ./src/dev/repro/clip_ddp_l4/launch.sh
   ```

6. **Worker sweep**

   Run otherwise identical configurations with `NUM_WORKERS=0`, `2`, `4`,
   and `8`. Persistent workers and prefetching are disabled automatically at
   zero workers.

7. **Known rank-straggler positive control**

   ```bash
   RUN_NAME=clip-ddp-rank0-delay \
     ./src/dev/repro/clip_ddp_l4/launch.sh \
     --rank-delay-ms=100 --delayed-rank=0
   ```

Run meaningful configurations at least three times. Exclude initialization,
data download, model download, and the `torch.compile` warmup from comparisons.
The first compile can be much longer than steady state.

## Evidence and interpretation

TraceML writes its report under:

```text
logs/<run-name>/final_summary.json
logs/<run-name>/final_summary.txt
```

For each run, retain independent wall-clock throughput and an external GPU
utilization series. Confirm the most important diagnosis in a separate short
`torch.profiler` or Nsight Systems run; do not run a heavyweight profiler
during the TraceML overhead comparison.

A result intended for publication should be repeatable, independently
corroborated, and improve when the diagnosed factor is removed. Report the
scope as:

> Inspired by an 8×A100 report, we investigated the same symptom class on a
> reproducible 4×L4 CLIP/DDP workload.

The 4×L4 experiment evaluates the same symptom class in a smaller environment;
the later 8×A100 run can be reported as a separate validation.
