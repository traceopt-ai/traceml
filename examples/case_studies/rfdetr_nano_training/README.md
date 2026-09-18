# RF-DETR Nano training on T4 GPUs

This case study measures where time goes in eager RF-DETR Nano training on a
single Tesla T4 and with four-T4 DDP, using real COCO train2017 batches. It
contributes the T4 baseline and step attribution requested in
[RF-DETR issue #1410](https://github.com/roboflow/rf-detr/issues/1410).

One T4 sustained about **18.5 images/s**; four-T4 DDP sustained about
**60.5 images/s**, a **3.26x** throughput increase with **81.5% weak-scaling
efficiency** at a fixed batch of four per GPU. The global batch grows from 4 to
16. Mean input waiting remained below
0.25 ms/step. Backward was the largest measured phase and increased under DDP,
consistent with synchronization and gradient communication occurring in that
region.

## Results

### Single GPU

Three independent native runs used 50 optimizer steps, discarded steps 1–10,
and measured steps 11–50.

| Run | Wall time/step | Images/s |
|---:|---:|---:|
| 1 | 215.5 ms | 18.56 |
| 2 | 214.5 ms | 18.65 |
| 3 | 217.6 ms | 18.38 |

TraceML measured the same 40-step window in three matched runs:

| Region | Mean time/step |
|---|---:|
| Step envelope | 221.3 ms |
| Forward | 44.3 ms |
| Backward | 79.3 ms |
| Optimizer region | 56.2 ms |
| Residual | 39.7 ms |
| Input wait | 0.23 ms |
| Host-to-device | 1.57 ms |

The optimizer region includes scheduler and EMA callback work. Residual is
unassigned time rather than criterion time. Host-to-device work can overlap
other regions and should not be added to the total.

A separate PyTorch Profiler run used `wait=20, warmup=5, active=10`:

| Scope | CPU duration/step | Attributed CUDA time/step |
|---|---:|---:|
| Criterion, including matcher | 37.0 ms | 32.1 ms |
| Matcher | 20.4 ms | 15.8 ms |

Matcher is nested inside criterion, so these rows and the CPU/CUDA columns are
not additive. RF-DETR reported that Triton linear assignment was unavailable
for this CUDA input and used its SciPy fallback.

TraceML added a median 5.6 ms/step (2.55%) across the three matched
native/traced pairs.

### Four-GPU DDP

The DDP runs used the same model and per-GPU batch size, for a global batch of
16. Each row is an independent native run.

| Run | Wall time/step | Images/s |
|---:|---:|---:|
| 1 | 263.7 ms | 60.67 |
| 2 | 272.1 ms | 58.80 |
| 3 | 264.3 ms | 60.53 |

Mean phase times across all ranks and traced runs:

| Region | Single T4 | Four-T4 DDP |
|---|---:|---:|
| Step envelope | 221.3 ms | 269.6 ms |
| Forward | 44.3 ms | 46.7 ms |
| Backward | 79.3 ms | 114.7 ms |
| Optimizer region | 56.2 ms | 56.6 ms |
| Residual | 39.7 ms | 49.7 ms |
| Input wait | 0.23 ms | 0.24 ms |
| Host-to-device | 1.57 ms | 1.63 ms |

Median native throughput increased from 18.56 to 60.53 images/s. The four-rank
step envelopes stayed closely aligned: the largest difference between ranks in
any traced run was 1.6 ms. The main phase-level change was backward, which rose
by about 35 ms/step. These timings locate the added cost within the backward
region but do not isolate individual NCCL operations.

TraceML's median measured overhead in the DDP pairs was 2.70%; the individual
pairs ranged from -1.88% to +3.63%, indicating run-to-run noise at this sample
size.

## Setup

| | Configuration |
|---|---|
| GPU | Tesla T4, 16 GB; driver 595.71.05 |
| CPU | Intel Xeon Platinum 8259CL; 24 physical cores |
| RF-DETR | [`0ed5be8`](https://github.com/roboflow/rf-detr/tree/0ed5be8e8d6762c4978a11671cbf34cfc0595e25) |
| TraceML | `622399c` (single GPU); `8549e91` (DDP) |
| PyTorch | 2.9.1+cu128 |
| Model | RF-DETR Nano, pretrained weights, resolution 384 |
| Dataset | COCO train2017 |
| Training | FP16, batch 4, two workers, seed 42 |
| Execution | Eager, fixed resolution, torchvision augmentation |

Validation, external loggers and progress bars were disabled. Native RF-DETR
optimizer, scheduler and EMA behavior was preserved. Fixed resolution disables
the native multi-scale and expanded-scale settings for this measurement.

Throughput is CUDA-synchronized wall time across the complete measurement
window. TraceML records asynchronous CUDA events for GPU regions and resolves
them later without synchronizing the training loop; input wait uses host time.
Criterion and matcher attribution comes from the separate PyTorch Profiler run.
The TraceML revisions differ only in documentation and integration messaging;
the training and timing implementation is unchanged.

## Reproduce

Run from the TraceML repository root on Linux with Python 3.11 and a
CUDA 12.8-compatible driver:

```bash
python3.11 -m venv data/rfdetr-case-venv
source data/rfdetr-case-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[lightning]'
python -m pip install \
  'rfdetr[train] @ git+https://github.com/roboflow/rf-detr.git@0ed5be8e8d6762c4978a11671cbf34cfc0595e25'
python -m pip check
```

Download and extract COCO 2017:

```bash
mkdir -p data/coco2017
curl -fL --retry 3 http://images.cocodataset.org/zips/train2017.zip \
  -o data/coco2017/train2017.zip
curl -fL --retry 3 http://images.cocodataset.org/zips/val2017.zip \
  -o data/coco2017/val2017.zip
curl -fL --retry 3 \
  http://images.cocodataset.org/annotations/annotations_trainval2017.zip \
  -o data/coco2017/annotations_trainval2017.zip
unzip -q data/coco2017/train2017.zip -d data/coco2017
unzip -q data/coco2017/val2017.zip -d data/coco2017
unzip -q data/coco2017/annotations_trainval2017.zip -d data/coco2017
```

Run three alternating native/traced pairs:

```bash
set -e
CASE=examples/case_studies/rfdetr_nano_training
DATASET="$PWD/data/coco2017"
BATCH="$PWD/logs/rfdetr-nano/t4-01"
NPROC=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2

for repeat in 1 2 3; do
  modes="baseline traced"
  if [ "$repeat" = 2 ]; then modes="traced baseline"; fi
  for mode in $modes; do
    disable=()
    if [ "$mode" = baseline ]; then disable=(--disable-traceml); fi
    traceml run "$CASE/train.py" --mode summary --nproc-per-node "$NPROC" \
      --logs-dir "$BATCH/telemetry" --run-name "pair-$repeat-$mode" \
      "${disable[@]}" --args \
      --dataset-dir "$DATASET" --output-dir "$BATCH/pair-$repeat-$mode"
  done
done
```

For four-GPU DDP, repeat the paired loop with a new output directory and four
processes:

```bash
BATCH="$PWD/logs/rfdetr-nano/t4-ddp-01"
NPROC=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

The batch size remains four per rank, giving a global batch of 16. Generate the
DDP report with the same `summarize.py` command and omit `--profile-dir`.

Run the profiler separately, then generate the report:

```bash
traceml run "$CASE/train.py" --disable-traceml --nproc-per-node 1 --args \
  --dataset-dir "$DATASET" --output-dir "$BATCH/profiler" --profile

python "$CASE/summarize.py" \
  --pair "$BATCH/pair-1-baseline" "$BATCH/pair-1-traced" \
  --pair "$BATCH/pair-2-baseline" "$BATCH/pair-2-traced" \
  --pair "$BATCH/pair-3-baseline" "$BATCH/pair-3-traced" \
  --profile-dir "$BATCH/profiler" \
  --output "$BATCH/issue-1410-single-t4.md"
```

`run.json` records the resolved configuration, source revisions, environment,
dataset annotation checksums and checkpoint checksum. Raw telemetry and profiler
traces remain in the selected `BATCH` directory.

## Scope

This is an eager-training baseline on one node with one or four T4 GPUs. It does
not evaluate `torch.compile`, CUDA graphs, multi-node training or model accuracy,
and it makes no claim about those configurations.
