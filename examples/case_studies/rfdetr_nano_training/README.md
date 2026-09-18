# RF-DETR Nano training on a single T4

This case study measures where time goes in eager RF-DETR Nano training on a
Tesla T4 with real COCO train2017 batches. It contributes the T4 baseline and
step attribution requested in
[RF-DETR issue #1410](https://github.com/roboflow/rf-detr/issues/1410).

The run sustained about **18.5 images/s**. Input waiting averaged only
**0.23 ms/step**, so two DataLoader workers were already keeping up. Backward
was the largest measured phase, while criterion and matcher also accounted for
meaningful work.

## Results

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

## Setup

| | Configuration |
|---|---|
| GPU | Tesla T4, 16 GB; driver 595.71.05 |
| CPU | Intel Xeon Platinum 8259CL; 24 physical cores |
| RF-DETR | [`0ed5be8`](https://github.com/roboflow/rf-detr/tree/0ed5be8e8d6762c4978a11671cbf34cfc0595e25) |
| TraceML | `622399cccc67f5c81b44096a3c47169134bd13a2` |
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
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2

for repeat in 1 2 3; do
  modes="baseline traced"
  if [ "$repeat" = 2 ]; then modes="traced baseline"; fi
  for mode in $modes; do
    disable=()
    if [ "$mode" = baseline ]; then disable=(--disable-traceml); fi
    traceml run "$CASE/train.py" --mode summary --nproc-per-node 1 \
      --logs-dir "$BATCH/telemetry" --run-name "pair-$repeat-$mode" \
      "${disable[@]}" --args \
      --dataset-dir "$DATASET" --output-dir "$BATCH/pair-$repeat-$mode"
  done
done
```

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

This is an eager-training baseline on one T4. It does not evaluate
`torch.compile`, CUDA graphs, DDP or model accuracy, and it makes no claim about
those configurations.
