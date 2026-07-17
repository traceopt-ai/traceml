# ResNet-18, input-bound dataloader (single T4)

TraceML flagged a from-scratch ResNet-18 training run as input-bound: the GPU was
starved waiting on data loading. Changing only the DataLoader (adding workers and
pinned-memory transfers) made every step 1.78x faster and flipped the verdict to
compute-bound.

## Setup

- Hardware: 1x NVIDIA Tesla T4, single GPU, no DDP.
- Software: PyTorch 2.11, torchvision 0.26, CUDA 13, traceml-ai 0.3.2.
- Model: torchvision ResNet-18 from scratch (`weights=None`).
- Data: fast.ai Imagenette, full resolution, `ImageFolder` + `RandomResizedCrop(224)`.
- Config: batch 64, 2,000 optimizer steps, seed 42, AMP off.

Model, batch, steps, and AMP are held constant across both runs, so the only
variable is data loading.

## What TraceML found

On the baseline run (`num_workers=0`), TraceML returned **INPUT-BOUND**. The GPU
sat around 51% utilization (nvidia-smi median) while the training process decoded
JPEGs synchronously, so the GPU waited on the CPU between steps.

## The fix

Change only the DataLoader:

| Setting | Baseline | Optimized | What it does |
|---|---|---|---|
| `num_workers` | 0 | 4 | Background subprocesses decode upcoming batches in parallel with GPU compute. |
| `pin_memory` | False | True | Page-locked host memory lets the host-to-GPU copy run as an async DMA. |
| `persistent_workers` | False | True | Keeps workers alive across epochs instead of re-forking each one. |
| H2D copy `non_blocking` | False | True | `.to(device)` returns immediately so the copy overlaps later work (needs pinned memory). |

## Result (before to after)

| Metric | Baseline | Optimized | Change |
|---|---|---|---|
| Step cadence (wall clock) | 315.5 ms | 177.2 ms | **1.78x faster** |
| Run duration (2,000 steps) | 633.4 s | 358.2 s | **-43.4%** |
| GPU utilization (nvidia-smi median) | 51% | 100% | no longer starving |
| TraceML verdict | INPUT-BOUND | COMPUTE-BOUND | bottleneck moved |

Numbers are wall-clock: step cadence is measured from telemetry receipt
timestamps and cross-checked against the run duration and an independent
nvidia-smi sample. The verdict flipping from input-bound to compute-bound is the
point: once the data pipeline keeps up, the GPU becomes the limit, which is where
you want to be.

## Reproduce

The [`data_loading_bottleneck.ipynb`](../../../notebooks/data_loading_bottleneck.ipynb)
notebook runs the same before/after (an input-bound baseline versus the
data-loading fix) on a free Colab T4 in a few minutes. This case study is the full
2,000-step version of that experiment on a single T4.
