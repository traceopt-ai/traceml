# ResNet-18, input-bound dataloader (single T4)

TraceML classified a from-scratch ResNet-18 training run as input-bound. After
only the input pipeline was changed, wall-clock step time decreased by 43.8%,
from 315.5 ms to 177.2 ms, and the verdict changed to compute-bound.

## Setup

- Hardware: 1x NVIDIA Tesla T4, single GPU, no DDP.
- Software: PyTorch 2.11, torchvision 0.26, CUDA 13, traceml-ai 0.3.2.
- Model: torchvision ResNet-18 from scratch (`weights=None`).
- Data: fast.ai Imagenette, full resolution, `ImageFolder` + `RandomResizedCrop(224)`.
- Config: batch 64, 2,000 optimizer steps, seed 42, AMP off.

Model, batch size, step count, and AMP are held constant across both runs. The
input-pipeline settings listed below are the only changes.

## What TraceML found

On the baseline run (`num_workers=0`), TraceML returned **INPUT-BOUND**. Median
GPU utilization reported by nvidia-smi was 51% while the training process decoded
JPEGs synchronously, leaving the GPU idle between steps.

## The fix

The optimized run changed these input-pipeline settings:

| Setting | Baseline | Optimized | What it does |
|---|---|---|---|
| `num_workers` | 0 | 4 | Background subprocesses decode upcoming batches in parallel with GPU compute. |
| `pin_memory` | False | True | Stores batches in page-locked host memory, enabling asynchronous H2D copies. |
| `persistent_workers` | False | True | Keeps workers alive across epochs instead of re-forking each one. |
| H2D copy `non_blocking` | False | True | Requests asynchronous H2D copies from pinned host memory. |

## Result (before to after)

| Metric | Baseline | Optimized | Change |
|---|---|---|---|
| Step cadence (wall clock) | 315.5 ms | 177.2 ms | **-43.8%** |
| Run duration (2,000 steps) | 633.4 s | 358.2 s | **-43.4%** |
| GPU utilization (nvidia-smi median) | 51% | 100% | +49 percentage points |
| TraceML verdict | INPUT-BOUND | COMPUTE-BOUND | limiting phase changed |

In the original measurement, step cadence was derived from telemetry receipt
timestamps and cross-checked against run duration; GPU utilization came from an
independent nvidia-smi sample. The lower wall-clock time and verdict change
indicate that input was no longer the limiting phase.

These numbers were measured on traceml-ai 0.3.2 and have not been re-run on
later releases. They apply to the hardware and software configuration recorded
above.

## Reproduce

The [`data_loading_bottleneck.ipynb`](../../../notebooks/data_loading_bottleneck.ipynb)
notebook runs a 300-step version of the same comparison in Colab. Its result
depends on the assigned CPU and GPU. This case study records the original
2,000-step run on a single T4.
