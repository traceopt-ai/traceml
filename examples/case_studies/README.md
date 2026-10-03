# Case studies

Measured write-ups and reproduction packages using TraceML. Each linked
investigation documents its workload, environment, methodology and limitations.

## Measured case studies

| Case study | Workload | Finding | Result |
|---|---|---|---|
| [ResNet-18 input pipeline](resnet18_input_bound/) | ResNet-18, single T4 | Synchronous image loading left the GPU idle | 43.8% lower step time after fixing the input pipeline |
| [RF-DETR Nano training](rfdetr_nano_training/) | RF-DETR Nano with COCO, one and four T4s | Input loading kept up; DDP added time mainly in backward | 60.5 images/s on four T4s; 3.26x weak scaling |
| [RF-DETR non-JPEG release regression](rfdetr_input_pipeline_regression/) | RF-DETR Nano, one NVIDIA L4 | RF-DETR 1.11.0 added input wait while GPU compute became faster | 14.8% slower steps; 1.11.1 restored the baseline |

## Reproduction packages

These packages provide pinned workloads and analysis tools but do not yet
publish a measured headline result.

| Reproduction | Workload | What it provides |
|---|---|---|
| [LeRobot dataset regression](lerobot_v3_image_regression/README.md) | ACT with LeRobot v3 image data | Pinned before/after runner and analysis notebook for an upstream image-loading regression |

## Adding a case study

A case study should state the question, record the exact workload and
environment, explain the measurement boundaries, and report the result with its
limits. When it evaluates a change, hold unrelated settings constant and include
the before/after wall-clock measurement.

Keep datasets and raw telemetry out of git. Commit the reproduction code and the
small result needed to support the write-up.
