# Case studies

Reproducible training investigations using TraceML measurements. Each study
records its workload, environment, measurement method, result and limits.

## Index

| Case study | Workload | Finding | Result |
|---|---|---|---|
| [ResNet-18 input pipeline](resnet18_input_bound/) | ResNet-18, single T4 | Synchronous image loading left the GPU idle | 43.8% lower step time after fixing the input pipeline |
| [RF-DETR Nano training](rfdetr_nano_training/) | RF-DETR Nano with COCO, one and four T4s | Input loading kept up; DDP added time mainly in backward | 60.5 images/s on four T4s; 3.26x throughput scaling |

## Adding a case study

A case study should state the question, record the exact workload and
environment, explain the measurement boundaries, and report the result with its
limits. When it evaluates a change, hold unrelated settings constant and include
the before/after wall-clock measurement.

Keep datasets and raw telemetry out of git. Commit the reproduction code and the
small result needed to support the write-up.
