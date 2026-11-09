# Case studies

These measured investigations show how TraceML observations relate to
wall-clock training behavior. The complete write-ups and reproduction code stay
with the executable examples in the TraceML repository.

## Measured investigations

### LeRobot v3 image-loading regression

Across two broken/fixed pairs run in opposite orders on one NVIDIA L4, the
TraceML diagnosis changed from input-bound to compute-bound. Step time fell by
**92.2%** and **92.4%**, predominantly because input wait disappeared after the
upstream fix.

[Read the complete case study and reproduce it](https://github.com/traceopt-ai/traceml/tree/main/examples/case_studies/lerobot_v3_image_regression)

### ResNet-18 input pipeline

A single-T4 run was input-bound while JPEG decoding happened synchronously.
Changing only the input pipeline reduced wall-clock step time by **43.8%** and
changed the TraceML verdict from input-bound to compute-bound.

[Read the complete case study and reproduce it](https://github.com/traceopt-ai/traceml/tree/main/examples/case_studies/resnet18_input_bound)

### RF-DETR Nano training

On real COCO batches, one T4 sustained about **18.6 images/s** and four-T4 DDP
sustained about **60.5 images/s**, or **3.26x** weak scaling. TraceML showed that
input waiting remained small while backward time increased under DDP.

[Read the complete case study and reproduce it](https://github.com/traceopt-ai/traceml/tree/main/examples/case_studies/rfdetr_nano_training)

### RF-DETR non-JPEG release regression

RF-DETR 1.11.0 increased median native step time by **14.78%** on a controlled
large-PNG workload. TraceML observed a **23.74%** increase in input wait while
measured GPU compute became faster. RF-DETR 1.11.1 restored step time and input
wait to within 0.5% of the 1.10.1 baseline.

[Read the complete case study and reproduce it](https://github.com/traceopt-ai/traceml/tree/main/examples/case_studies/rfdetr_input_pipeline_regression)
