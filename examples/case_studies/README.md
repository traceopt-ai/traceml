# Case studies

Measured before/after case studies where TraceML diagnosed a bottleneck in a
training run, a targeted intervention was applied, and the result was evaluated
using wall-clock measurements.

Each case study is a subfolder with its own write-up: the setup, what TraceML
found, the fix, and the measured before/after.

## Index

| Case study | Model | Bottleneck | Result |
|---|---|---|---|
| [resnet18_input_bound](resnet18_input_bound/) | ResNet-18 (single T4) | Input-bound data loading | 43.8% lower step time; median GPU utilization 51% to 100% |

## Adding a case study

1. Run a training job under TraceML and identify the bottleneck from phase timing.
2. Apply one targeted intervention, holding unrelated workload settings constant
   so the before/after comparison isolates that intervention.
3. Write up the before/after using wall-clock metrics: step cadence, run duration,
   GPU utilization, and TraceML's verdict.
4. Keep raw telemetry out of git; commit the write-up and small summaries only.
