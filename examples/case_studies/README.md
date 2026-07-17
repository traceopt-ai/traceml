# Case studies

Real-world before/after case studies where TraceML diagnosed a bottleneck in an
actual training run, a targeted fix was applied, and the improvement was verified
against the wall clock.

Each case study is a subfolder with its own write-up: the setup, what TraceML
found, the fix, and the measured before/after.

## Index

| Case study | Model | Bottleneck | Result |
|---|---|---|---|
| [resnet18_input_bound](resnet18_input_bound/) | ResNet-18 (single T4) | Input-bound dataloader | 1.78x faster steps, GPU util 51% to 100% |

## Adding a case study

1. Run a real training job under TraceML and confirm the bottleneck is genuine (a
   large phase share, plus a fix that actually moves the wall clock).
2. Apply one targeted fix, holding everything else constant so the before/after
   isolates a single change.
3. Write up the before/after using wall-clock metrics: step cadence, run duration,
   GPU utilization, and TraceML's verdict.
4. Keep raw telemetry out of git; commit the write-up and small summaries only.
