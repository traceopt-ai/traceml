---
hide:
  - navigation
---

# Find what's slowing down your PyTorch training { #traceml }

TraceML breaks down training step time into input wait, data transfer, forward,
backward, and optimizer work. If you use a [supported trainer](#choose-your-training-setup),
launch your existing script with `traceml run train.py`.

[Quickstart](user_guide/quickstart.md){ .md-button style="background: var(--md-default-fg-color); color: var(--md-default-bg-color); border-color: var(--md-default-fg-color); margin-inline-end: 0.6rem" }
[Read a diagnosis](user_guide/reading-output.md#start-with-the-diagnosis){ style="color: inherit; text-decoration: underline" }

## Example diagnosis

```text
Verdict: INPUT-BOUND (CRITICAL)
Why: Input Wait took 64% of Step Time.
Next: Increase workers, prefetch, or
      storage throughput.
```

Try a before-and-after DataLoader experiment with
[plain PyTorch](https://github.com/traceopt-ai/traceml/blob/main/notebooks/data_loading_bottleneck.ipynb)
or [Hugging Face Trainer](https://github.com/traceopt-ai/traceml/blob/main/notebooks/huggingface_dataloading_bottleneck.ipynb).

## Choose your training setup

The launcher instruments standard [Hugging Face Trainer](user_guide/integrations/huggingface.md),
[PyTorch Lightning](user_guide/integrations/lightning.md), and
[RF-DETR object detection](user_guide/integrations/rfdetr.md) training automatically.
For a [plain PyTorch loop](user_guide/quickstart.md#custom-pytorch-loop), mark the
step boundary in your script. See the [integration guides](user_guide/integrations.md)
for other setups.

## Investigate and compare

<div class="grid cards" markdown>

-   **Compare two runs**

    ---

    Compare a baseline with a candidate to see where step time changed. Use
    the [regression guard](user_guide/regression-guard.md) to check a threshold in CI.

    [:octicons-arrow-right-24: Compare runs](user_guide/compare.md)

-   **Training slowed even as GPU compute got faster**

    ---

    On a controlled PNG workload, median training steps took **14.8% longer**
    after upgrading RF-DETR from 1.10.1 to 1.11.0. TraceML measured **23.7%
    more input wait**. In 1.11.1, both measurements returned to within 0.5% of
    baseline.

    [:octicons-arrow-right-24: Read the case study](case-studies.md#rf-detr-non-jpeg-release-regression)

</div>

[Python API](user_guide/public-api.md) ·
[Contribute](developer_guide/contributing.md) ·
[GitHub](https://github.com/traceopt-ai/traceml)
