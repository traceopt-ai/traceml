# TraceML

**Diagnose slow PyTorch training with zero-code instrumentation. Catch
regressions in CI.**

TraceML shows where each training step goes—input loading, data transfer,
forward, backward, and optimizer work—then identifies the bottleneck and saves
evidence you can compare locally or check in CI.

**Works automatically with:**
[Hugging Face Trainer](user_guide/integrations/huggingface.md) ·
[PyTorch Lightning](user_guide/integrations/lightning.md) ·
[RF-DETR](user_guide/integrations/rfdetr.md)

```bash
pip install traceml-ai
traceml run train.py
```

For these standard trainers, no TraceML code is required in the training
script. Plain PyTorch loops and other stacks use a small
[explicit integration](user_guide/integrations.md).

A completed run ends with a diagnosis, its evidence, and the next place to
investigate:

```text
Verdict: INPUT-BOUND
Why: Input Wait took 64% of Step Time.
Next: Increase workers, prefetch, or storage throughput.
```

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Quickstart**

    ---

    Run a supported trainer without changing its script, or instrument a
    custom PyTorch loop.

    [:octicons-arrow-right-24: Get started](user_guide/quickstart.md)

-   :material-file-search-outline:{ .lg .middle } **Reading output**

    ---

    Understand the terminal card, the dashboard, and the fields in
    `final_summary.json`.

    [:octicons-arrow-right-24: Read the guide](user_guide/reading-output.md)

-   :material-puzzle-outline:{ .lg .middle } **Integrations**

    ---

    Choose the supported setup for Hugging Face, Lightning, RF-DETR,
    Accelerate, MONAI, Ray, DeepSpeed, W&B, or MLflow.

    [:octicons-arrow-right-24: See integrations](user_guide/integrations.md)

-   :material-scale-balance:{ .lg .middle } **Compare runs**

    ---

    See what changed between two runs and optionally fail CI on a Step Time
    regression.

    [:octicons-arrow-right-24: Compare runs](user_guide/compare.md)

-   :material-code-braces:{ .lg .middle } **Public API**

    ---

    The full `traceml_ai` reference: `init()`, `trace_step()`, and
    `summary()`.

    [:octicons-arrow-right-24: API reference](user_guide/public-api.md)

-   :material-source-branch:{ .lg .middle } **Developer guide**

    ---

    Architecture, pipeline contracts, and how to contribute to TraceML.

    [:octicons-arrow-right-24: Contribute](developer_guide/contributing.md)

</div>

---

TraceML is open source under the Apache 2.0 license. Learn more at
[traceopt.ai](https://traceopt.ai) or star the project on
[GitHub](https://github.com/traceopt-ai/traceml).
