# TraceML

**Real-time bottleneck finder for PyTorch training runs.**

TraceML instruments your existing training loop and tells you *why* it's
slow — input-bound, compute-bound, waiting on a distributed straggler, or
leaking memory — instead of leaving you to guess from raw CPU/GPU graphs. It
runs alongside your script, streams a live terminal or dashboard view during
the run, and writes a structured `final_summary.json` plus a human-readable
report at the end.

```bash
pip install traceml-ai
```

Already using Hugging Face, Lightning, Ray, DeepSpeed, W&B, or MLflow? See
[Integrations](user_guide/integrations.md) for framework-specific setup.

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Quickstart**

    ---

    Instrument your training step and get your first diagnosis in a few
    minutes.

    [:octicons-arrow-right-24: Get started](user_guide/quickstart.md)

-   :material-file-search-outline:{ .lg .middle } **Reading output**

    ---

    Understand the terminal card, the dashboard, and the fields in
    `final_summary.json`.

    [:octicons-arrow-right-24: Read the guide](user_guide/reading-output.md)

-   :material-puzzle-outline:{ .lg .middle } **Integrations**

    ---

    Use TraceML with Hugging Face, Accelerate, Lightning, Ray, DeepSpeed,
    W&B, or MLflow.

    [:octicons-arrow-right-24: See integrations](user_guide/integrations.md)

-   :material-scale-balance:{ .lg .middle } **Compare runs**

    ---

    Diff two `final_summary.json` files to see what changed between runs.

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
