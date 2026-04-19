# Runtime

Per-rank in-process agent. Runs samplers on a timer loop, executes the user script via `runpy`, and ships telemetry to the aggregator over TCP. Contains the sampler orchestration and crash-handling logic.

::: traceml.runtime
