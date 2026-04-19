# Samplers

Periodic telemetry collectors: step timing, step memory, layer-level events, system metrics (CPU/GPU/RAM), process stats. Each sampler owns a bounded-deque table in the local `Database` and writes on a fixed interval (default 1s). The `DBIncrementalSender` ships only new rows to the aggregator.

::: traceml.samplers
