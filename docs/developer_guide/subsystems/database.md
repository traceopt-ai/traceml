# Database

Bounded append-only in-memory table store. Each table is a `collections.deque(maxlen=N)` — O(1) append with automatic eviction of oldest rows. The per-rank `Database` is written by samplers; the aggregator's `RemoteDBStore` lazily creates a `Database` per `(rank, sampler_name)` tuple as telemetry arrives.

::: traceml.database
