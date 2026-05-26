# Architecture

![TraceML architecture overview](../assets/Architecture_excali_b.png)

TraceML runs as three cooperating processes during a training job. The CLI spawns an **aggregator** server and one or more **training** ranks via `torchrun`. Training ranks run user code in-process with TraceML hooks attached; telemetry is shipped over TCP to the aggregator, which renders the unified view.

## Telemetry data flow

```mermaid
flowchart LR
    subgraph "Per-rank training process"
        S[Sampler] -->|append| DB[In-memory Database]
        DB -->|new rows| Sender[DBIncrementalSender]
    end
    Sender -->|length-prefixed msgpack| TCP([TCP])
    TCP --> RS[RemoteDBStore]
    subgraph "Aggregator process"
        RS --> R[Renderer]
        R --> UI[CLI / NiceGUI driver]
    end
```

Samplers maintain an incremental append counter per rank per table. The sender ships only new rows. The aggregator's `RemoteDBStore` keeps each rank's data separate, and renderers pull read-only views from it.

## Layers

| Layer | Directory | Responsibility |
|---|---|---|
| CLI | `src/traceml_ai/launcher/` | Argument parsing, process spawning, signal handling |
| Runtime | `src/traceml_ai/runtime/` | In-process agent per rank; user-script executor |
| Aggregator | `src/traceml_ai/aggregator/` | TCP server, unified store, display orchestration |
| Samplers | `src/traceml_ai/samplers/` | Periodic telemetry collection (timing, memory, system) |
| Database | `src/traceml_ai/database/` | Bounded in-memory tables; rank-aware remote store |
| Transport | `src/traceml_ai/transport/` | TCP bidirectional + DDP rank detection |
| Renderers | `src/traceml_ai/renderers/` | Transform stored data into Rich/Plotly output |
| Display drivers | `src/traceml_ai/aggregator/display_drivers/` | CLI vs NiceGUI output medium |
| Public API | `src/traceml_ai/api.py` | Top-level instrumentation entry points |
| Integrations | `src/traceml_ai/integrations/` | Hugging Face, Lightning, and Ray adapters |
| Utils | `src/traceml_ai/utils/` | Hooks, patches, memory/timing helpers |

For the user-facing API surface (`trace_step`, `TraceMLTrainer`, `TraceMLCallback`, CLI usage), see the [Public API](../user_guide/public-api.md). The source tree above is the canonical reference for internals — start from the entry points and follow the imports.

## Design principles

- **Fail-open** — training must never crash because telemetry broke. Sampler/transport errors are logged, execution continues.
- **Bounded overhead** — every new sampler justifies its overhead. Deque-based bounded tables evict oldest records at fixed `maxlen`.
- **Process isolation** — no shared memory. TCP + env vars only.
- **Out-of-process UI** — aggregator crashes don't crash training.
