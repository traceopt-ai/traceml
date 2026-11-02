# Export Records

This page is the contract for TraceML's structured export records.
An export record is one row of a projection table the aggregator writes to SQLite, given names and types.
The records live in `traceml_ai.aggregator.export`.

## What a record is

Each projection table the aggregator writes to SQLite has one frozen dataclass.
Its fields are the table's columns in table order, minus the autoincrement `id`.
The record is built by the same projection code that builds the SQLite row, so the two cannot disagree about a value.

| Record | `STREAM` | SQLite table |
|---|---|---|
| `StepTimeRecord` | `step_time` | `step_time_samples` |
| `StepMemoryRecord` | `step_memory` | `step_memory_samples` |
| `SystemRecord` | `system` | `system_samples` |
| `SystemGpuRecord` | `system_gpu` | `system_gpu_samples` |
| `ProcessRecord` | `process` | `process_samples` |
| `RuntimeEnvironmentRecord` | `runtime_environment` | `runtime_environment` |
| `StdoutStderrRecord` | `stdout_stderr` | `stdout_stderr_samples` |

Every record class carries `STREAM`, `SCHEMA_VERSION` (currently `1`) and `COLUMNS`, its field names in order.
`to_dict()` returns `stream`, `schema_version` and then every field.
A missing value is `None`, never `0`.
`StepTimeRecord` is the one record that differs from its row.
It carries the decoded `events` map in place of the `events_json` text.

`records_from_payload(payload, *, observed_timestamp_unix_ns=None)` takes one admitted telemetry payload, a dict or a list of dicts.
It returns a `RecordBatch` of `records` and `skipped`, and it never raises.
When `observed_timestamp_unix_ns` is given, every envelope in the payload gets that value.
Otherwise each envelope is stamped with `time.time_ns()` when its records are built.

`skipped` counts three kinds of unit, not rows.

- An item that does not normalise to a telemetry envelope counts once.
  So does an item that fails in any other way.
- A projection that raises counts once.
  That envelope loses all of its rows for the tables that projection writes, however many there would have been.
- A row that does not fit its record counts once.

A table that a projection writes but that has no record is skipped, and each of its rows counts once.
No projection table lacks a record today, and a test keeps it that way.

## Common fields

Most records start with the same identity and time fields.

| Field | Type | Unit | `None` means |
|---|---|---|---|
| `observed_timestamp_unix_ns` | int | ns since Unix epoch | never `None`; when the record was built, once per envelope |
| `rank` | int | | legacy alias of `global_rank`, same value |
| `global_rank` | int | | the rank sent no usable global rank |
| `local_rank` | int | | the rank sent no local rank |
| `world_size` | int | | the rank sent no world size |
| `local_world_size` | int | | the rank sent no local world size |
| `node_rank` | int | | the rank sent no node rank |
| `hostname` | str | | the rank sent no hostname |
| `timestamp_unix_ns` | int | ns since Unix epoch | the row had no integer source time, for example a rank older than integer timestamps |
| `seq` | int | | the row had no sample counter |

`timestamp_unix_ns` is when the rank measured the row.
`observed_timestamp_unix_ns` is stamped when the record is built, once per envelope.
The SQLite writer stamps its rows the same way.
The two stamps are taken separately, so the export value and the SQLite value for one row differ by the drain latency.
`seq` is the sampler's tick counter on that rank, so it can repeat within a rank.

## Ordering and duplicates

Records come out in payload order.
Within one rank, step records arrive in step order.
Records from different ranks interleave.
The key of a step record is `(global_rank, step)`.
Duplicate rows are passed through, not removed, so a consumer that needs one row per step keys on `(global_rank, step)`.
Do not key on `seq`.

## Records

### StepTimeRecord

| Field | Type | Unit | `None` means |
|---|---|---|---|
| `observed_timestamp_unix_ns` | int | ns | see common fields |
| `rank` | int | | see common fields |
| `global_rank` | int | | see common fields |
| `local_rank` | int | | see common fields |
| `world_size` | int | | see common fields |
| `local_world_size` | int | | see common fields |
| `node_rank` | int | | see common fields |
| `hostname` | str | | see common fields |
| `timestamp_unix_ns` | int | ns | see common fields |
| `seq` | int | | see common fields |
| `step` | int | | the row had no integer step |
| `events` | map | | never `None`; empty map when the row had no events |

`events[name][device]` is a map of `is_gpu` (bool), `duration_ms`, `cpu_ms`, `gpu_ms` (float, milliseconds) and `n_calls` (int).
Each value is `None` when the rank did not send it.
`cpu_ms` is CPU wall time and `gpu_ms` is CUDA event time, each summed over the step's calls.
`gpu_ms` is `None` when no GPU timing was recorded.
`duration_ms` is the legacy effective duration, and `is_gpu` says whether GPU time backs it.

### StepMemoryRecord

| Field | Type | Unit | `None` means |
|---|---|---|---|
| `observed_timestamp_unix_ns` | int | ns | see common fields |
| `rank` | int | | see common fields |
| `global_rank` | int | | see common fields |
| `local_rank` | int | | see common fields |
| `world_size` | int | | see common fields |
| `local_world_size` | int | | see common fields |
| `node_rank` | int | | see common fields |
| `hostname` | str | | see common fields |
| `timestamp_unix_ns` | int | ns | see common fields |
| `seq` | int | | see common fields |
| `device` | str | | the row named no device |
| `step` | int | | the row had no integer step |
| `peak_alloc_bytes` | float | bytes | not measured |
| `peak_reserved_bytes` | float | bytes | not measured |

### SystemRecord

| Field | Type | Unit | `None` means |
|---|---|---|---|
| `observed_timestamp_unix_ns` | int | ns | see common fields |
| `global_rank` | int | | see common fields |
| `local_rank` | int | | see common fields |
| `world_size` | int | | see common fields |
| `local_world_size` | int | | see common fields |
| `node_rank` | int | | see common fields |
| `hostname` | str | | see common fields |
| `timestamp_unix_ns` | int | ns | see common fields |
| `seq` | int | | see common fields |
| `cpu_percent` | float | percent of the host, 0 to 100 | not measured |
| `ram_used_bytes` | float | bytes | not measured |
| `ram_total_bytes` | float | bytes | not measured |
| `gpu_available` | int | 1 or 0 | not reported |
| `gpu_count` | int | | not reported |

### SystemGpuRecord

One record per GPU in a system snapshot.

| Field | Type | Unit | `None` means |
|---|---|---|---|
| `observed_timestamp_unix_ns` | int | ns | see common fields |
| `global_rank` | int | | see common fields |
| `local_rank` | int | | see common fields |
| `world_size` | int | | see common fields |
| `local_world_size` | int | | see common fields |
| `node_rank` | int | | see common fields |
| `hostname` | str | | see common fields |
| `timestamp_unix_ns` | int | ns | see common fields |
| `seq` | int | | see common fields |
| `gpu_idx` | int | | never `None`; GPU index on the host |
| `util` | float | percent, 0 to 100 | not measured |
| `mem_used_bytes` | float | bytes | not measured |
| `mem_total_bytes` | float | bytes | not measured |
| `temperature_c` | float | degrees Celsius | not measured |
| `power_usage_w` | float | watts | not measured |
| `power_limit_w` | float | watts | not measured |

### ProcessRecord

| Field | Type | Unit | `None` means |
|---|---|---|---|
| `observed_timestamp_unix_ns` | int | ns | see common fields |
| `rank` | int | | see common fields |
| `global_rank` | int | | see common fields |
| `local_rank` | int | | see common fields |
| `world_size` | int | | see common fields |
| `local_world_size` | int | | see common fields |
| `node_rank` | int | | see common fields |
| `hostname` | str | | see common fields |
| `timestamp_unix_ns` | int | ns | see common fields |
| `seq` | int | | see common fields |
| `cpu_percent` | float | percent of one core, can exceed 100 | not measured |
| `cpu_logical_core_count` | int | | not reported |
| `ram_used_bytes` | float | bytes, resident set size | not measured |
| `ram_total_bytes` | float | bytes | not measured |
| `gpu_available` | int | 1 or 0 | not reported |
| `gpu_count` | int | | not reported |
| `gpu_device_index` | int | | the process reported no GPU |
| `gpu_mem_used_bytes` | float | bytes | not measured |
| `gpu_mem_reserved_bytes` | float | bytes | not measured |
| `gpu_mem_total_bytes` | float | bytes | not measured |

### RuntimeEnvironmentRecord

| Field | Type | Unit | `None` means |
|---|---|---|---|
| `observed_timestamp_unix_ns` | int | ns | see common fields |
| `rank` | int | | see common fields |
| `global_rank` | int | | see common fields |
| `local_rank` | int | | see common fields |
| `world_size` | int | | see common fields |
| `local_world_size` | int | | see common fields |
| `node_rank` | int | | see common fields |
| `hostname` | str | | see common fields |
| `pid` | int | | the rank sent no process id |
| `timestamp_unix_ns` | int | ns | see common fields |
| `seq` | int | | see common fields |
| `topology` | str | | not reported |
| `distributed_initialized` | int | 1 or 0 | not reported |
| `distributed_backend` | str | | not reported |
| `training_strategy` | str | | not reported |
| `strategy_source` | str | | not reported |
| `strategy_confidence` | str | | not reported |

### StdoutStderrRecord

| Field | Type | Unit | `None` means |
|---|---|---|---|
| `observed_timestamp_unix_ns` | int | ns | see common fields |
| `rank` | int | | the payload had no global rank |
| `timestamp_unix_ns` | int | ns | see common fields |
| `line` | str | | never `None`; empty lines are not recorded |

## A custom consumer

```python
from traceml_ai.aggregator.export import StepTimeRecord, records_from_payload

FORWARD = "_traceml_internal:forward_time"


def forward_cpu_ms(payload):
    """Yield (global_rank, step, forward CPU ms) for each step record."""
    for record in records_from_payload(payload).records:
        if isinstance(record, StepTimeRecord):
            for stats in record.events.get(FORWARD, {}).values():
                yield record.global_rank, record.step, stats.get("cpu_ms")
```
