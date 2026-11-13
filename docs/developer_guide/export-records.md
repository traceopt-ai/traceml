# Export Records

This page is the contract for TraceML's structured export records.
An export record is one row of a projection table the aggregator writes to SQLite, given names and types.
The records live in `traceml_ai.aggregator.export`.
To switch export on and read the files, see [Live Export](../user_guide/live-export.md).

## What a record is

A record is a frozen dataclass for one projection table the aggregator writes to SQLite.
Its fields are the table's columns in table order, minus the autoincrement `id`.
The record uses the same projection code as the SQLite row, so their projected fields stay aligned.

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
The JSON Lines exporter omits `hostname` unless `exporter_hostname` is on.
When enabled, records that have a `hostname` field keep it, with `null` when the rank sent no hostname.
An unavailable value remains `None`; a measured zero remains `0`.
`StepTimeRecord` is the one record that differs from its row, in one field.
It carries the decoded `events` map in place of the `events_json` text.

`records_from_payload(payload, *, observed_timestamp_unix_ns=None, streams=None)` takes one admitted telemetry payload, a dict or a list of dicts.
It returns a `RecordBatch` of `records` and `skipped`.
A string, an empty collection or an unknown name in `streams` raises `ValueError` before payload processing.
When `observed_timestamp_unix_ns` is given, every envelope in the payload gets that value.
Otherwise each envelope is stamped with `time.time_ns()` when its records are built.
When `streams` is given, only records whose `STREAM` is in it are built.
The rows of every other stream are neither built nor counted in `skipped`, because leaving them out is not a loss.
`None` keeps every stream.

`skipped` counts three kinds of unit, not rows.

- An item that does not normalise to a telemetry envelope counts once.
  So does an item that fails in any other way.
- A projection that raises counts once.
  That envelope loses all of its rows for the tables that projection writes, however many there would have been.
- A row that does not fit its record counts once.

A table that a projection writes but that has no record is skipped, and each of its rows counts once.
Every table the projections write has a record, and a test keeps it that way.

## Common fields

Most records start with these identity and time fields, in this order.
`SystemRecord` and `SystemGpuRecord` have no `rank` field, because their rows have none.
`RuntimeEnvironmentRecord` has `pid` between `hostname` and `timestamp_unix_ns`.
`StdoutStderrRecord` has only `observed_timestamp_unix_ns`, `rank`, `timestamp_unix_ns` and `line`.

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
The two stamps are taken separately, so they are not guaranteed to be equal.
`seq` is the sampler's tick counter on that rank, so it can repeat within a rank.

## Ordering and duplicates

Records preserve payload order and are not sorted by this mapper.
Within one envelope, records follow the writer's table order, so a system snapshot gives its `SystemRecord` before its `SystemGpuRecord` entries.
The StepTime sampler emits steps in order within one rank.
Records from different ranks interleave.
Each stream has its own key, listed under [Stream selection](#stream-selection).
Duplicate rows are passed through, not removed, so a consumer that needs one row per key deduplicates on that key.
Do not key a step record on `seq`, because one sampler tick can carry several steps.

## Stream selection

The `export_streams` setting chooses which records reach the exporter.
The export worker applies it once, before any exporter sees a record.
So the `jsonl` exporter and a custom exporter receive the same selected records.
The default is `step_time`.

| Name | Records |
|---|---|
| `step_time` | `StepTimeRecord` |
| `step_memory` | `StepMemoryRecord` |
| `process` | `ProcessRecord` |
| `system` | `SystemRecord` and `SystemGpuRecord` |
| `all` | every record, including `RuntimeEnvironmentRecord` and `StdoutStderrRecord` |

The setting accepts only these five names, comma separated.
`RuntimeEnvironmentRecord` and `StdoutStderrRecord` are reachable only through `all`.
An unknown name, or a value that names no stream, raises `ValueError` when the aggregator starts with an exporter set.
The message lists the supported names.
`traceml run` and `traceml serve` check the value before launch and exit with the same message.

The fields vary by stream, and so does the key that identifies one row.

| `STREAM` | One record per | Key |
|---|---|---|
| `step_time` | step of a rank | `(global_rank, step)` |
| `step_memory` | step and device of a rank | `(global_rank, device, step)` |
| `system` | host snapshot of a node's local rank 0 | `(global_rank, seq)` |
| `system_gpu` | GPU in that host snapshot | `(global_rank, seq, gpu_idx)` |
| `process` | process snapshot of a rank | `(global_rank, seq)` |
| `runtime_environment` | rank process, at most once | `global_rank` |
| `stdout_stderr` | captured line of a rank | none; read the lines in file order |

A snapshot stream increments `seq` once per sampler tick and sends one snapshot per tick, so `seq` identifies a snapshot within a rank.
`gpu_idx` is the GPU's position in the snapshot's GPU list.

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
`gpu_ms` is `None` unless every call of that phase in the step measured the GPU clock; a measured zero stays `0.0`.
`duration_ms` is the legacy effective duration, and `is_gpu` says whether GPU time backs it.

### StepMemoryRecord

One record per step and device for one rank.
The two peaks are PyTorch's peak allocated and peak reserved memory on `device` during the step, in bytes.

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
| `peak_alloc_bytes` | float | bytes | the row had no peak |
| `peak_reserved_bytes` | float | bytes | the row had no peak |

### SystemRecord

One record per host-level system snapshot sent by a rank.
`cpu_percent` is host CPU utilization in percent.
`gpu_available` is an integer boolean from the row, `1` or `0`.

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
| `cpu_percent` | float | percent | not measured |
| `ram_used_bytes` | float | bytes | not measured |
| `ram_total_bytes` | float | bytes | not measured |
| `gpu_available` | int | 0 or 1 | not reported |
| `gpu_count` | int | | not reported |

`SystemRecord` has no `rank` field, because the `system_samples` row has none.
Its GPUs are separate `SystemGpuRecord` records from the same snapshot.

### SystemGpuRecord

One record per GPU in a system snapshot, so a snapshot with two GPUs gives two records.
`util` is GPU utilization in percent, `temperature_c` is in degrees C and the power fields are in watts.
Like `SystemRecord`, it has no `rank` field.

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
| `gpu_idx` | int | | never `None`; the GPU's position in the snapshot |
| `util` | float | percent | not measured |
| `mem_used_bytes` | float | bytes | not measured |
| `mem_total_bytes` | float | bytes | not measured |
| `temperature_c` | float | degrees C | not measured |
| `power_usage_w` | float | watts | not measured |
| `power_limit_w` | float | watts | not measured |

### ProcessRecord

One record per training-process snapshot for one rank.
`cpu_percent` is the process's CPU utilization in percent of one logical core, so it can exceed 100.
`ram_used_bytes` is the process's resident memory.
The `gpu_mem_*` fields are PyTorch's allocated, reserved and total memory on the process's device.
`gpu_available` is an integer boolean from the row, `1` or `0`.

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
| `cpu_percent` | float | percent | not measured |
| `cpu_logical_core_count` | int | | not reported |
| `ram_used_bytes` | float | bytes | not measured |
| `ram_total_bytes` | float | bytes | not measured |
| `gpu_available` | int | 0 or 1 | not reported |
| `gpu_count` | int | | not reported |
| `gpu_device_index` | int | | the row had no GPU device |
| `gpu_mem_used_bytes` | float | bytes | not measured |
| `gpu_mem_reserved_bytes` | float | bytes | not measured |
| `gpu_mem_total_bytes` | float | bytes | not measured |

### RuntimeEnvironmentRecord

One record per runtime-environment row of a rank.
It holds the rank's topology, `torch.distributed` state and detected training strategy.
`distributed_initialized` is an integer boolean from the row, `1` or `0`.

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
| `distributed_initialized` | int | 0 or 1 | not reported |
| `distributed_backend` | str | | not reported |
| `training_strategy` | str | | not reported |
| `strategy_source` | str | | not reported |
| `strategy_confidence` | str | | not reported |

### StdoutStderrRecord

One record per captured stdout or stderr line of a rank.
Its only fields are `observed_timestamp_unix_ns`, `rank`, `timestamp_unix_ns` and `line`.

| Field | Type | Unit | `None` means |
|---|---|---|---|
| `observed_timestamp_unix_ns` | int | ns | see common fields |
| `rank` | int | | the rank's `global_rank`; this record has no `global_rank` field |
| `timestamp_unix_ns` | int | ns | see common fields |
| `line` | str | | never `None`; empty lines are not recorded |

## Custom exporters

A custom exporter receives the same records as the built-in `jsonl` exporter, those of the selected streams.
The built-in `wandb` exporter in `traceml_ai.aggregator.export.wandb_exporter` is a worked example of a remote sink on the same interface.
It implements the `Exporter` interface from `traceml_ai.aggregator.export`: `open()`, `write(records)`, `flush()` and `close(*, dropped_batches=0)`.
The export worker calls all four from its own background thread, and the calls never overlap.
It calls `open` once, then `write` and `flush` zero or more times, then `close` once.
`close` is not called when `open` raised.
One `write` batch may hold several selected record types, interleaved in payload order.
An exporter that treats streams differently dispatches on `record.STREAM` or `type(record)`.
`dropped_batches` is the number of payload batches the worker never handed to `write`.
It is one count for the whole worker, not one per stream, so the `jsonl` exporter writes the same value in every closing line.
A closing line's `written_records`, by contrast, counts only the lines of its own file.
A method may raise.
In telemetry mode, an `open` failure prevents the aggregator from accepting
workers, so training does not start under the default launcher policy. In
other modes, and for `write`, `flush`, or `close` failures after startup, the
worker logs the failure, stops using the exporter, and records the loss in
`finalization_warning.json`.

Set the `exporter` setting to `module:factory` to use one.
The value works in each place `jsonl` does.
Use `exporter: "my_exporters:make_exporter"` in `traceml.yaml`, `--exporter my_exporters:make_exporter` on `traceml run` and `traceml serve`, or `TRACEML_EXPORTER=my_exporters:make_exporter`.
With Ray, use `TraceMLRayConfig(exporter="my_exporters:make_exporter")`.
`traceml watch` has no exporter and ignores the setting.
The launcher checks the resolved value from whichever of these sets it.
It checks only the form: a dotted module path, a colon, and one identifier.
It never imports the module.
The aggregator process imports the module, reads the attribute and calls it as `factory(context)`.
The factory runs on the aggregator's main thread at startup, before the worker thread exists.
Create connections, clients and files in `open`, not in the factory, so they belong to the thread that uses them.
The training ranks never import it.
Install the module, or put its directory on `PYTHONPATH`.
`traceml run` starts the aggregator as a separate script, so the working directory is not on its import path.

`context` is an `ExporterContext` with four fields.

- `session_root` is the run's folder, `<logs-dir>/<run-name>`.
- `export_dir` is `session_root / "aggregator" / "export"`, where `jsonl` writes. TraceML does not create it for a factory.
- `include_hostname` tells the exporter whether to keep a record's `hostname` field, if present. It is set by `exporter_hostname`.
- `streams` is the selected `STREAM` names, in the order of the record table above. A factory can use it to prepare one sink per stream.

The failures are an import error, a missing attribute, an exception or `SystemExit` from the module or the factory, and a factory that returns `None`.
Each one is treated like an exporter whose `open` raised.
The worker logs it once with the `[TraceML]` prefix, naming the value and the failure.
In telemetry mode this is a startup failure. In other modes the run continues,
`final_summary.json` is unchanged, and `finalization_warning.json` records the
loss. Its `export` block has `exporter_failed` set to `true`, and
`failed_records` equals the records admitted.
TraceML does not check the returned object up front.
If it lacks `open`, telemetry mode fails during startup. If it lacks another
method, export fails when the worker first calls that method, and the loss is
recorded the same way.

This factory returns an exporter that writes every record as one JSON line to one file.
Each line keeps the `stream` field from `to_dict()`, so a reader can tell the selected streams apart.

```python
import json


class OneFileExporter:
    def __init__(self, path, include_hostname):
        self._path = path
        self._include_hostname = include_hostname
        self._file = None

    def open(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "x", encoding="utf-8")

    def write(self, records):
        for record in records:
            data = record.to_dict()
            if not self._include_hostname:
                data.pop("hostname", None)
            self._file.write(json.dumps(data) + "\n")

    def flush(self):
        self._file.flush()

    def close(self, *, dropped_batches=0):
        self._file.close()


def make_exporter(context):
    return OneFileExporter(
        context.export_dir / "records.jsonl",
        context.include_hostname,
    )
```

## Internal mapping example

This example uses the internal payload mapper.
A custom exporter does not need it; see [Custom exporters](#custom-exporters).

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
