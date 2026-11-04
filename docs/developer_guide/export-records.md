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

Step time is the only stream with a record so far.
The other streams (step memory, system, process, runtime environment, stdout/stderr) will get records in a follow-up.

Every record class carries `STREAM`, `SCHEMA_VERSION` (currently `1`) and `COLUMNS`, its field names in order.
`to_dict()` returns `stream`, `schema_version` and then every field.
An unavailable value remains `None`; a measured zero remains `0`.
`StepTimeRecord` differs from its row in one field.
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
Every table the step-time projection writes has a record, and a test keeps it that way.

## Common fields

A record starts with these identity and time fields.

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
The StepTime sampler emits steps in order within one rank.
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

## Custom exporters

A custom exporter receives the same records as the built-in `jsonl` exporter.
It implements the `Exporter` interface from `traceml_ai.aggregator.export`: `open()`, `write(records)`, `flush()` and `close(*, dropped_batches=0)`.
The export worker calls all four from its own background thread, and the calls never overlap.
It calls `open` once, then `write` and `flush` once per tick, then `close` once.
`close` is not called when `open` raised.
`dropped_batches` is the number of payload batches the worker never handed to `write`.
A method may raise.
The worker then logs the failure, stops using the exporter, and records the loss in `finalization_warning.json`.

Set the `exporter` setting to `module:factory` to use one.
The value works in each place `jsonl` does.
That is `exporter: "my_exporters:make_exporter"` in `traceml.yaml`, `--exporter my_exporters:make_exporter` on `traceml run` and `traceml serve`, or `TRACEML_EXPORTER=my_exporters:make_exporter`.
The launcher checks only that both sides of the colon are non-empty.
It never imports the module.
The aggregator process imports the module, reads the attribute and calls it as `factory(context)`.
The factory runs on the aggregator's main thread at startup, before the worker thread exists.
Create connections, clients and files in `open`, not in the factory, so they belong to the thread that uses them.
The training ranks never import it.
Install the module, or put its directory on `PYTHONPATH`.
`traceml run` starts the aggregator as a separate script, so the working directory is not on its import path.

`context` is an `ExporterContext` with three fields.

- `session_root` is the run's folder, `<logs-dir>/<run-name>`.
- `export_dir` is `session_root / "aggregator" / "export"`, where `jsonl` writes. TraceML does not create it for a factory.
- `include_hostname` is the `exporter_hostname` setting.

A factory failure never fails the run.
An import error, a missing attribute, an exception from the factory, or a factory that returns `None` is logged once with the `[TraceML]` prefix.
The run then continues without export, exactly as with no exporter set.
`final_summary.json` is the same either way.

This factory returns an exporter that appends every record as one JSON line to one file.

```python
import json


class OneFileExporter:
    def __init__(self, path):
        self._path = path
        self._file = None

    def open(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a", encoding="utf-8")

    def write(self, records):
        for record in records:
            self._file.write(json.dumps(record.to_dict()) + "\n")

    def flush(self):
        self._file.flush()

    def close(self, *, dropped_batches=0):
        self._file.close()


def make_exporter(context):
    return OneFileExporter(context.export_dir / "records.jsonl")
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
