# Live Export

## What it is

With the `exporter` setting on, the aggregator writes telemetry records to JSON Lines files while the run is going.
It writes one file per selected stream, under the run's `aggregator/export/` folder.
By default it exports only the `step_time` stream, which has one line for every step each rank reports.
A background worker writes the files, so training does not wait for file writes.
At shutdown, TraceML waits briefly for pending records to be written.

## Switching it on

Export is off by default.
The built-in exporter is `jsonl`.
A custom exporter is selected by setting `exporter` to a `module:factory` value, described in [Custom exporters](../developer_guide/export-records.md#custom-exporters).
Pick whichever layer fits your setup:

```bash
traceml run --exporter jsonl train.py
traceml serve --exporter jsonl
TRACEML_EXPORTER=jsonl traceml run train.py
```

```yaml
# traceml.yaml
exporter: jsonl
```

```python
from traceml_ai.integrations.ray import TraceMLRayConfig

TraceMLRayConfig(exporter="jsonl")
```

The launcher resolves the setting like every other one.
A CLI flag beats the `TRACEML_*` environment variable, which beats `traceml.yaml`.
With Ray, the fields on `TraceMLRayConfig` set it.
Ray reads no environment variable and no `traceml.yaml` for it, so `exporter=""` means off.

### Export without local analysis

Choose `telemetry` mode to export measurements without a local SQLite database,
diagnostics, live UI, or summary generation:

```bash
traceml run train.py --mode telemetry --exporter jsonl
```

Telemetry mode always disables local history, including when inherited
configuration sets `history_enabled: true`. It requires an exporter and rejects
HTML reports and regression guards. `traceml.summary()` and
`traceml.final_summary()` are unavailable in this mode. `watch` does not support
telemetry mode.

Operational logs, run metadata, and launcher-captured training output may still
be written locally. Raw sampler logging is controlled separately by
`--enable-logging`, `TRACEML_ENABLE_LOGGING`, or `enable_logging` in
`traceml.yaml`. It is off by default; enabling it also writes local `.msgpack`
files, including in telemetry mode.

To export measurements and retain local analysis and the final summary:

```bash
traceml run train.py --mode summary --exporter jsonl
```

The aggregator remains active in both cases. It admits measurements, consumes
rank-completion markers, and drains pending telemetry and exports within the
existing shutdown bounds. Completion markers do not guarantee delivery or
indicate that training succeeded. In telemetry mode, the required exporter
must open before the aggregator accepts workers; an initialization failure is
an aggregator startup failure, so the default launcher policy does not start
training. After training starts, export failures and losses remain warnings and
do not replace the training exit code. In summary mode, export remains optional
and its initialization failures are warnings because local analysis continues.

For separately launched workers, select telemetry mode on the worker when its
code may call `traceml.summary()` or `traceml.final_summary()`:

```bash
# Terminal 1
traceml serve --mode telemetry --exporter jsonl --run-name my-run

# Terminal 2: train.py calls traceml.init() and instruments its steps
TRACEML_UI_MODE=telemetry TRACEML_SESSION_ID=my-run python train.py
```

Alternatively, select `ui_mode="telemetry"` in the worker's `traceml.init()`.
Workers send measurements independently of this setting; the telemetry-mode
server still exports them without SQLite history. Setting the worker mode makes
summary API calls fail immediately instead of waiting for a service that the
telemetry server intentionally does not run. The standalone server does not
negotiate execution mode with its workers. Stop `serve` with Ctrl+C or SIGTERM
after training ends. CLI-launched runs and Ray stop their aggregator through
their existing lifecycle.

JSONL remains a local-file exporter. Direct platform delivery requires an
appropriate custom or remote exporter; telemetry mode does not add one.

Lines carry no hostname by default, because the file may be uploaded elsewhere.
To include `hostname` on records that have this field, use `--exporter-hostname`, `exporter_hostname: true` in YAML, `TRACEML_EXPORTER_HOSTNAME=1`, or `TraceMLRayConfig(exporter_hostname=True)`.
For these records, a rank that sent no hostname gives `"hostname": null`.
Stdout/stderr records have no hostname field.

## Choosing streams

With export on and nothing else set, only `step_time` is exported.
The `export_streams` setting chooses the streams.
It takes a comma-separated list of these names:

| Name | Files | One line per |
|---|---|---|
| `step_time` (default) | `step_time.jsonl` | step of each rank, with its phase timings |
| `step_memory` | `step_memory.jsonl` | step and device of each rank, with peak memory |
| `process` | `process.jsonl` | snapshot of each training process, with its CPU, RAM and GPU memory |
| `system` | `system.jsonl` and `system_gpu.jsonl` | host snapshot, and each GPU in that snapshot |
| `all` | every file above, plus `runtime_environment.jsonl` and `stdout_stderr.jsonl` | record of every stream |

To add step memory to the default, name both streams:

```bash
traceml run --exporter jsonl --export-streams step_time,step_memory train.py
TRACEML_EXPORTER=jsonl TRACEML_EXPORT_STREAMS=step_time,step_memory traceml run train.py
```

```yaml
# traceml.yaml
exporter: jsonl
export_streams: step_time,step_memory
```

```python
TraceMLRayConfig(exporter="jsonl", export_streams="step_time,step_memory")
```

To add process and system telemetry, name those streams as well:

```bash
traceml run --exporter jsonl --export-streams step_time,process,system train.py
```

```yaml
# traceml.yaml
exporter: jsonl
export_streams: step_time,process,system
```

`system` writes two files.
`system.jsonl` has one line per host snapshot.
`system_gpu.jsonl` has one line per GPU in that snapshot.
On a multi-rank run, only local rank 0 of each node sends system snapshots.

To export everything, use `all`:

```bash
traceml run --exporter jsonl --export-streams all train.py
```

```yaml
# traceml.yaml
exporter: jsonl
export_streams: all
```

`all` adds two streams that no other name selects.
`runtime_environment.jsonl` holds each rank's topology, `torch.distributed` state and detected training strategy.
`stdout_stderr.jsonl` is the stream for captured stdout and stderr lines.
Ranks of this release do not capture stdout or stderr, so that file holds only its closing line.

Spaces around the names are ignored, and a repeated name is harmless.
The order you write them in does not change the output.
`export_streams` resolves like `exporter`.
The CLI flag beats `TRACEML_EXPORT_STREAMS`, which beats `traceml.yaml`, which beats the default.
With Ray, only `TraceMLRayConfig(export_streams=...)` sets it.
An unknown name stops `traceml run` and `traceml serve` before launch, and the message lists the supported names.
The setting changes nothing while `exporter` is off, though `traceml run` and `traceml serve` still check the value.

## The files

Each selected stream has its own file, `<logs-dir>/<run-name>/aggregator/export/<stream>.jsonl`.
With the default, that is `step_time.jsonl` alone.
On Ray the run folder is named by `session_id`.
There is one set of files for the whole job, however many ranks and nodes it has, because one aggregator serves the job.

Fields vary by stream.
Every data line starts with `stream` and `schema_version`.
The [Export Records](../developer_guide/export-records.md) tables list the other fields of each stream.
Key step-time lines on `(global_rank, step)`.
Key step-memory lines on `(global_rank, device, step)`, because one step has one line per device.
The developer page gives the key of every stream.

Every selected file is created when the aggregator starts.
A stream with no records still leaves a file, holding only its closing line.
An existing file is never overwritten or appended to.
If the run folder already has one of the selected files, export stays off for that run.
The loss is then reported as below, with `exporter_failed` true.

These two lines come from `step_time.jsonl` of a two-rank CPU run of four steps.
The first is a data line and the second is the closing line:

```json
{"stream":"step_time","schema_version":1,"observed_timestamp_unix_ns":1793808200784977000,"rank":1,"global_rank":1,"local_rank":1,"world_size":2,"local_world_size":2,"node_rank":0,"timestamp_unix_ns":1793808199881904896,"seq":2,"step":1,"events":{"_traceml_internal:backward_time":{"cpu":{"cpu_ms":15.316009521484375,"duration_ms":15.316009521484375,"gpu_ms":null,"is_gpu":false,"n_calls":1}},"_traceml_internal:dataloader_next":{"cpu":{"cpu_ms":3.7009716033935547,"duration_ms":3.7009716033935547,"gpu_ms":null,"is_gpu":false,"n_calls":1}},"_traceml_internal:forward_time":{"cpu":{"cpu_ms":8.20612907409668,"duration_ms":8.20612907409668,"gpu_ms":null,"is_gpu":false,"n_calls":1}},"_traceml_internal:optimizer_step":{"cpu":{"cpu_ms":0.5419254302978516,"duration_ms":0.5419254302978516,"gpu_ms":null,"is_gpu":false,"n_calls":1}},"_traceml_internal:step_time":{"cpu":{"cpu_ms":27.03070640563965,"duration_ms":27.03070640563965,"gpu_ms":null,"is_gpu":false,"n_calls":1}}}}
{"stream":"step_time","closing":true,"schema_version":1,"written_records":8,"dropped_batches":0}
```

`events[name][device]` holds one phase of that step on one device.
It gives the CPU time, the GPU time, the duration TraceML uses, and how many calls were summed.
`gpu_ms` is `null` on a CPU run.
It is also `null` when any call in the step lacked a GPU measurement.

A file that finished cleanly ends with one closing line.
Its `written_records` counts the data lines of that file only.
Its `dropped_batches` is one count for the whole export worker, because a dropped batch can hold records of any stream.
So every closing line of a run shows the same `dropped_batches`.

## When something was lost

After training starts, export never changes the training outcome. Telemetry
mode first requires its exporter to open successfully, as described above.
When the export did not write every record of the selected streams,
`finalization_warning.json` in the run folder gets an `export` block.
The block holds `written_records`, `dropped_batches`, `failed_records`, `skipped_items` and `exporter_failed`.
If export is the only problem, the file also says `"reason": "export_incomplete"`.
If another warning already started the file, the `export` block is added to it.

The block's `written_records` is the total over all files.
Its `dropped_batches` is the number every closing line shows.
`failed_records` and `skipped_items` appear only in the warning file, so read it when you need every count.
A stream you did not select is not a loss.
Its records are never built, so they are not counted as written, failed or skipped.
`dropped_batches` counts whole payload batches before selection, so a dropped batch may have held no selected record.
After the run finishes, a file without a closing line means a write failed or the aggregator did not finish cleanly.

## Field reference

The field-by-field tables for every record are in [Export Records](../developer_guide/export-records.md).
