# Live Export

## What it is

With the `exporter` setting on, the aggregator writes every admitted step-time record to a JSON Lines file while the run is going.
There is one line for every step each rank reports.
The file sits beside the SQLite store under the run's `aggregator/` folder.
A background worker writes it, so the run never waits on the file.
Step time is the only stream so far.
Other streams follow.

## Switching it on

Export is off by default.
`jsonl` is the only exporter so far.
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

Lines carry no hostname by default, because the file may be uploaded elsewhere.
To add `hostname` to every line, use `--exporter-hostname`, `exporter_hostname: true` in YAML, `TRACEML_EXPORTER_HOSTNAME=1`, or `TraceMLRayConfig(exporter_hostname=True)`.
With the switch on, a rank that sent no hostname gives `"hostname": null`.

## The files

The step-time file is `<logs-dir>/<run-name>/aggregator/export/step_time.jsonl`.
On Ray the run folder is named by `session_id`.
There is one file for the whole job, however many ranks and nodes it has, because one aggregator serves the job.
Each line carries `global_rank`, `local_rank`, `node_rank` and `step`.
Key a consumer on `(global_rank, step)`.

The file is created when the aggregator starts.
A run that records no steps still leaves a file, holding only its closing line.
An existing file is never overwritten or appended to.
If the run folder already has one, export stays off for that run and the loss is reported as below, with `exporter_failed` true.

These two lines come from a two-rank CPU run of four steps.
The first is a data line and the second is the closing line:

```json
{"stream":"step_time","schema_version":1,"observed_timestamp_unix_ns":1793808200784977000,"rank":1,"global_rank":1,"local_rank":1,"world_size":2,"local_world_size":2,"node_rank":0,"timestamp_unix_ns":1793808199881904896,"seq":2,"step":1,"events":{"_traceml_internal:backward_time":{"cpu":{"cpu_ms":15.316009521484375,"duration_ms":15.316009521484375,"gpu_ms":null,"is_gpu":false,"n_calls":1}},"_traceml_internal:dataloader_next":{"cpu":{"cpu_ms":3.7009716033935547,"duration_ms":3.7009716033935547,"gpu_ms":null,"is_gpu":false,"n_calls":1}},"_traceml_internal:forward_time":{"cpu":{"cpu_ms":8.20612907409668,"duration_ms":8.20612907409668,"gpu_ms":null,"is_gpu":false,"n_calls":1}},"_traceml_internal:optimizer_step":{"cpu":{"cpu_ms":0.5419254302978516,"duration_ms":0.5419254302978516,"gpu_ms":null,"is_gpu":false,"n_calls":1}},"_traceml_internal:step_time":{"cpu":{"cpu_ms":27.03070640563965,"duration_ms":27.03070640563965,"gpu_ms":null,"is_gpu":false,"n_calls":1}}}}
{"stream":"step_time","closing":true,"schema_version":1,"written_records":8,"dropped_batches":0}
```

`events[name][device]` holds one phase of that step on one device.
It gives the CPU time, the GPU time, the duration TraceML uses, and how many calls were summed.
`gpu_ms` is `null` on a CPU run.
It is also `null` when any call in the step lacked a GPU measurement.

## When something was lost

Export never fails the run.
When the export did not write every admitted record, `finalization_warning.json` in the run folder gets an `export` block.
The block holds `written_records`, `dropped_batches`, `failed_records`, `skipped_items` and `exporter_failed`.
If export is the only problem, the file also says `"reason": "export_incomplete"`.
If another warning already started the file, the `export` block is added to it.

The closing line's `written_records` and `dropped_batches` are the same counts for that file.
`failed_records` and `skipped_items` appear only in the warning file, so read it when you need every count.
A file without a closing line means a write to it failed or the aggregator did not finish.

## Field reference

The field-by-field tables for every record are in [Export Records](../developer_guide/export-records.md).
