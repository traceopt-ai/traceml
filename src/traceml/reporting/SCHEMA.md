# Final Summary JSON

TraceML writes one end-of-run JSON file. Each section has the same outer shape
so the output is easy to store, diff, and consume from tooling.

Sections:

- `system`: node-level CPU, RAM, GPU utilization, GPU memory, temperature,
  power, and headroom.
- `process`: process-level CPU, RSS, and GPU memory across global ranks.
- `step_time`: aligned training-step timing across global ranks.
- `step_memory`: aligned per-step peak allocated/reserved memory.

## Section Shape

```json
{
  "metadata": {
    "mode": "single_node | multi_node | no_data",
    "duration_s": null,
    "samples": null,
    "nodes_expected": null,
    "nodes_observed": null,
    "nodes_coverage": null,
    "nodes_partial": null,
    "gpus_observed": null,
    "global_ranks_seen": null,
    "global_ranks_used": null,
    "training_total_steps": null,
    "training_latest_step": null,
    "section_metric_names": []
  },
  "diagnosis": {},
  "issues": [],
  "global": {
    "index_by": "node_rank | global_rank",
    "window": {
      "kind": "sample_window | step_window",
      "alignment": "none | common_steps",
      "samples": null,
      "steps_analyzed": null,
      "start_step": null,
      "end_step": null,
      "completed_step": null,
      "window_size": null
    },
    "average": {"<metric_name>": null},
    "median": {"<metric_name>": {"value": null, "idx": null}},
    "worst": {"<metric_name>": {"value": null, "idx": null}}
  },
  "groups": {
    "by": "node_rank | global_rank",
    "rows": {
      "0": {
        "identity": {
          "global_rank": null,
          "local_rank": null,
          "node_rank": null,
          "hostname": null,
          "local_world_size": null,
          "world_size": null
        },
        "metrics": {"<metric_name>": null}
      }
    }
  },
  "units": {},
  "card": ""
}
```

## Field Rules

- `diagnosis` is the primary section-level result.
- `issues` is the sorted list of material issues behind the diagnosis.
- `groups.rows` contains row data only: `identity` and `metrics`.
- Row-level diagnosis is intentionally omitted for now.
- `global.average`, `global.median`, `global.worst`, and
  `groups.rows[*].metrics` must use exactly `metadata.section_metric_names`.
- `global.index_by` must match `groups.by`.
- `idx` points to a key in `groups.rows`.
- `metadata.global_ranks_seen` is all observed ranks.
- `metadata.global_ranks_used` is the ranks included in `groups.rows` and the
  `global` comparison.

`step_time` and `step_memory` use `common_steps` alignment. If a rank does not
have the common step window, it can be counted in `global_ranks_seen` but not
in `global_ranks_used`.

## Metric Names

```json
{
  "system": [
    "cpu_percent",
    "ram_bytes",
    "ram_percent",
    "gpu_util_percent",
    "gpu_mem_bytes",
    "gpu_mem_percent",
    "gpu_temp_c",
    "gpu_power_w",
    "gpu_headroom_bytes"
  ],
  "process": [
    "cpu_percent",
    "cpu_capacity_percent",
    "ram_bytes",
    "ram_percent",
    "gpu_mem_used_bytes",
    "gpu_mem_reserved_bytes",
    "gpu_mem_reserved_percent",
    "gpu_mem_headroom_bytes"
  ],
  "step_time": [
    "total_step_ms",
    "dataloader_ms",
    "compute_ms",
    "wait_ms",
    "forward_ms",
    "backward_ms",
    "optimizer_ms"
  ],
  "step_memory": [
    "peak_allocated_bytes",
    "peak_reserved_bytes"
  ]
}
```

Metric suffixes are units:

- `_bytes`
- `_ms`
- `_percent`
- `_c`
- `_w`

## Step Time Wait

`wait_ms` is residual unattributed step time:

```text
compute_ms = forward_ms + backward_ms + optimizer_ms
traced_step_ms = max(raw_trace_step_wall_ms, compute_ms)
wait_ms = traced_step_ms - compute_ms
total_step_ms = dataloader_ms + traced_step_ms
```

The public contract is:

```text
total_step_ms = dataloader_ms + compute_ms + wait_ms
```

`traced_step_ms` and the raw `trace_step` wall timer are internal measurement
details and are not emitted in final-summary JSON. `wait_ms` can include
validation, checkpointing, logging, framework orchestration, CPU stalls,
transfer stalls, or other work outside the traced compute phases. Do not treat
it as NCCL, all-reduce, or synchronization overhead without profiler evidence.
