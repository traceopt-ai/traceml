# Final Report Section Shape

Each final-report section keeps top-level JSON fields non-overlapping:

- `metadata`: flat, table-friendly section context. Every section emits the
  same keys and uses `null` when a field does not apply.
- `diagnosis`: the single section-level diagnosis shown in the text card.
- `global.window`: flat window metadata for the run-level calculation. Every
  section emits the same window keys.
- `global.average`: average values over `global.window`. Keys must match
  `metadata.section_metric_names` exactly.
- `global.median` and `global.worst`: median and worst comparison points over
  `global.window`. Keys must match `metadata.section_metric_names` exactly.
- `groups.by`: the detail dimension, either `node_rank` or `global_rank`.
- `groups.rows`: detailed rows keyed by that dimension. Each row has
  `identity`, `diagnosis`, `issues`, and `metrics`.
- `groups.rows[*].metrics`: row-level average values. Keys must match
  `metadata.section_metric_names` exactly.

The top-level section keys are intentionally strict:

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
    "average": {
      "<metric_name>": null
    },
    "median": {
      "<metric_name>": {
        "value": null,
        "idx": null
      }
    },
    "worst": {
      "<metric_name>": {
        "value": null,
        "idx": null
      }
    }
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
        "diagnosis": null,
        "issues": [],
        "metrics": {
          "<metric_name>": null
        }
      }
    }
  },
  "units": {},
  "card": ""
}
```

This split keeps the JSON stable for downstream tools and avoids storing the
same meaning in multiple places.

## Group Identity

Every row in `groups.rows` includes the same runtime identity fields:

```json
{
  "global_rank": 0,
  "local_rank": 0,
  "node_rank": 0,
  "hostname": "worker-0",
  "local_world_size": 4,
  "world_size": 8
}
```

These fields answer where the row came from in a distributed job. They are the
shared identity contract for every section.

## Metric Names

The metric names listed in `metadata.section_metric_names` are the contract for
`global.average`, `global.median`, and `global.worst`.

For every section:

```text
set(global.average.keys()) == set(metadata.section_metric_names)
set(global.median.keys()) == set(metadata.section_metric_names)
set(global.worst.keys()) == set(metadata.section_metric_names)
set(groups.rows[*].metrics.keys()) == set(metadata.section_metric_names)
global.index_by == groups.by
```

Metric names include units in the field name:

- memory: `*_bytes`
- time: `*_ms`
- percent: `*_percent`
- temperature: `*_c`
- power: `*_w`

Current section metric names:

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
    "step_time_ms",
    "dataloader_ms",
    "forward_ms",
    "backward_ms",
    "optimizer_ms",
    "compute_ms",
    "wait_ms"
  ],
  "step_memory": [
    "peak_allocated_bytes",
    "peak_reserved_bytes"
  ]
}
```

`global.median` and `global.worst` use one `idx` per metric. The meaning of
`idx` is defined by `global.index_by`, and it points to a key in `groups.rows`.

- System uses `index_by: "node_rank"`, so `idx` values look like `"0"`.
- Process, Step Time, and Step Memory use `index_by: "global_rank"`, so `idx`
  values look like `"0"`, `"1"`, and so on.
