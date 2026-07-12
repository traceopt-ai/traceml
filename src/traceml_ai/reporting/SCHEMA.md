# Final Summary JSON

TraceML writes one end-of-run JSON file. The current schema version is `1.5`.
Each section has the same outer shape so the output is easy to store, diff, and
consume from tooling.

Sections:

- `system`: node-level CPU, RAM, GPU utilization, GPU memory, temperature,
  power, and headroom.
- `process`: process-level CPU, RSS, and GPU memory across global ranks.
- `step_time`: aligned training-step timing across global ranks.
- `step_memory`: aligned per-step peak allocated/reserved memory.

## Top-Level Shape

```json
{
  "schema_version": 1.5,
  "generated_at": "...",
  "duration_s": null,
  "meta": {
    "run_name": null,
    "mode": "single_node | multi_node | no_data",
    "world_size": null,
    "nodes_observed": null,
    "gpus_observed": null
  },
  "primary_diagnosis": {},
  "system": {},
  "process": {},
  "step_time": {},
  "step_memory": {},
  "text": ""
}
```

`meta` contains run-level identity and observed topology. Section-level
`metadata` remains section-specific coverage and metric-contract information.

`primary_diagnosis` is a top-level performance finding promoted from existing
section diagnoses. It answers "why was training slow?" and is intentionally
narrower than section-level health/resource diagnoses.

`text` is a compact human-readable verdict report. It is presentation text for
the CLI/TXT artifact, not a structured contract for downstream parsers. It
starts with `TraceML Verdict`, `Why`, and `Next`, then shows compact section
status plus System and Step Time evidence tables. Detailed section prose
remains in each section-local `card` field.

## Primary Diagnosis Shape

```json
{
  "kind": "INPUT_BOUND",
  "status": "INPUT-BOUND",
  "severity": "info | warn | crit",
  "section": "step_time | system",
  "scope": "performance",
  "summary": "...",
  "action": "...",
  "evidence": {}
}
```

`primary_diagnosis` is derived from already-built section payloads. It does not
read telemetry tables or recompute diagnostics. In schema `1.5`, Step Time
diagnoses drive primary performance diagnosis. System GPU utilization is only
supporting evidence, except for the fallback
`LOW_GPU_UTILIZATION_UNEXPLAINED` when Step Time has no useful performance
cause. Process, System health, and Step Memory health findings remain in their
section diagnoses.

Selection policy:

- `INPUT_STRAGGLER`, `COMPUTE_STRAGGLER`, `H2D_STRAGGLER`,
  `RESIDUAL_STRAGGLER`, and `STRAGGLER` use rank-comparison evidence and are
  promoted from `step_time.diagnosis`.
- `RESIDUAL_HEAVY`, `INPUT_BOUND`, and `COMPUTE_BOUND` use phase-share evidence
  and are promoted from `step_time.diagnosis`.
- `LOW_GPU_UTILIZATION_UNEXPLAINED` appears only when Step Time is `BALANCED`
  and System reports `LOW_GPU_UTILIZATION` or `MODERATE_GPU_UTILIZATION`.
- `NO_CLEAR_PERFORMANCE_BOTTLENECK` appears when Step Time is `BALANCED` and
  GPU utilization is not low/moderate.
- `INSUFFICIENT_STEP_TIME_DATA` appears when Step Time is `NO_DATA` or
  `WARMUP`.

High temperature, memory pressure, memory creep, high RSS, high CPU, and other
resource-health findings are not promoted into `primary_diagnosis` in schema
`1.5`. They remain available under their section's `diagnosis` and `issues`.

Primary diagnosis evidence uses a small union:

```json
{
  "type": "phase_share",
  "basis": "average",
  "steps_analyzed": 256,
  "total_step_ms": 272.3,
  "dataloader_ms": 161.0,
  "h2d_ms": 0.1,
  "compute_ms": 109.6,
  "residual_ms": 1.6,
  "shares": {
    "dataloader_pct": 59.1,
    "h2d_pct": 0.0,
    "compute_pct": 40.3,
    "residual_pct": 0.6
  },
  "gpu_util_avg_percent": 37.8
}
```

`phase_share` is used for `INPUT_BOUND`, `RESIDUAL_HEAVY`, and `COMPUTE_BOUND`.
Values come from `step_time.global.average`.

```json
{
  "type": "rank_comparison",
  "metric": "dataloader_ms",
  "phase": "dataloader",
  "steps_analyzed": 256,
  "median": {"rank": 0, "value_ms": 0.7},
  "worst": {"rank": 2, "value_ms": 180.9},
  "delta_ms": 180.2,
  "ratio": 262.4,
  "gpu_util_avg_percent": 80.0
}
```

`rank_comparison` is used for `INPUT_STRAGGLER`, `COMPUTE_STRAGGLER`,
`H2D_STRAGGLER`, `RESIDUAL_STRAGGLER`, and `STRAGGLER`. Values come from
`step_time.global.median[metric]` and
`step_time.global.worst[metric]`. Generic `STRAGGLER` may contain a
`comparisons` array instead of a single metric comparison.

Fallback evidence types are:

- `utilization_fallback` for `LOW_GPU_UTILIZATION_UNEXPLAINED`
- `no_clear_bottleneck` for `NO_CLEAR_PERFORMANCE_BOTTLENECK`
- `insufficient_data` for `INSUFFICIENT_STEP_TIME_DATA`

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
  "diagnosis": {
    "kind": "...",
    "status": "...",
    "severity": "info | warn | crit",
    "summary": "...",
    "action": "...",
    "metric": null,
    "phase": null,
    "score": null,
    "share_pct": null,
    "skew_pct": null,
    "ranks": [],
    "evidence": {}
  },
  "issues": [
    {
      "kind": "...",
      "status": "...",
      "severity": "info | warn | crit",
      "summary": "...",
      "action": "...",
      "metric": null,
      "phase": null,
      "score": null,
      "share_pct": null,
      "skew_pct": null,
      "ranks": [],
      "evidence": {}
    }
  ],
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

- `issues` is the canonical sorted list of diagnostic findings or states.
- `issues` is always non-empty.
- `diagnosis` is always equal to `issues[0]`.
- Neutral states such as `NORMAL`, `BALANCED`, `NO_DATA`, `WARMUP`, and
  `NO_GPU` are represented with the same issue shape as actionable findings.
- `kind` is the stable internal key for code, comparisons, and frontend logic.
- `status` is the user-facing display label.
- `summary` is the short explanation. Older `reason` fields should be treated
  as pre-`1.4` input, not the current final-summary contract.
- Section-specific details such as `scope`, `samples_used`, `steps_used`,
  `note`, and `confidence` belong in `evidence`.
- `groups.rows` contains row data only: `identity` and `metrics`.
- Row-level diagnosis is intentionally omitted for now.
- `global.average`, `global.median`, `global.worst`, and
  `groups.rows[*].metrics` must use exactly `metadata.section_metric_names`.
- `global.index_by` must match `groups.by`.
- `idx` points to a key in `groups.rows`.
- `metadata.global_ranks_seen` is all observed ranks.
- `metadata.global_ranks_used` is the ranks included in `groups.rows` and the
  `global` comparison.
- `card` is the section-local detailed text block used by section tooling and
  retained in JSON even when top-level `text` uses the compact verdict report.

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
    "h2d_ms",
    "compute_ms",
    "residual_ms",
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

## Step Time Residual

`residual_ms` is residual unattributed step time:

```text
compute_ms = forward_ms + backward_ms + optimizer_ms
known_step_ms = h2d_ms + compute_ms
traced_step_ms = max(raw_trace_step_wall_ms, known_step_ms)
residual_ms = traced_step_ms - known_step_ms
total_step_ms = dataloader_ms + traced_step_ms
```

The public contract is:

```text
total_step_ms = dataloader_ms + h2d_ms + compute_ms + residual_ms
```

`traced_step_ms` and the raw `trace_step` wall timer are internal measurement
details and are not emitted in final-summary JSON. `residual_ms` can include
validation, checkpointing, logging, framework orchestration, CPU stalls,
unobserved transfer stalls, or other work outside the traced H2D and compute
phases. Do not treat it as NCCL, all-reduce, or synchronization overhead
without profiler evidence.
