# Final Summary JSON

TraceML writes one end-of-run JSON file. The current schema version is `1.7`.
Each section has the same outer shape so the output is easy to store, diff, and
consume from tooling.

Schema `1.7` publishes canonical Step Time vocabulary. Every public timing
metric is nullable: `null` means the
underlying timing signal was never measured in the analyzed window (missing
instrumentation), while a measured zero stays `0.0`. Null metrics are
excluded from `global.average`, `global.median`, and `global.worst` and from
rank median/worst selection; a rank with only some metrics measured (for
example an H2D-only rank) keeps its row with `null` for the others.

For user-facing definitions and examples, see the
[Step Time glossary](../../../docs/user_guide/reading-output.md#step-time-glossary).

Sections:

- `system`: node-level CPU, RAM, GPU utilization, GPU memory, temperature,
  power, and headroom.
- `process`: process-level CPU, RSS, and GPU memory across global ranks.
- `step_time`: aligned training-step timing across global ranks.
- `step_memory`: aligned per-step peak allocated/reserved memory.

## Top-Level Shape

```json
{
  "schema_version": 1.7,
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
read telemetry tables or recompute diagnostics. In schema `1.6`, Step Time
diagnoses drive primary performance diagnosis. System GPU utilization is only
supporting evidence, except for the fallback
`LOW_GPU_UTILIZATION_UNEXPLAINED` when Step Time has no useful performance
cause. Process, System health, and Step Memory health findings remain in their
section diagnoses.

Selection policy:

- `INPUT_STRAGGLER`, `COMPUTE_STRAGGLER`, `H2D_STRAGGLER`, and `STRAGGLER`
  use rank-comparison evidence and are promoted from `step_time.diagnosis`.
- `RESIDUAL_HEAVY`, `INPUT_BOUND`, `H2D_BOUND`, and `COMPUTE_BOUND` use
  phase-share evidence and are promoted from `step_time.diagnosis`.
- `LOW_GPU_UTILIZATION_UNEXPLAINED` appears only when Step Time is `BALANCED`
  and System reports `LOW_GPU_UTILIZATION` or `MODERATE_GPU_UTILIZATION`.
- `NO_CLEAR_PERFORMANCE_BOTTLENECK` appears when Step Time is `BALANCED` and
  GPU utilization is not low/moderate.
- `INSUFFICIENT_STEP_TIME_DATA` appears when Step Time is `NO_DATA`,
  `WARMUP`, or `INCOMPLETE_DATA`. For `INCOMPLETE_DATA` its summary and
  action name the missing-phase problem, its evidence carries
  `step_time_status: "INCOMPLETE DATA"`, and the Step Time section's
  diagnosis evidence lists `missing_signals` plus per-signal
  `signal_coverage`.
- Step Time may emit warning-only bottleneck diagnoses before its confident
  threshold; critical Step Time diagnoses require the confident window size.
  Live and summary use the same global-rank Step Time SQLite window loader;
  summary uses a larger selected-clock window, but not a separate Step Time
  diagnosis gate.
- Step Time diagnosis may consume advisory runtime training strategy context
  when available. This does not add a public summary metric; missing or
  unrecognized strategy metadata defaults to `ddp`. FSDP Step Time diagnosis
  severity is capped at warning.

High temperature, memory pressure, memory creep, high RSS, high CPU, and other
resource-health findings are not promoted into `primary_diagnosis` in schema
`1.6`. They remain available under their section's `diagnosis` and `issues`.

Primary diagnosis evidence uses a small union:

```json
{
  "type": "phase_share",
  "basis": "average",
  "steps_analyzed": 256,
  "input_wait_ms": 80.0,
  "step_time_ms": 240.0,
  "traced_step_time_ms": 160.0,
  "diagnosis_clock": "gpu",
  "dataloader_fetch_cpu_ms": 40.0,
  "h2d_ms": 0.4,
  "compute_ms": 120.0,
  "residual_ms": 39.6,
  "score": 0.333,
  "score_basis": "median_per_rank_step_time_share",
  "score_denominator": "selected-clock Step Time per rank",
  "gpu_util_avg_percent": 37.8
}
```

`phase_share` is used for `INPUT_BOUND`, `H2D_BOUND`, `RESIDUAL_HEAVY`, and
`COMPUTE_BOUND`. Millisecond values come from `step_time.global.average` and
are supporting observations only. For scored typical bottlenecks, `score` is
the authoritative median per-rank Step Time fraction. Informational
`COMPUTE_BOUND` uses the median per-rank
`(forward_ms + backward_ms + optimizer_ms) / step_time_ms` share with a
90% threshold, but has no score because it is excluded from impact-based
primary ordering.

```json
{
  "type": "rank_comparison",
  "metric": "input_wait_ms",
  "phase": "input",
  "steps_analyzed": 256,
  "median": {"rank": 0, "value_ms": 0.7},
  "worst": {"rank": 2, "value_ms": 180.9},
  "delta_ms": 180.2,
  "ratio": 262.4,
  "gpu_util_avg_percent": 80.0
}
```

`rank_comparison` is used for `INPUT_STRAGGLER`, `COMPUTE_STRAGGLER`,
`H2D_STRAGGLER`, and `STRAGGLER`. Values come from `step_time.global` rank
summaries. Generic `STRAGGLER` may contain a `comparisons` array instead of a
single metric comparison.

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
- `score` is an optional section-specific ranking signal. In Step Time, scored
  typical bottlenecks use median per-rank Step Time impact and stragglers use
  visible wait cost divided by victim Step Time.
- Section-specific details such as `scope`, `samples_used`, `steps_used`,
  `note`, and `confidence` belong in `evidence`.
- `groups.rows` contains row data only: `identity` and `metrics`.
- Row-level diagnosis is intentionally omitted for now.
- `global.average`, `global.median`, `global.worst`, and
  `groups.rows[*].metrics` must use exactly `metadata.section_metric_names`.
  Keys are always present; a Step Time metric whose signal was never
  measured carries `null` (`{"value": null, "idx": null}` for rank points)
  and never a fabricated `0.0`.
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
    "input_wait_ms",
    "step_time_ms",
    "traced_step_time_ms",
    "step_time_cpu_ms",
    "step_time_gpu_ms",
    "traced_step_time_cpu_ms",
    "traced_step_time_gpu_ms",
    "dataloader_fetch_cpu_ms",
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

Step Time uses one selected clock per aligned window. GPU timing is used when
the window has complete GPU event timings; otherwise explicit CPU timing is
used. `input_wait_ms`, `step_time_ms`, and `traced_step_time_ms` expose
selected-clock values. The four explicit aggregate fields preserve both clocks:

```text
step_time_cpu_ms
step_time_gpu_ms
traced_step_time_cpu_ms
traced_step_time_gpu_ms
```

`dataloader_fetch_cpu_ms` is supplemental CPU DataLoader-fetch evidence. It
is never added into Input Wait or Step Time and has no phase-share percentage.
Every displayed phase share uses the selected `step_time_ms` denominator.
Those table shares are observational averages and may differ from the
authoritative median per-rank diagnosis `score`.

When GPU is selected, `input_wait_ms` is GPU-clocked while
`dataloader_fetch_cpu_ms` remains CPU evidence, so they can differ. When CPU
is selected they describe the same CPU input-wait interval; neither is counted
twice.

`residual_ms` is residual unattributed step time. It is averaged from
per-step clamped residuals, not recomputed from already-averaged phase totals:

```text
compute_ms = forward_ms + backward_ms + optimizer_ms
known_step_ms = h2d_ms + compute_ms
traced_step_time_ms = selected traced envelope timing
step_time_ms = selected input_wait_ms + selected traced_step_time_ms
residual_ms = average(max(0, traced_step_time_ms - known_step_ms))
```

The selected-clock diagnosis contract is:

```text
input_wait_ms = selected-clock input wait
traced_step_time_ms = selected-clock Traced Step Time
step_time_ms = complete selected-clock step duration
diagnosis_clock = "cpu" | "gpu"
```

Schema `1.7` does not publish historical timing aliases. Readers that support
older summary files must use an explicit schema-versioned compatibility adapter
at their input boundary; new output remains canonical.

`duration_ms` is stored compatibility timing and is not a Step Time display or
diagnosis fallback. `residual_ms` can include validation, checkpointing,
logging, framework orchestration, CPU stalls, unobserved transfer stalls, or
other work outside the traced H2D and compute phases. Do not treat it as NCCL,
all-reduce, or synchronization overhead without profiler evidence.
