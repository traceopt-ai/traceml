# Final Summary Diagnoses

TraceML emits a sorted list of `DiagnosticIssue` findings or states per final
summary section. `issues[0]` is the primary diagnosis, and final-summary JSON
also copies that same item to `diagnosis`.

The final summary also includes a top-level `primary_diagnosis`. That field is
a run-level performance finding promoted from existing section diagnoses. It is
not a replacement for section diagnoses and it is not a health-warning rollup.
In schema `1.6`, Step Time drives the top-level primary diagnosis; System GPU
utilization can appear as supporting evidence or as an unexplained-utilization
fallback. System, Process, and Step Memory resource findings remain canonical
inside their sections.

Use `kind` as the stable internal key for logic and comparisons. Use `status`
as the user-facing display label. In many cases they are similar, but they are
not required to match; System statuses intentionally use compact labels such as
`HIGH GPU MEM` and `MODERATE GPU UTIL`.

Diagnoses are conservative. They identify likely bottleneck categories, not a
complete root-cause proof.

## System

Host/node pressure and GPU-utilization symptoms.

- `NO_DATA`: no system telemetry.
- `NORMAL`: no CPU, RAM, or available GPU pressure.
- `VERY_HIGH_GPU_MEMORY`: critical GPU memory pressure.
- `HIGH_GPU_MEMORY`: elevated GPU memory pressure.
- `HIGH_GPU_TEMPERATURE`: elevated GPU temperature.
- `HIGH_GPU_POWER`: elevated GPU power use.
- `HIGH_HOST_MEMORY`: elevated host RAM use.
- `HIGH_CPU`: elevated host CPU use.
- `LOW_GPU_UTILIZATION`: average GPU utilization is below 30%. The GPU was
  mostly idle.
- `MODERATE_GPU_UTILIZATION`: average GPU utilization is between 30% and 70%,
  inclusive. The GPU was only partly utilized.

`LOW_GPU_UTILIZATION` and `MODERATE_GPU_UTILIZATION` are observed System
symptoms, not root-cause proof. Use Step Time to identify whether the likely
cause is input loading, compute balance, residuals, synchronization, or other
training behavior. Average GPU utilization above 70% does not emit a
GPU-utilization issue and can remain `NORMAL` when no pressure rules fire.

## Process

Traced process pressure.

- `NO_DATA`: no process telemetry.
- `NORMAL`: no process CPU, RSS, or available GPU memory pressure.
- `VERY_HIGH_PROCESS_GPU_MEMORY`: critical process GPU memory pressure.
- `HIGH_PROCESS_GPU_MEMORY`: elevated process GPU memory pressure.
- `GPU_MEMORY_RESERVED_OVERHANG` with status
  `HIGH CUDA ALLOCATOR RESERVED/ALLOCATED RATIO`: PyTorch CUDA allocator
  reserved memory is much higher than allocated tensor memory.
- `RANK_GPU_MEMORY_IMBALANCE`: GPU memory differs materially across ranks.
- `HIGH_PROCESS_RSS`: elevated process RSS.
- `HIGH_PROCESS_CPU`: elevated process CPU relative to CPU capacity.

## Step Time

Training-step timing.

- `NO_DATA`: no usable step-time data.
- `WARMUP`: some data exists, but not enough for diagnosis.
- `BALANCED`: no clear timing bottleneck or rank straggler.
- `STRAGGLER`: one rank has a mixed clean-step straggler signal.
- `INPUT_STRAGGLER`: one rank has materially higher selected-clock input wait.
- `COMPUTE_STRAGGLER`: one rank has materially higher clean compute time.
- `H2D_STRAGGLER`: one rank has materially higher host-to-device transfer time.
- `RESIDUAL_STRAGGLER`: one rank has materially higher residual `residual_proxy`.
- `INPUT_BOUND`: selected-clock input wait dominates iteration time.
- `COMPUTE_BOUND`: forward/backward/optimizer time dominates the typical step.
- `RESIDUAL_HEAVY`: unattributed residual time is a material share of the step.

Step-time diagnosis uses one selected clock for the analyzed window. It uses
GPU event timing when every rank/step has GPU timing for the step envelope,
input wait, and traced phase events present in the window. Otherwise it uses
explicit `cpu_ms` timing. The live CLI Step Time table, dashboard, and final
summary use the same global-rank SQLite loader and selected-clock window
builder for diagnosis-facing timing; they differ only by row window sizing.
Summary JSON exposes selected-clock `input_wait_ms` and `step_time_ms`.
The compatibility `dataloader_ms` field remains CPU dataloader fetch time, and
`total_step_ms` remains CPU dataloader fetch plus CPU step envelope timing.
These compatibility fields are not selected-clock phase-share denominators.
`duration_ms` stays stored compatibility timing and is not used for Step Time
display or diagnosis. In the final text report, most selected-clock phase
shares are divided by `step_time_ms`; CPU compatibility rows are labeled
separately.

`INPUT_BOUND` uses selected-clock
`input_wait_ms / iteration_time_ms`, where
`iteration_time_ms = input_wait_ms + step_time_ms`. This compares pre-step
input wait with the full selected-clock iteration time while leaving
`step_time_ms` as the traced step envelope. Live and summary diagnosis both
warn at 10% and are critical at 20%.

Shared Step Time diagnosis needs at least 2 steps to emit warning-only
bottleneck diagnoses. Critical diagnoses are allowed once the window has at
least 20 steps. Live and summary use the same diagnosis gates; they differ by
the selected timing window size.

When runtime environment metadata is available, Step Time diagnosis also
receives an advisory training strategy such as `ddp` or `fsdp`. This context is
used only to choose diagnosis attribution behavior; it is not a public Step
Time metric. Missing or unrecognized strategy metadata defaults to `ddp` to
preserve the current straggler behavior. DDP remains eligible for critical
Step Time diagnoses once thresholds and confidence gates are met. FSDP Step
Time diagnoses are capped at warning because collective masking can make
attribution under-confident.

`RESIDUAL_HEAVY` is not a communication diagnosis. `residual_ms` is residual
unattributed step time averaged from per-step clamped residuals:

```text
compute_ms = forward_ms + backward_ms + optimizer_ms
known_step_ms = h2d_ms + compute_ms
traced_step_ms = selected step envelope timing
iteration_time_ms = selected input_wait_ms + selected traced_step_ms
residual_ms = average(max(0, traced_step_ms - known_step_ms))
total_step_ms = CPU dataloader_ms + CPU step envelope timing
```

Rank-local stragglers use clean-step evidence. TraceML first discounts backward
time that can be explained by another rank's non-backward work:

```text
residual_r = residual_proxy_r
non_bwd_r = input_wait_r + h2d_r + forward_r + optimizer_r + residual_r
clean_bwd_r = max(0, backward_r - max(0, max(non_bwd) - non_bwd_r))
clean_compute_r = forward_r + clean_bwd_r + optimizer_r
clean_step_r = input_wait_r + h2d_r + clean_compute_r + residual_r
score = (max(clean_step) - median(clean_step)) / median(actual_step)
```

If `score < 0.10`, TraceML does not report a rank-local straggler. Otherwise it
blames the largest worst-rank excess over peer median among input wait, clean
compute, H2D, and residual. The largest excess must dominate the next-largest
excess by `1.25x`; otherwise the diagnosis stays mixed `STRAGGLER`.

It can include validation, checkpointing, logging, framework orchestration, CPU
stalls, unobserved transfer stalls, or other work inside the timed step but
outside the traced H2D and compute phases. It is not direct NCCL, all-reduce,
or synchronization timing.

## Step Memory

Per-step peak allocated/reserved memory.

- `NO_DATA`: no usable step-memory data.
- `NO_GPU` with status `NO GPU`: no GPU telemetry or CPU-only run.
- `BALANCED`: no clear memory pressure, imbalance, or rising trend.
- `HIGH_PRESSURE`: peak memory is near device capacity.
- `IMBALANCE`: peak memory differs materially across ranks.
- `CREEP_EARLY` with status `MEMORY RISING`: memory is rising, but below the
  confirmed-growth threshold.
- `CREEP_CONFIRMED` with status `MEMORY CREEP`: memory is rising and crossed
  the confirmed-growth threshold.

`HIGH_PRESSURE` requires known GPU capacity. If capacity is unavailable,
TraceML does not guess pressure.
