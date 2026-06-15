# Final Summary Diagnoses

TraceML emits a sorted list of `DiagnosticIssue` findings or states per final
summary section. `issues[0]` is the primary diagnosis, and final-summary JSON
also copies that same item to `diagnosis`.

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
cause is input loading, compute balance, waits, synchronization, or other
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
- `WARMUP`: some data exists, but not enough for a stable diagnosis.
- `BALANCED`: no clear timing bottleneck or rank straggler.
- `STRAGGLER`: one rank has materially slower total step time.
- `INPUT_STRAGGLER`: one rank has materially higher dataloader time.
- `COMPUTE_STRAGGLER`: one rank has materially higher compute time.
- `INPUT_BOUND`: dataloader time dominates the typical step.
- `COMPUTE_BOUND`: forward/backward/optimizer time dominates the typical step.
- `OVERHEAD_HEAVY`: step overhead is a material share of the step.

`OVERHEAD_HEAVY` is not a communication diagnosis. `step_overhead_ms` is
measured overhead inside the traced step:

```text
compute_ms = forward_ms + backward_ms + optimizer_ms
known_step_ms = h2d_ms + compute_ms
traced_step_ms = max(raw_trace_step_wall_ms, known_step_ms)
step_overhead_ms = traced_step_ms - known_step_ms
total_step_ms = dataloader_ms + traced_step_ms
```

When input and compute straggler signals appear together, step-time diagnosis
attributes the primary cause to `INPUT_STRAGGLER` if dataloader excess is within
the configured tolerance of compute excess. Otherwise the primary diagnosis
stays `STRAGGLER`, with both input and compute findings preserved as secondary
issues.

It can include validation, checkpointing, logging, framework orchestration, CPU
stalls, unobserved transfer stalls, or other work inside the timed step but
outside the traced H2D and compute phases. It is not proof of GPU idle time,
direct NCCL, all-reduce, or synchronization timing.

Legacy summaries may still contain `WAIT_HEAVY` or `wait_ms`; readers should
treat them as aliases for `OVERHEAD_HEAVY` and `step_overhead_ms`.

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
