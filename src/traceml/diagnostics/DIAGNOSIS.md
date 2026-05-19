# Final Summary Diagnoses

TraceML emits one primary diagnosis per section. `issues` contains the sorted
supporting issues when more than one signal fires.

Diagnoses are conservative. They identify likely bottleneck categories, not a
complete root-cause proof.

## System

Host/node pressure.

- `NO_DATA`: no system telemetry.
- `NORMAL`: no CPU, RAM, or available GPU pressure.
- `VERY_HIGH_GPU_MEMORY`: critical GPU memory pressure.
- `HIGH_GPU_MEMORY`: elevated GPU memory pressure.
- `HIGH_GPU_TEMPERATURE`: elevated GPU temperature.
- `HIGH_GPU_POWER`: elevated GPU power use.
- `HIGH_HOST_MEMORY`: elevated host RAM use.
- `HIGH_CPU`: elevated host CPU use.
- `LOW_GPU_UTILIZATION`: GPU utilization is low while host-side signals suggest
  the GPU may be underfed.

## Process

Traced process pressure.

- `NO_DATA`: no process telemetry.
- `NORMAL`: no process CPU, RSS, or available GPU memory pressure.
- `VERY_HIGH_PROCESS_GPU_MEMORY`: critical process GPU memory pressure.
- `HIGH_PROCESS_GPU_MEMORY`: elevated process GPU memory pressure.
- `GPU_MEMORY_RESERVED_OVERHANG`: reserved GPU memory is much higher than used
  GPU memory.
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
- `WAIT_HEAVY`: unattributed residual time is a material share of the step.

`WAIT_HEAVY` is not a communication diagnosis. `wait_ms` is residual
unattributed step time:

```text
compute_ms = forward_ms + backward_ms + optimizer_ms
traced_step_ms = max(raw_trace_step_wall_ms, compute_ms)
wait_ms = traced_step_ms - compute_ms
total_step_ms = dataloader_ms + traced_step_ms
```

It can include validation, checkpointing, logging, framework orchestration, CPU
stalls, transfer stalls, or other work inside the timed step but outside the
compute phases. It is not direct NCCL, all-reduce, or synchronization timing.

## Step Memory

Per-step peak allocated/reserved memory.

- `NO_DATA`: no usable step-memory data.
- `NO GPU`: no GPU telemetry or CPU-only run.
- `BALANCED`: no clear memory pressure, imbalance, or rising trend.
- `HIGH_PRESSURE`: peak memory is near device capacity.
- `IMBALANCE`: peak memory differs materially across ranks.
- `MEMORY RISING`: memory is rising, but below the confirmed-growth threshold.
- `MEMORY CREEP`: memory is rising and crossed the confirmed-growth threshold.

`HIGH_PRESSURE` requires known GPU capacity. If capacity is unavailable,
TraceML does not guess pressure.
