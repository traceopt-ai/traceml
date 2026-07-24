# Phase 2 Methodology

This benchmark uses one external timing harness for TraceML-on and TraceML-off
runs. TraceML is never measured with its own timers.

## Timing Modes

- `step`: headline overhead. Synchronize CUDA once before and once after the
  full training step.
- `phase`: attribution only. Synchronize CUDA around dataloader, H2D, forward,
  backward, optimizer, and TraceML step-boundary work.

## Baselines

- `never_init`: TraceML disabled through the launcher; no TraceML init in user code.
- `trace_manual`: TraceML runtime and `trace_step`, no auto phase patches.
- `trace_auto`: default TraceML instrumentation.

`residual_hooks_optimizer_active` is an internal diagnostic only. It keeps the
optimizer hook active while armed-gated wrappers pass through; it is not a
headline baseline.

## Reporting

Report median, p95, p99, std, absolute overhead, and percent overhead. If a
delta is inside the baseline noise floor, report it as a bound, not zero.

## Distributed Controls

Multi-node runs use fixed ports from config. Every node uses the same config,
same run id, same master address, and matching node rank.

Before every cell, a 120-second config-controlled barrier confirms every node
is on the same case. Multi-node runs reject `--keep-going` and use a fresh run
id, so a failed node cannot silently advance the remaining matrix.

Linux default-route NIC counters are captured per worker. In the target 1-rank
per-node topology this is rank-scoped, but it includes DDP and TraceML traffic;
compare each active cell with `never_init`.
