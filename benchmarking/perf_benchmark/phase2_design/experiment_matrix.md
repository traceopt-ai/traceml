# Experiment Matrix

## Required Cells

| Cell | Purpose |
|---|---|
| `never_init` | True baseline |
| `trace_manual` | Runtime + `trace_step` cost |
| `trace_auto` | Default instrumentation cost |

Internal-only diagnostic: `residual_hooks_optimizer_active`.

## Required Timing Modes

| Mode | Use |
|---|---|
| `step` | Headline overhead |
| `phase` | Cost attribution |

## Required Workloads

| Workload | Purpose |
|---|---|
| `tiny_mlp` | Adversarial fixed-overhead case |
| `small_mlp` | Mid-sized synthetic compute |
| `tiny_transformer` | Transformer-shaped distributed case |

## Required Topologies

| Config | Purpose |
|---|---|
| `local_smoke.json` | Fast CPU/GPU plumbing check |
| `single_gpu.json` | Publishable single-GPU baseline |
| `multinode_2x1.json` | Target distributed run |
| `multinode_4x1.json` | Fan-in/skew scaling run |
