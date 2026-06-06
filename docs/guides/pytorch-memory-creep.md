# Find PyTorch memory creep during training

Use this guide when PyTorch GPU memory rises during training, memory headroom
shrinks over time, or a run eventually gets close to out-of-memory behavior.

TraceML step-memory diagnostics separate memory pressure, memory imbalance, and
memory growth over the observed window.

## Run TraceML

Run your training script:

```bash
traceml run train.py
```

Then read the Step Memory section in:

```text
logs/<run_name>/final_summary.txt
logs/<run_name>/final_summary.json
```

## What to look for

The most relevant Step Memory diagnoses are:

| Diagnosis status | Structured kind | Meaning |
|---|---|---|
| `MEMORY CREEP` | `CREEP_CONFIRMED` | memory is rising across the window |
| `MEMORY RISING` | `CREEP_EARLY` | memory is rising from early to recent steps |
| `HIGH PRESSURE` | `HIGH_PRESSURE` | memory is near device capacity |
| `IMBALANCE` | `IMBALANCE` | peak memory differs materially across ranks |

Start with the diagnosis, then inspect the note, worst rank, and memory trend.

## What to check after `MEMORY CREEP`

Common causes to inspect:

- graph-backed tensors retained across steps
- appending `loss`, `logits`, hidden states, or activations to a list without
  detaching them
- caches that grow during training
- step-local state that stays referenced after the step ends
- validation or logging code that stores tensors instead of scalar values

If a worst rank is shown, inspect that rank first.

## What to check after `MEMORY RISING`

`MEMORY RISING` is an early signal. It means the observed window is moving up,
but it is weaker than `MEMORY CREEP`.

Good next steps:

- let the run continue long enough to collect another window
- check whether the same trend appears again
- compare with a known stable run

## Pressure and imbalance are different

`HIGH PRESSURE` means the run is close to device memory capacity. It does not
by itself prove memory is growing.

`IMBALANCE` means one rank is using materially more memory than the typical
rank. Inspect per-rank workload, input shapes, and rank-local branches.

System GPU memory and process GPU memory are useful context, but Step Memory is
the section that diagnoses training-step memory trend.

## Compare a fix

After detaching tensors, clearing caches, changing logging, or reducing memory
load, compare the before and after summaries:

```bash
traceml compare old_run/final_summary.json new_run/final_summary.json
```

Check whether Step Memory diagnosis, peak memory, memory trend, and worst-rank
skew changed.

## Related

- [Find why PyTorch training is slow](slow-pytorch-training.md)
- [How to Read TraceML Output](../user_guide/reading-output.md)
- [Compare Runs](../user_guide/compare.md)
