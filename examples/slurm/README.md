# Running TraceML on Slurm

A copy-paste starting point for multi-node, multi-GPU TraceML runs on a
Slurm-managed GPU cluster.

| File | What it is |
|---|---|
| `traceml_ddp.sbatch` | A 2-node sbatch job. Derives TraceML's launch flags from Slurm variables and runs the launcher once per node via `srun`. |
| `launch.sh` | The per-node wrapper `srun` runs on each node. It expands the per-node Slurm variables and calls `traceml run`. |

The full walkthrough, including the network/aggregator model, lives in the
[Slurm guide](../../docs/user_guide/slurm.md).

## Submit

From the TraceML repository root (the scripts use paths relative to the submit
directory):

```bash
sbatch examples/slurm/traceml_ddp.sbatch
```

## What it does

- One `traceml run` launches per node (`--ntasks-per-node=1`). TraceML spawns
  the per-GPU workers itself, like `torchrun`.
- Node 0 (the first host in the allocation) owns the TraceML aggregator; the
  other nodes stream telemetry to it.
- Flags are derived from Slurm:

  | TraceML flag | Source |
  |---|---|
  | `--nnodes` | `$SLURM_NNODES` |
  | `--node-rank` | `$SLURM_NODEID` |
  | `--nproc-per-node` | `$SLURM_GPUS_ON_NODE` |
  | `--master-addr` | first host of `$SLURM_JOB_NODELIST` |

## Adapt it

- **More/fewer nodes:** change `#SBATCH --nodes`.
- **GPUs per node:** change `#SBATCH --gres=gpu:N` (some clusters use
  `--gpus-per-node=N`). `--nproc-per-node` follows automatically via
  `$SLURM_GPUS_ON_NODE`.
- **Your own script:** edit the script path in `launch.sh`.
- **Environment:** uncomment the `module load` / `conda activate` lines in
  `traceml_ddp.sbatch`, before the `srun` line.

## Where results land

Node 0 writes the run report to:

```
<logs-dir>/<run-name>/final_summary.json
<logs-dir>/<run-name>/final_summary.txt
```

`--logs-dir` defaults to `./logs` relative to the submit directory. Put it on a
shared filesystem so the summary is reachable from anywhere. See the
[Slurm guide](../../docs/user_guide/slurm.md) for details.
