# Running on Slurm

This guide shows the recommended way to run TraceML on a Slurm-managed GPU
cluster, and how to derive TraceML's launch flags from the Slurm environment.

If you are new to multi-node TraceML, read
[Distributed Training](distributed-training.md) first. This page focuses on the
Slurm-specific glue.

A ready-to-use template lives in
[`examples/slurm/`](https://github.com/traceopt-ai/traceml/tree/main/examples/slurm):
`traceml_ddp.sbatch` (the job) and `launch.sh` (the per-node wrapper).

## Mental model

TraceML's launcher wraps `torchrun` (`torch.distributed.run`) and adds a
telemetry **aggregator**:

- Run **one `traceml run` per node**. TraceML spawns the per-GPU workers itself
  (one per GPU), so you do **not** start one task per GPU under Slurm.
- **Node 0 owns the aggregator.** It is the first host in the allocation and
  also serves as the `torchrun` rendezvous master. Every other node streams its
  telemetry to node 0.
- Use **`--mode=summary`** for multi-node runs. The live `cli` and `dashboard`
  modes are intended for single-node use.

## Map Slurm variables to TraceML flags

| TraceML flag | Slurm source | Notes |
|---|---|---|
| `--nnodes` | `$SLURM_NNODES` | Number of nodes in the allocation. |
| `--node-rank` | `$SLURM_NODEID` | Per-node rank `0..N-1`. Valid when you run **one task per node**. |
| `--nproc-per-node` | `$SLURM_GPUS_ON_NODE` | GPUs allocated on this node → one worker per GPU. |
| `--master-addr` | first host of `$SLURM_JOB_NODELIST` | `scontrol show hostnames "$SLURM_JOB_NODELIST" \| head -n 1`. |
| `--run-name` | e.g. `ddp-$SLURM_JOB_ID` | Required for multi-node; must match on every node. |

!!! note "Why `SLURM_GPUS_ON_NODE`?"
    Prefer `SLURM_GPUS_ON_NODE` over `SLURM_GPUS_PER_NODE`. The latter is only
    set when you request GPUs with `--gpus*` flags and can carry a type prefix
    (for example `a100:4`), which is not a plain integer. `SLURM_GPUS_ON_NODE`
    is always an integer count of the GPUs on the node.

    For CPU-only or task-based allocations, use `$SLURM_NTASKS_PER_NODE`
    instead as the worker count per node.

## Deriving the master address

`--master-addr` (and the aggregator that workers connect to) must be node 0.
Resolve it from the allocation:

```bash
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
```

If the hostname is not routable between nodes on your cluster, resolve it to an
IP address instead:

```bash
export MASTER_ADDR="$(srun --nodes=1 --ntasks=1 -w "$MASTER_ADDR" hostname --ip-address)"
```

## The per-node expansion gotcha

TraceML uses **static rendezvous**: each node is told its rank with
`--node-rank`. There is no automatic rank election, so every node must compute
its own rank.

`$SLURM_NODEID` is set per node, but it is only expanded correctly if expansion
happens **on each node**. This does **not** work:

```bash
# WRONG: the batch shell on node 0 expands $SLURM_NODEID once (to 0) for every
# task, so all nodes claim rank 0 and the run never rendezvous.
srun traceml run train.py --node-rank=$SLURM_NODEID ...
```

Instead, put the command in a small wrapper script that `srun` runs on each
node, so the variable is expanded per node:

```bash
# traceml_ddp.sbatch
srun examples/slurm/launch.sh
```

```bash
# launch.sh (runs once per node)
exec traceml run examples/ddp_minimal.py \
  --mode=summary \
  --run-name="${RUN_NAME}" \
  --nnodes="${SLURM_NNODES}" \
  --node-rank="${SLURM_NODEID}" \
  --nproc-per-node="${SLURM_GPUS_ON_NODE}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port=29500
```

`MASTER_ADDR` and `RUN_NAME` are identical on every node, so the sbatch file
exports them once and Slurm propagates them to the wrapper.

## The network and aggregator model

A multi-node TraceML run uses **two** TCP endpoints, both on node 0:

| Purpose | Default port | Who connects |
|---|---|---|
| `torchrun` rendezvous (`--master-port`) | `29500` | All ranks reach node 0 to set up the process group. |
| TraceML aggregator (`--aggregator-port`) | `29765` | Worker nodes stream telemetry to node 0. |

- Node 0 starts the aggregator and, for multi-node runs, **binds it to
  `0.0.0.0`** by default so other nodes can connect.
- Worker nodes connect to the aggregator at `--master-addr:29765` by default
  (the aggregator's connect host defaults to `--master-addr`).

!!! warning "Firewall"
    Both ports must be reachable on node 0 from the other nodes: the
    `torchrun` master port (`29500`) and the TraceML aggregator port
    (`29765`). On clusters with host firewalls, allow inbound traffic to node 0
    on both.

### When to use `--aggregator-host`

Set `--aggregator-host` when workers should send telemetry to a **different
reachable address** than `--master-addr` — for example when node 0 has a
separate management/data interface or hostname that the other nodes should use
for telemetry. Pass the same value on every node:

```bash
traceml run train.py ... --aggregator-host=node0-data.cluster.local
```

### When to use `--aggregator-bind-host`

Multi-node runs already bind the aggregator to `0.0.0.0`, so you usually do not
need this. Set `--aggregator-bind-host` only to pin the aggregator to a
specific interface on node 0 instead of all interfaces:

```bash
traceml run train.py ... --aggregator-bind-host=10.0.0.5
```

## Minimal PyTorch DDP command

A single node's launch (what the wrapper runs) looks like this:

```bash
traceml run examples/ddp_minimal.py \
  --mode=summary \
  --run-name=ddp-demo \
  --nnodes=2 \
  --node-rank=0 \
  --nproc-per-node=4 \
  --master-addr=node0 \
  --master-port=29500
```

`examples/ddp_minimal.py` is a runnable DDP script that already calls
`traceml.init(...)` and `traceml.trace_step(...)`.

## Full template

```bash title="examples/slurm/traceml_ddp.sbatch"
#!/bin/bash
#SBATCH --job-name=traceml-ddp
#SBATCH --nodes=2                 # number of nodes -> --nnodes
#SBATCH --ntasks-per-node=1       # ONE TraceML launcher per node (it spawns the GPU workers)
#SBATCH --gres=gpu:4              # all GPUs on each node; some clusters use --gpus-per-node=4 instead
#SBATCH --cpus-per-task=16        # CPU cores for dataloading; adjust to your node
#SBATCH --time=00:30:00
#SBATCH --output=traceml-%j.out

set -euo pipefail

# --- Cluster-specific setup (run BEFORE srun) ---
# module load cuda/12.x
# source ~/miniconda3/bin/activate myenv
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export RUN_NAME="ddp-${SLURM_JOB_ID}"

cd "$SLURM_SUBMIT_DIR"
srun examples/slurm/launch.sh
```

Submit it from the repository root:

```bash
sbatch examples/slurm/traceml_ddp.sbatch
```

## Where to find results

Node 0's aggregator writes the run report to:

```text
<logs-dir>/<run-name>/final_summary.json
<logs-dir>/<run-name>/final_summary.txt
```

`--logs-dir` defaults to `./logs` relative to the submit directory. Put it on a
**shared filesystem** (for example your scratch space) so the summary is
reachable after the job ends. You can compare two runs later with
[`traceml compare`](compare.md).

## Cluster-specific notes

!!! tip "Adapt these for your cluster"
    - **Launcher.** This template uses `srun` to start one task per node. Some
      sites wrap jobs differently (`mpirun`, site launcher scripts); the
      requirement is just that **one `traceml run` starts per node** and that
      `$SLURM_NODEID` is expanded per node.
    - **GPU request.** `--gres=gpu:N` and `--gpus-per-node=N` are both common;
      use whichever your cluster accepts. Keep `--ntasks-per-node=1` so the
      single task sees all the node's GPUs.
    - **Environment.** Run `module load` / `conda activate` in the sbatch file
      before `srun`. If `traceml` is not found on the worker nodes, also
      activate the environment inside `launch.sh`.
    - **Shared filesystem.** `launch.sh`, your training script, and `--logs-dir`
      should live on storage visible to every node.
    - **Firewall.** Allow inbound traffic to node 0 on the `torchrun` master
      port (`29500`) and the aggregator port (`29765`).
    - **NCCL.** If multi-node NCCL hangs or fails to connect, pin the network
      interface with `export NCCL_SOCKET_IFNAME=<iface>` and debug with
      `export NCCL_DEBUG=INFO`.
    - **Aggregator startup window.** Node 0 must start its aggregator promptly
      after `srun` launches the tasks. Keep slow setup (`module load`,
      `conda activate`) **before** the `srun` line so the workers do not give up
      waiting for the aggregator.

No extra runtime dependency is required beyond Slurm and your existing TraceML
install.
