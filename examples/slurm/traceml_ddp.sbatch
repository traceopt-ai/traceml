#!/bin/bash
#
# Minimal 2-node, multi-GPU TraceML run on a Slurm-managed cluster.
#
# Submit from the TraceML repository root:
#
#     sbatch examples/slurm/traceml_ddp.sbatch
#
# It launches one `traceml run` per node (TraceML spawns the per-GPU workers
# itself, like torchrun). Node 0 owns the TraceML aggregator; the other nodes
# stream telemetry to it. The final report lands in
# <logs-dir>/<run-name>/final_summary.json on node 0.

#SBATCH --job-name=traceml-ddp
#SBATCH --nodes=2                 # number of nodes -> --nnodes
#SBATCH --ntasks-per-node=1       # ONE TraceML launcher per node (it spawns the GPU workers)
#SBATCH --gres=gpu:4              # all GPUs on each node; some clusters use --gpus-per-node=4 instead
#SBATCH --cpus-per-task=16        # CPU cores for dataloading; adjust to your node
#SBATCH --time=00:30:00
#SBATCH --output=traceml-%j.out   # combined stdout/stderr; %j = job id

set -euo pipefail

# --- Cluster-specific setup -------------------------------------------------
# Do all environment setup BEFORE `srun`. Node 0's aggregator must start within
# ~15s of the workers, so avoid slow activation between srun launching the
# tasks and TraceML binding the aggregator. Edit these for your cluster.
#
# module load cuda/12.x
# source ~/miniconda3/bin/activate myenv
#
# Pin NCCL to the right network interface if multi-node NCCL hangs/fails:
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
# ----------------------------------------------------------------------------

# Node 0 = the first host in the allocation. It owns the TraceML aggregator,
# and torchrun uses it as the rendezvous master.
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"

# If the hostname is not routable between nodes, resolve it to an IP instead:
# export MASTER_ADDR="$(srun --nodes=1 --ntasks=1 -w "$MASTER_ADDR" hostname --ip-address)"

# --run-name is required for multi-node runs so every node writes into the same
# TraceML run. Tagging it with the job id keeps concurrent jobs from colliding.
export RUN_NAME="ddp-${SLURM_JOB_ID}"

# Run from the submit directory; launch.sh and the training script paths are
# relative to it.
cd "$SLURM_SUBMIT_DIR"

# srun launches launch.sh once per node. The wrapper expands the per-node Slurm
# variables ($SLURM_NODEID, $SLURM_NNODES, $SLURM_GPUS_ON_NODE) on each node.
srun examples/slurm/launch.sh
