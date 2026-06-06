#!/bin/bash
#
# Per-node TraceML launcher for Slurm.
#
# srun runs this script once on every node in the allocation (because the
# sbatch template requests one task per node). The Slurm variables below are
# therefore evaluated *on each node*, which is exactly what TraceML's static
# rendezvous needs:
#
#   SLURM_NODEID       -> --node-rank   (0..N-1, unique per node)
#   SLURM_NNODES       -> --nnodes
#   SLURM_GPUS_ON_NODE -> --nproc-per-node (one worker per GPU on this node)
#
# MASTER_ADDR and RUN_NAME are identical on every node, so the sbatch template
# exports them once and Slurm propagates them here.
#
# IMPORTANT: do not inline this command as `srun traceml run ... \
# --node-rank=$SLURM_NODEID` in the sbatch file. There the batch shell would
# expand $SLURM_NODEID once on the first node (to 0) for every task, so all
# nodes would claim node rank 0 and the run would never rendezvous. Keeping the
# command in this wrapper defers expansion to each node.

set -euo pipefail

# Run from the directory the job was submitted from. The paths below are
# relative to it, so submit this job from the TraceML repository root (or edit
# the paths to point at your own training script with an absolute path).
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Optional: activate your environment here as well. Activating it in the sbatch
# template usually propagates via Slurm, but some clusters reset the
# environment per task. Uncomment if `traceml` is not found on the workers.
# source ~/miniconda3/bin/activate myenv

exec traceml run examples/ddp_minimal.py \
  --mode=summary \
  --run-name="${RUN_NAME}" \
  --nnodes="${SLURM_NNODES}" \
  --node-rank="${SLURM_NODEID}" \
  --nproc-per-node="${SLURM_GPUS_ON_NODE}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port=29500
