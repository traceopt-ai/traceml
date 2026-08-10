#!/usr/bin/env bash
# Copyright 2026 OptAI UG (haftungsbeschraenkt)
# SPDX-License-Identifier: Apache-2.0

# Launch the CLIP/DDP reproduction through TraceML. Environment variables
# expose the experiment axes while any additional CLI arguments are forwarded
# to train_clip_ddp.py after --args.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_clip_ddp.py"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
TARGET_MODEL="${TARGET_MODEL:-openai/clip-vit-base-patch32}"
DATASET_SOURCE="${DATASET_SOURCE:-generated}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-128}"
TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-1024}"
MAX_STEPS="${MAX_STEPS:-300}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
ATTENTION_IMPL="${ATTENTION_IMPL:-flash_attention_2}"
TORCH_COMPILE="${TORCH_COMPILE:-1}"
PRECISION="${PRECISION:-bf16}"
DISABLE_TRACEML="${DISABLE_TRACEML:-0}"
DRY_RUN="${DRY_RUN:-0}"
LOGS_DIR="${LOGS_DIR:-${REPO_ROOT}/logs}"
RUN_STAMP="$(date -u +%Y%m%d-%H%M%S)"
RUN_NAME="${RUN_NAME:-clip-ddp-${NPROC_PER_NODE}gpu-${DATASET_SOURCE}-${RUN_STAMP}}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/checkpoints/clip_ddp_l4/${RUN_NAME}}"

# Preserve the environment choices from the public issue.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS="${TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS:-1}"

compile_flag="--torch-compile"
if [[ "${TORCH_COMPILE}" == "0" ]]; then
  compile_flag="--no-torch-compile"
fi

launcher_args=(
  run
  "${TRAIN_SCRIPT}"
  --mode=summary
  "--run-name=${RUN_NAME}"
  "--logs-dir=${LOGS_DIR}"
  "--nproc-per-node=${NPROC_PER_NODE}"
)
if [[ "${DISABLE_TRACEML}" == "1" ]]; then
  launcher_args+=(--disable-traceml)
fi

training_args=(
  "--target-model=${TARGET_MODEL}"
  "--dataset-source=${DATASET_SOURCE}"
  "--per-device-train-batch-size=${PER_DEVICE_BATCH_SIZE}"
  "--target-global-batch-size=${TARGET_GLOBAL_BATCH_SIZE}"
  "--max-steps=${MAX_STEPS}"
  "--warmup-steps=${WARMUP_STEPS}"
  "--num-workers=${NUM_WORKERS}"
  "--prefetch-factor=${PREFETCH_FACTOR}"
  "--attention-impl=${ATTENTION_IMPL}"
  "${compile_flag}"
  "--precision=${PRECISION}"
  "--run-name=${RUN_NAME}"
  "--save-dir=${SAVE_DIR}"
)

cd "${REPO_ROOT}"
echo "CLIP/DDP reproduction: ${RUN_NAME}"
echo "GPUs=${NPROC_PER_NODE} dataset=${DATASET_SOURCE} batch/device=${PER_DEVICE_BATCH_SIZE}"
echo "attention=${ATTENTION_IMPL} compile=${TORCH_COMPILE} TraceML disabled=${DISABLE_TRACEML}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'traceml'
  printf ' %q' "${launcher_args[@]}" --args "${training_args[@]}" "$@"
  printf '\n'
  exit 0
fi

exec traceml "${launcher_args[@]}" --args "${training_args[@]}" "$@"
