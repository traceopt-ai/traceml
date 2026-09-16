#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Reproduce the LeRobot v3 image loading regression.

Usage:
  run_reproduction.sh [--pairs 1|2]

The default runs broken then fixed once. --pairs 2 adds a second pair in the
opposite order to expose cache and execution-order effects.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

pairs=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pairs)
      [[ $# -ge 2 ]] || die "--pairs requires 1 or 2"
      pairs="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown argument: $1"
      ;;
  esac
done
[[ "$pairs" == "1" || "$pairs" == "2" ]] || die "--pairs must be 1 or 2"

# Everything that defines the experiment is pinned here.
readonly LEROBOT_REPO="https://github.com/huggingface/lerobot.git"
readonly BROKEN_COMMIT="f6b16f6d97155e3ce34ab2a1ec145e9413588197"
readonly FIXED_COMMIT="67a6c8dfef90351830ad02afc5bf1bd299f9a521"
readonly DATASET_ID="imstevenpmwork/aloha_sim_transfer_cube_human_image"
readonly DATASET_REVISION="13e0d3bff90f02ec417761b58199eb1d33a72efb"
readonly STEPS=200
readonly BATCH_SIZE=8
readonly NUM_WORKERS=4
readonly SEED=1000

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
state_root="$repo_root/logs/lerobot_v3_image_regression"
source_root="$state_root/source/lerobot"
venv_root="$state_root/venv"
cache_root="$state_root/cache"
patch_file="$script_dir/traceml.patch"
requirements_file="$script_dir/requirements.txt"

need() {
  command -v "$1" >/dev/null 2>&1 || die "$1 is required"
}

check_host() {
  [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] ||
    die "this reproduction requires Linux x86-64"
  need git
  need python3.10
  need nvidia-smi
  [[ "$(python3.10 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.10" ]] ||
    die "python3.10 must run Python 3.10"
  python3.10 -m ensurepip --version >/dev/null 2>&1 ||
    die "Python venv support is required; on Ubuntu/Debian install python3.10-venv"
  nvidia-smi >/dev/null || die "nvidia-smi cannot access an NVIDIA GPU"

  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  [[ "$CUDA_VISIBLE_DEVICES" != "-1" && "$CUDA_VISIBLE_DEVICES" != *","* ]] ||
    die "set CUDA_VISIBLE_DEVICES to one GPU"
}

prepare_source() {
  mkdir -p "$(dirname "$source_root")"
  if [[ ! -e "$source_root" ]]; then
    git clone --quiet --no-checkout "$LEROBOT_REPO" "$source_root" ||
      die "could not clone LeRobot; check GitHub access"
  fi
  [[ -d "$source_root/.git" ]] || die "$source_root is not a Git checkout"

  local revision
  for revision in "$BROKEN_COMMIT" "$FIXED_COMMIT"; do
    git -C "$source_root" cat-file -e "${revision}^{commit}" 2>/dev/null ||
      git -C "$source_root" fetch --quiet origin "$revision" ||
      die "could not fetch LeRobot commit $revision"
  done
}

add_worktree() {
  local revision="$1"
  local destination="$2"
  git -C "$source_root" worktree add --quiet --detach "$destination" "$revision" ||
    die "could not create the LeRobot worktree for $revision"
}

instrument() {
  local destination="$1"
  git -C "$destination" apply --check --unidiff-zero "$patch_file" ||
    die "TraceML patch does not apply to $(git -C "$destination" rev-parse HEAD)"
  git -C "$destination" apply --unidiff-zero "$patch_file"
  git -C "$destination" diff --check
}

prepare_environment() {
  if [[ ! -e "$venv_root" ]]; then
    python3.10 -m venv "$venv_root" ||
      die "could not create the Python 3.10 environment"
    "$venv_root/bin/python" -m pip install --upgrade pip setuptools wheel
  elif [[ ! -x "$venv_root/bin/python" ]] ||
    ! "$venv_root/bin/python" -m pip --version >/dev/null 2>&1; then
    die "the reusable environment is incomplete; remove logs/lerobot_v3_image_regression/venv and rerun"
  fi

  local python="$venv_root/bin/python"
  "$python" -m pip install -r "$requirements_file"
  "$python" -m pip install -c "$requirements_file" -e "$broken_tree"
  "$python" -m pip install -c "$requirements_file" -e "$repo_root"
  "$python" - <<'PY' || die "PyTorch cannot access exactly one CUDA GPU"
import torch

assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
PY
}

select_lerobot() {
  local worktree="$1"
  local python="$venv_root/bin/python"

  "$python" -m pip install --quiet --no-deps -e "$worktree" ||
    die "could not activate the selected LeRobot worktree"
  "$python" - "$worktree" <<'PY' || die "LeRobot did not import from the selected worktree"
import sys
from pathlib import Path

import lerobot

active = Path(lerobot.__file__).resolve()
expected = Path(sys.argv[1]).resolve()
raise SystemExit(0 if active.is_relative_to(expected) else 1)
PY
}

record_environment() {
  {
    echo "experiment=$experiment_id"
    echo "lerobot_broken=$BROKEN_COMMIT"
    echo "lerobot_fixed=$FIXED_COMMIT"
    echo "dataset=$DATASET_ID@$DATASET_REVISION"
    echo "traceml=$(git -C "$repo_root" rev-parse HEAD)"
    echo "steps=$STEPS"
    echo "batch_size=$BATCH_SIZE"
    echo "num_workers=$NUM_WORKERS"
    echo "seed=$SEED"
    "$venv_root/bin/python" --version
    "$venv_root/bin/python" - <<'PY'
import os
from importlib.metadata import version
from pathlib import Path

for package in ("accelerate", "datasets", "torch", "torchcodec", "torchvision"):
    print(f"{package}={version(package)}")

cpu_model = "unknown"
for line in Path("/proc/cpuinfo").read_text().splitlines():
    if line.startswith("model name"):
        cpu_model = line.split(":", 1)[1].strip()
        break
print(f"cpu_model={cpu_model}")
print(f"cpu_logical_count={os.cpu_count()}")
PY
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  } > "$experiment_root/environment.txt"
  "$venv_root/bin/python" -m pip list --format=freeze \
    > "$experiment_root/installed-packages.txt"
}

run_revision() {
  local pair="$1"
  local label="$2"
  local worktree="$3"
  local run_name="pair_${pair}_${label}_${experiment_id}"
  local trace_root="$experiment_root/traceml"
  local summary="$trace_root/$run_name/final_summary.json"

  select_lerobot "$worktree"
  echo "Running pair $pair: $label"
  "$venv_root/bin/traceml" run \
    --mode summary \
    --logs-dir "$trace_root" \
    --run-name "$run_name" \
    "$worktree/src/lerobot/scripts/lerobot_train.py" \
    --args \
    "--dataset.repo_id=$DATASET_ID" \
    "--dataset.revision=$DATASET_REVISION" \
    "--policy.type=act" \
    "--policy.device=cuda" \
    "--wandb.enable=true" \
    "--wandb.mode=offline" \
    "--log_freq=2" \
    "--steps=$STEPS" \
    "--batch_size=$BATCH_SIZE" \
    "--num_workers=$NUM_WORKERS" \
    "--seed=$SEED" \
    "--policy.push_to_hub=false" \
    "--policy.repo_id=$DATASET_ID" \
    "--output_dir=$experiment_root/lerobot/$run_name" \
    "--job_name=$run_name" \
    2>&1 | tee "$experiment_root/terminal/${run_name}.log"

  [[ -f "$summary" ]] || die "$label run did not produce $summary"
  RUN_SUMMARY="$summary"
}

check_host

mkdir -p "$state_root/runs" "$cache_root"
export PIP_CACHE_DIR="$cache_root/pip"
export HF_HOME="$cache_root/huggingface"
export TORCH_HOME="$cache_root/torch"
export WANDB_MODE=offline
export WANDB_DISABLE_GIT=true
export WANDB_DISABLE_CODE=true
export WANDB_DISABLE_MACHINE_INFO=true
export WANDB_SILENT=true
export PIP_DISABLE_PIP_VERSION_CHECK=1

started_at="$(date -u +%Y%m%dT%H%M%SZ)"
experiment_root="$(mktemp -d "$state_root/runs/${started_at}_XXXXXX")"
experiment_id="$(basename "$experiment_root")"
broken_tree="$experiment_root/worktrees/broken"
fixed_tree="$experiment_root/worktrees/fixed"

mkdir -p \
  "$experiment_root/compare" \
  "$experiment_root/lerobot" \
  "$experiment_root/terminal" \
  "$experiment_root/traceml" \
  "$experiment_root/wandb" \
  "$experiment_root/worktrees"
export WANDB_DIR="$experiment_root/wandb"

prepare_source
add_worktree "$BROKEN_COMMIT" "$broken_tree"
add_worktree "$FIXED_COMMIT" "$fixed_tree"
instrument "$broken_tree"
instrument "$fixed_tree"
prepare_environment
record_environment

declare -a broken_summaries=()
declare -a fixed_summaries=()
RUN_SUMMARY=""

for ((pair = 1; pair <= pairs; pair++)); do
  if (( pair == 1 )); then
    order=(broken fixed)
  else
    order=(fixed broken)
  fi

  for label in "${order[@]}"; do
    if [[ "$label" == "broken" ]]; then
      run_revision "$pair" broken "$broken_tree"
      broken_summaries[$pair]="$RUN_SUMMARY"
    else
      run_revision "$pair" fixed "$fixed_tree"
      fixed_summaries[$pair]="$RUN_SUMMARY"
    fi
  done

  compare_output="$experiment_root/compare/pair_${pair}_broken_vs_fixed"
  "$venv_root/bin/traceml" compare \
    "${broken_summaries[$pair]}" \
    "${fixed_summaries[$pair]}" \
    --output "$compare_output" \
    2>&1 | tee "$experiment_root/terminal/compare_pair_${pair}.log"
done

echo "Results: ${experiment_root#"$repo_root"/}"
