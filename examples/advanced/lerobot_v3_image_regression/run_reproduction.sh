#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Reproduce and compare the LeRobot v3 image-dataset input regression.

Usage:
  run_reproduction.sh [--pairs 1|2]

Options:
  --pairs 1|2  Number of A/B pairs to run. The default is 1. A second pair
               reverses execution order to reduce order and cache bias.
  -h, --help   Show this help and exit without preparing the environment.
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
      [[ $# -ge 2 ]] || die "--pairs requires 1 or 2."
      pairs="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$pairs" != "1" && "$pairs" != "2" ]]; then
  echo "--pairs must be 1 or 2." >&2
  exit 2
fi

# Reproduction inputs. Keep every revision-sensitive value together.
readonly LEROBOT_REPOSITORY="https://github.com/huggingface/lerobot.git"
readonly BROKEN_COMMIT="f6b16f6d97155e3ce34ab2a1ec145e9413588197"
readonly FIXED_COMMIT="67a6c8dfef90351830ad02afc5bf1bd299f9a521"
readonly EXPECTED_UPSTREAM_CHANGE="src/lerobot/datasets/lerobot_dataset.py"
readonly DATASET_ID="imstevenpmwork/aloha_sim_transfer_cube_human_image"
readonly DATASET_REVISION="13e0d3bff90f02ec417761b58199eb1d33a72efb"
readonly POLICY_TYPE="act"
readonly TRAINING_STEPS="200"
readonly MINIMUM_FREE_KIB=$((10 * 1024 * 1024))

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
state_root="$repo_root/logs/lerobot_v3_image_regression"
source_root="$state_root/source/lerobot"
venv_root="$state_root/venv"
cache_root="$state_root/cache"
requirements_file="$script_dir/requirements.txt"
patch_file="$script_dir/traceml.patch"

on_error() {
  local status=$?
  echo "ERROR: reproduction failed at line $1 (exit $status)." >&2
  exit "$status"
}
trap 'on_error "$LINENO"' ERR

require_command() {
  local name="$1"
  local guidance="$2"
  command -v "$name" >/dev/null 2>&1 || die "$name is required. $guidance"
}

check_prerequisites() {
  [[ "$(uname -s)" == "Linux" ]] ||
    die "This reproduction supports Linux x86-64 only."
  [[ "$(uname -m)" == "x86_64" ]] ||
    die "This reproduction supports Linux x86-64 only."

  require_command git "Install Git and rerun this command."
  require_command python3.10 "Install Python 3.10 and rerun this command."
  require_command nvidia-smi \
    "Install a working NVIDIA driver with CUDA 12.8 support."

  local python_version
  python_version="$(python3.10 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  [[ "$python_version" == "3.10" ]] ||
    die "python3.10 resolved to Python $python_version; Python 3.10 is required."

  local smi_output
  if ! smi_output="$(nvidia-smi 2>&1)"; then
    die "nvidia-smi could not communicate with an NVIDIA GPU. Check the driver and GPU access."
  fi

  local cuda_version
  cuda_version="$(sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' <<<"$smi_output" | head -n 1)"
  [[ -n "$cuda_version" ]] ||
    die "nvidia-smi did not report a supported CUDA version. CUDA 12.8 or newer is required."

  local cuda_major="${cuda_version%%.*}"
  local cuda_minor="${cuda_version#*.}"
  cuda_minor="${cuda_minor%%.*}"
  if (( cuda_major < 12 || (cuda_major == 12 && cuda_minor < 8) )); then
    die "The NVIDIA driver reports CUDA $cuda_version; CUDA 12.8 or newer is required."
  fi

  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=0
  elif [[ "$CUDA_VISIBLE_DEVICES" == "-1" || "$CUDA_VISIBLE_DEVICES" == *","* ]]; then
    die "Set CUDA_VISIBLE_DEVICES to exactly one GPU before running this single-GPU reproduction."
  fi

  local free_kib
  free_kib="$(df -Pk "$repo_root" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] ||
    die "Could not determine free disk space for $repo_root."
  (( free_kib >= MINIMUM_FREE_KIB )) ||
    die "At least 10 GiB of free disk space is required before downloading dependencies and data."
}

prepare_source() {
  mkdir -p "$(dirname "$source_root")"

  if [[ -e "$source_root" ]]; then
    [[ -d "$source_root/.git" ]] ||
      die "$source_root exists but is not a LeRobot Git checkout. Move it aside and rerun."
    local origin
    origin="$(git -C "$source_root" remote get-url origin)"
    [[ "$origin" == "$LEROBOT_REPOSITORY" ]] ||
      die "$source_root has an unexpected Git origin. Move it aside and rerun."
  else
    if ! git clone --quiet --no-checkout "$LEROBOT_REPOSITORY" "$source_root"; then
      die "Could not clone LeRobot. Check network access to GitHub and rerun."
    fi
  fi

  local revision
  for revision in "$BROKEN_COMMIT" "$FIXED_COMMIT"; do
    if ! git -C "$source_root" cat-file -e "${revision}^{commit}" 2>/dev/null; then
      if ! git -C "$source_root" fetch --quiet origin "$revision"; then
        die "Could not fetch pinned LeRobot commit $revision. Check network access to GitHub."
      fi
    fi
  done

  local changed_paths
  changed_paths="$(git -C "$source_root" diff --name-only "$BROKEN_COMMIT" "$FIXED_COMMIT")"
  if [[ "$changed_paths" != "$EXPECTED_UPSTREAM_CHANGE" ]]; then
    echo "Observed upstream paths:" >&2
    echo "$changed_paths" >&2
    die "The pinned LeRobot commits no longer differ only in $EXPECTED_UPSTREAM_CHANGE."
  fi
}

add_worktree() {
  local revision="$1"
  local destination="$2"

  [[ ! -e "$destination" ]] ||
    die "Refusing to reuse existing worktree path $destination."
  if ! git -C "$source_root" worktree add --quiet --detach "$destination" "$revision"; then
    die "Could not create the detached LeRobot worktree for $revision."
  fi
}

prepare_environment() {
  if [[ ! -x "$venv_root/bin/python" ]]; then
    if ! python3.10 -m venv "$venv_root"; then
      die "Could not create the Python 3.10 virtual environment. Ensure the venv module is installed."
    fi
  fi

  if ! "$venv_root/bin/python" -m pip install --upgrade pip setuptools wheel; then
    die "Could not prepare pip. Check Python package index network access."
  fi
  if ! "$venv_root/bin/python" -m pip install -r "$requirements_file"; then
    die "Could not install the pinned regression stack. Check package index and PyTorch wheel access."
  fi
  if ! "$venv_root/bin/python" -m pip install -e "$broken_tree"; then
    die "Could not install LeRobot dependencies. Check package index network access."
  fi
  if ! "$venv_root/bin/python" -m pip install -e "$repo_root"; then
    die "Could not install TraceML from the current checkout."
  fi

  if ! "$venv_root/bin/python" - "$requirements_file" <<'PY'
from importlib.metadata import version
from pathlib import Path
import sys

expected = {}
for raw_line in Path(sys.argv[1]).read_text().splitlines():
    line = raw_line.split("#", 1)[0].strip()
    if "==" in line:
        name, pinned_version = line.split("==", 1)
        expected[name] = pinned_version
observed = {name: version(name) for name in expected}
wrong = {
    name: (expected[name], observed[name])
    for name in expected
    if observed[name] != expected[name]
}
if wrong:
    details = ", ".join(
        f"{name}: expected {want}, found {got}"
        for name, (want, got) in wrong.items()
    )
    raise SystemExit(f"version mismatch: {details}")

import torch

cuda_tag = expected["torch"].rsplit("+cu", 1)[-1]
expected_cuda = f"{cuda_tag[:-1]}.{cuda_tag[-1]}"
if torch.version.cuda != expected_cuda:
    raise SystemExit(
        f"expected a CUDA {expected_cuda} PyTorch build, found {torch.version.cuda!r}"
    )
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")
if torch.cuda.device_count() != 1:
    raise SystemExit(
        f"expected exactly one visible CUDA device, found {torch.cuda.device_count()}"
    )
PY
  then
    die "The reusable environment does not match the pinned stack or cannot access one CUDA GPU."
  fi
}

apply_instrumentation() {
  local destination="$1"

  if ! git -C "$destination" apply --check --unidiff-zero "$patch_file"; then
    die "The TraceML patch does not apply cleanly to $(git -C "$destination" rev-parse HEAD)."
  fi
  git -C "$destination" apply --unidiff-zero "$patch_file"
  git -C "$destination" diff --check
}

select_lerobot() {
  local destination="$1"

  if ! "$venv_root/bin/python" -m pip install --quiet --no-deps -e "$destination"; then
    die "Could not select the LeRobot checkout at $destination."
  fi

  local imported_from
  imported_from="$("$venv_root/bin/python" -c 'from pathlib import Path; import lerobot; print(Path(lerobot.__file__).resolve())')"
  [[ "$imported_from" == "$destination"/* ]] ||
    die "LeRobot imported from an unexpected checkout. Expected $destination, found $imported_from."
}

write_environment_record() {
  local output_file="$1"
  {
    echo "experiment_id=$experiment_id"
    echo "utc_started=$started_at"
    echo "platform=$(uname -srm)"
    echo "python=$($venv_root/bin/python --version 2>&1)"
    echo "lerobot_broken_commit=$BROKEN_COMMIT"
    echo "lerobot_fixed_commit=$FIXED_COMMIT"
    echo "dataset=$DATASET_ID"
    echo "dataset_revision=$DATASET_REVISION"
    echo "policy=$POLICY_TYPE"
    echo "steps=$TRAINING_STEPS"
    echo "pairs=$pairs"
    echo "traceml_commit=$(git -C "$repo_root" rev-parse HEAD)"
    "$venv_root/bin/python" - <<'PY'
from importlib.metadata import version

for name in ("accelerate", "datasets", "torch", "torchcodec", "torchvision"):
    print(f"{name}={version(name)}")
PY
    nvidia-smi \
      --query-gpu=name,memory.total,driver_version \
      --format=csv,noheader,nounits |
      sed 's/^/gpu=/'
  } | tee "$output_file"
}

run_revision() {
  local pair_number="$1"
  local label="$2"
  local destination="$3"
  local run_name="pair_${pair_number}_${label}_${experiment_id}"
  local trace_root="$experiment_root/traceml"
  local trainer_output="$experiment_root/lerobot/$run_name"
  local terminal_log="$experiment_root/terminal/${run_name}.log"
  local summary_file="$trace_root/$run_name/final_summary.json"

  select_lerobot "$destination"

  echo
  echo "Starting pair $pair_number, $label revision."
  if ! "$venv_root/bin/traceml" run \
    --mode summary \
    --logs-dir "$trace_root" \
    --run-name "$run_name" \
    "$destination/src/lerobot/scripts/lerobot_train.py" \
    --args \
    "--dataset.repo_id=$DATASET_ID" \
    "--dataset.revision=$DATASET_REVISION" \
    "--policy.type=$POLICY_TYPE" \
    "--policy.device=cuda" \
    "--wandb.enable=true" \
    "--wandb.mode=offline" \
    "--log_freq=2" \
    "--steps=$TRAINING_STEPS" \
    "--policy.push_to_hub=false" \
    "--policy.repo_id=$DATASET_ID" \
    "--output_dir=$trainer_output" \
    "--job_name=$run_name" \
    2>&1 | tee "$terminal_log"
  then
    die "The $label run failed. See its terminal log; for download failures, verify access to the public Hugging Face dataset."
  fi

  [[ -f "$summary_file" ]] ||
    die "The $label run completed without $summary_file. Inspect $terminal_log."
  RUN_SUMMARY="$summary_file"
}

check_prerequisites

mkdir -p "$state_root/runs" "$cache_root" "$state_root/source"
export PIP_CACHE_DIR="$cache_root/pip"
export HF_HOME="$cache_root/huggingface"
export TORCH_HOME="$cache_root/torch"
export WANDB_MODE=offline
export WANDB_DISABLE_GIT=true
export WANDB_DISABLE_CODE=true
export WANDB_DISABLE_MACHINE_INFO=true
export WANDB_ANONYMOUS=must
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

if ! cmp -s \
  "$broken_tree/src/lerobot/scripts/lerobot_train.py" \
  "$fixed_tree/src/lerobot/scripts/lerobot_train.py"
then
  die "The unpatched training scripts differ across the pinned LeRobot commits."
fi

prepare_environment
apply_instrumentation "$broken_tree"
apply_instrumentation "$fixed_tree"

if ! cmp -s \
  "$broken_tree/src/lerobot/scripts/lerobot_train.py" \
  "$fixed_tree/src/lerobot/scripts/lerobot_train.py"
then
  die "The shared TraceML patch did not produce identical training scripts."
fi

write_environment_record "$experiment_root/environment.txt"

declare -a broken_summaries
declare -a fixed_summaries
RUN_SUMMARY=""

for ((pair_number = 1; pair_number <= pairs; pair_number++)); do
  if (( pair_number % 2 == 1 )); then
    order=(broken fixed)
  else
    order=(fixed broken)
  fi

  for label in "${order[@]}"; do
    if [[ "$label" == "broken" ]]; then
      run_revision "$pair_number" "$label" "$broken_tree"
      broken_summaries[$pair_number]="$RUN_SUMMARY"
    else
      run_revision "$pair_number" "$label" "$fixed_tree"
      fixed_summaries[$pair_number]="$RUN_SUMMARY"
    fi
  done

  compare_base="$experiment_root/compare/pair_${pair_number}_broken_vs_fixed"
  if ! "$venv_root/bin/traceml" compare \
    "${broken_summaries[$pair_number]}" \
    "${fixed_summaries[$pair_number]}" \
    --output "$compare_base" \
    2>&1 | tee "$experiment_root/terminal/compare_pair_${pair_number}.log"
  then
    die "TraceML could not compare pair $pair_number."
  fi
done

relative_result="${experiment_root#"$repo_root"/}"
echo
echo "Completed $pairs A/B pair(s)."
echo "Results: $relative_result"
