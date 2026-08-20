#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Run the Qwen3-8B LoRA gradient-accumulation matrix on one GPU.

Usage:
  run_matrix.sh [--steps N] [--order forward|reverse] [--repeat N]
                [--max-batch 4|8] [--skip-preflight]

Defaults:
  --steps 500 --order forward --repeat 1 --max-batch 8
EOF
}

steps=500
order="forward"
repeat=1
max_batch=8
run_preflight=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps)
      steps="$2"
      shift 2
      ;;
    --order)
      order="$2"
      shift 2
      ;;
    --repeat)
      repeat="$2"
      shift 2
      ;;
    --max-batch)
      max_batch="$2"
      shift 2
      ;;
    --skip-preflight)
      run_preflight=0
      shift
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

if ! [[ "$steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "--steps must be a positive integer." >&2
  exit 2
fi
if ! [[ "$repeat" =~ ^[1-9][0-9]*$ ]]; then
  echo "--repeat must be a positive integer label." >&2
  exit 2
fi
if [[ "$order" != "forward" && "$order" != "reverse" ]]; then
  echo "--order must be forward or reverse." >&2
  exit 2
fi
if [[ "$max_batch" != "4" && "$max_batch" != "8" ]]; then
  echo "--max-batch must be 4 or 8." >&2
  exit 2
fi
if ! command -v traceml >/dev/null 2>&1; then
  echo "The traceml command is not available in this environment." >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
train_script="$script_dir/train.py"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
experiment_id="${timestamp}_r${repeat}_${order}"
experiment_root="$repo_root/logs/qwen3_8b_lora_ga/$experiment_id"

mkdir -p \
  "$experiment_root/terminal" \
  "$experiment_root/traceml" \
  "$experiment_root/trainer"

{
  echo "experiment_id=$experiment_id"
  echo "utc_started=$timestamp"
  echo "steps=$steps"
  echo "order=$order"
  echo "repeat=$repeat"
  echo "max_batch=$max_batch"
  command -v python
  python --version
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} | tee "$experiment_root/environment.txt"

run_one() {
  local batch_size="$1"
  local accumulation="$2"
  local optimizer_steps="$3"
  local run_name="$4"
  local output_root="$5"
  local trace_root="$6"
  local dataset_samples="${7:-50000}"
  local terminal_log="$experiment_root/terminal/${run_name}.log"

  echo
  echo "Starting $run_name: batch=$batch_size accumulation=$accumulation steps=$optimizer_steps"
  traceml run \
    --mode summary \
    --summary-window-rows "$optimizer_steps" \
    --logs-dir "$trace_root" \
    --run-name "$run_name" \
    "$train_script" \
    --args \
    --batch-size "$batch_size" \
    --max-steps "$optimizer_steps" \
    --dataset-samples "$dataset_samples" \
    --output-root "$output_root" \
    --run-name "$run_name" \
    2>&1 | tee "$terminal_log"

  if grep -q "Timed out waiting for all ranks to report finished" "$terminal_log"; then
    echo "TraceML finalization timed out for $run_name; stopping the matrix." >&2
    return 1
  fi
}

if [[ "$run_preflight" -eq 1 ]]; then
  preflight_batch="$max_batch"
  preflight_accumulation=$((8 / preflight_batch))
  run_one \
    "$preflight_batch" \
    "$preflight_accumulation" \
    5 \
    "preflight_bs${preflight_batch}_ga${preflight_accumulation}" \
    "$experiment_root/preflight/trainer" \
    "$experiment_root/preflight/traceml" \
    512
  echo "Preflight completed. Physical batch $preflight_batch fits."
fi

if [[ "$max_batch" -eq 8 ]]; then
  forward_configs=("1:8" "2:4" "4:2" "8:1")
  reverse_configs=("8:1" "4:2" "2:4" "1:8")
else
  forward_configs=("1:8" "2:4" "4:2")
  reverse_configs=("4:2" "2:4" "1:8")
fi

if [[ "$order" == "forward" ]]; then
  configs=("${forward_configs[@]}")
else
  configs=("${reverse_configs[@]}")
fi

for config in "${configs[@]}"; do
  IFS=: read -r batch_size accumulation <<<"$config"
  run_name="r${repeat}_bs${batch_size}_ga${accumulation}"
  run_one \
    "$batch_size" \
    "$accumulation" \
    "$steps" \
    "$run_name" \
    "$experiment_root/trainer" \
    "$experiment_root/traceml"
done

echo
echo "Completed ${#configs[@]} training runs."
echo "Results: $experiment_root"
