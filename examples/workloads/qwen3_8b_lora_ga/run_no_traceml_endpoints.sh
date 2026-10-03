#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Run the 1x8 and 8x1 Qwen3-8B LoRA endpoints without TraceML.

Usage:
  run_no_traceml_endpoints.sh [--steps N] [--order forward|reverse]

Defaults:
  --steps 500 --order reverse

Reverse order is the default because the TraceML measurements ran from the
smallest physical batch to the largest.
EOF
}

steps=500
order="reverse"

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
if [[ "$order" != "forward" && "$order" != "reverse" ]]; then
  echo "--order must be forward or reverse." >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
train_script="$script_dir/train_no_traceml.py"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
experiment_root="$repo_root/logs/qwen3_8b_lora_ga/no_traceml/$timestamp"

mkdir -p "$experiment_root/terminal" "$experiment_root/trainer"

{
  echo "utc_started=$timestamp"
  echo "steps=$steps"
  echo "order=$order"
  command -v python
  python --version
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} | tee "$experiment_root/environment.txt"

run_one() {
  local batch_size="$1"
  local accumulation=$((8 / batch_size))
  local run_name="baseline_bs${batch_size}_ga${accumulation}"
  local terminal_log="$experiment_root/terminal/${run_name}.log"

  echo
  echo "Starting $run_name: batch=$batch_size accumulation=$accumulation steps=$steps"
  PYTHONUNBUFFERED=1 python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    "$train_script" \
    --batch-size "$batch_size" \
    --max-steps "$steps" \
    --dataset-samples 50000 \
    --output-root "$experiment_root/trainer" \
    --run-name "$run_name" \
    2>&1 | tee "$terminal_log"
}

if [[ "$order" == "forward" ]]; then
  configs=(1 8)
else
  configs=(8 1)
fi

for batch_size in "${configs[@]}"; do
  run_one "$batch_size"
done

echo
echo "Completed both no-TraceML endpoint runs."
echo "Results: $experiment_root"
