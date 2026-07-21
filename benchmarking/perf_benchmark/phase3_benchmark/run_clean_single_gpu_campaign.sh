#!/usr/bin/env bash
# Run an interleaved, apples-to-apples single-GPU TraceML overhead campaign.
#
# Default: 10 fresh repeats each of never_init, trace_manual, and trace_auto.
# Results are grouped under tmp/ and are directly consumable by
# aggregate_results.py. The GIL probe is deliberately omitted: train_worker
# defaults it to false unless --gil-probe is explicitly supplied.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WORKER="${SCRIPT_DIR}/train_worker.py"
AGGREGATE="${SCRIPT_DIR}/aggregate_results.py"

REPEATS=10
BATCH_SIZE=256
MODEL=tiny_mlp
OUTPUT_ROOT=""
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'EOF'
Usage: run_clean_single_gpu_campaign.sh [options]

Options:
  --repeats N       Independent repeats per cell (default: 10)
  --batch-size N    Per-rank batch size (default: 256)
  --model NAME      Benchmark model (default: tiny_mlp)
  --output-root DIR Campaign output directory (default: tmp/clean_single_gpu_<UTC timestamp>)
  -h, --help        Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repeats)
      REPEATS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
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

if ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--repeats must be a positive integer" >&2
  exit 2
fi
if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "--batch-size must be a positive integer" >&2
  exit 2
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="${REPO_ROOT}/tmp/clean_single_gpu_$(date -u +%Y%m%dT%H%M%SZ)"
elif [[ "$OUTPUT_ROOT" != /* ]]; then
  OUTPUT_ROOT="${REPO_ROOT}/${OUTPUT_ROOT}"
fi

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to reuse existing output directory: $OUTPUT_ROOT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/runs" "$OUTPUT_ROOT/traceml-logs"

cat >"$OUTPUT_ROOT/README.txt" <<EOF
Clean single-GPU TraceML overhead campaign

Cells: never_init (launcher disabled), trace_manual, trace_auto
Timing mode: step (one CUDA synchronize before and after each full step)
GIL probe: disabled for every worker
Repeats per cell: $REPEATS
Model: $MODEL
Per-rank batch size: $BATCH_SIZE
Run order: rotated every repeat to limit time-of-run bias
EOF

PYTHONPATH_VALUE="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
LAUNCH_MAIN='from traceml_ai.launcher.cli import main; main()'

run_cell() {
  local cell="$1"
  local repeat="$2"
  local seed="$3"
  local run_dir="${OUTPUT_ROOT}/runs/${cell}_step_${MODEL}_synthetic/repeat_$(printf '%02d' "$repeat")"
  local run_name="clean_$(printf 'r%02d' "$repeat")_${cell}"
  local trace_mode
  local -a launcher_args

  case "$cell" in
    never_init)
      trace_mode=never_init
      launcher_args=(--disable-traceml)
      ;;
    trace_manual)
      trace_mode=trace_manual
      launcher_args=()
      ;;
    trace_auto)
      trace_mode=trace_auto
      launcher_args=()
      ;;
    *)
      echo "Unknown cell: $cell" >&2
      exit 2
      ;;
  esac

  mkdir -p "$run_dir"
  echo "[clean-campaign] repeat=${repeat} cell=${cell}"
  (
    cd "$REPO_ROOT"
    env PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" -c "$LAUNCH_MAIN" run "$WORKER" \
      --mode summary \
      --logs-dir "$OUTPUT_ROOT/traceml-logs" \
      --run-name "$run_name" \
      --nproc-per-node 1 \
      --nnodes 1 \
      --node-rank 0 \
      --master-addr 127.0.0.1 \
      --master-port 29500 \
      --aggregator-port 29765 \
      --finalize-timeout-sec 300 \
      --summary-window-rows 2000 \
      --interval 1 \
      --no-dashboard-auto-open \
      "${launcher_args[@]}" \
      --args \
        --output-dir "$run_dir" \
        --cell-name "$cell" \
        --trace-mode "$trace_mode" \
        --timing-mode step \
        --steps 1000 \
        --warmup 100 \
        --model "$MODEL" \
        --batch-size "$BATCH_SIZE" \
        --dataloader synthetic \
        --device cuda \
        --require-cuda \
        --pin-memory \
        --seed "$seed" \
        --collector-interval-sec 1
  ) 2>&1 | tee "$run_dir/launcher.log"

  env PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" -c '
import json
import sys

path, expected_cell, expected_mode = sys.argv[1:]
payload = json.load(open(path, encoding="utf-8"))
assert payload["cell_name"] == expected_cell, payload["cell_name"]
assert payload["args"]["trace_mode"] == expected_mode
assert payload["args"]["timing_mode"] == "step"
assert payload["args"]["gil_probe"] is False
assert payload["rank"]["world_size"] == 1
assert payload["phase_stats_measured"]["total_step_ms"]["n"] == 1000
median = payload["phase_stats_measured"]["total_step_ms"]["median_ms"]
print("[clean-campaign] median_ms={:.6f}".format(median))
' "$run_dir/rank_0.json" "$cell" "$trace_mode"
}

orders=(
  "never_init trace_manual trace_auto"
  "trace_auto never_init trace_manual"
  "trace_manual trace_auto never_init"
)

for ((repeat = 1; repeat <= REPEATS; repeat++)); do
  seed=$((1337 + (repeat - 1) * 7919))
  read -r -a order <<<"${orders[$(((repeat - 1) % ${#orders[@]}))]}"
  for cell in "${order[@]}"; do
    run_cell "$cell" "$repeat" "$seed"
  done
done

env PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$AGGREGATE" \
  --results-dir "$OUTPUT_ROOT"

echo "[clean-campaign] complete: $OUTPUT_ROOT"
echo "[clean-campaign] report: $OUTPUT_ROOT/report.md"
