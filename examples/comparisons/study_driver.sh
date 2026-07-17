#!/bin/bash
# Profiler-comparison study driver (runs on a single-GPU box).
# Runs 5 configs on the SAME input-bound workload (baseline, num_workers=0):
#   1 bare  2 traceml  3 torch.profiler  4 cProfile  5 combo (profiler+cProfile)
# Captures wall time, artifact bytes, and an independent nvidia-smi GPU-util
# trace per config. nohup-safe: writes ~/study_out/STUDY_DONE at the end.
set -u
# Point PY at the Python that has torch + traceml-ai installed.
PY="${PY:-python3}"

DATA_DIR="${DATA_DIR:-$HOME/data/imagenette2}"
STEPS="${STEPS:-300}"
BATCH="${BATCH:-64}"
ACTIVE="${ACTIVE:-15}"           # torch.profiler active window
OUT="${OUT:-$HOME/study_out}"
H="$HOME/harness"

mkdir -p "$OUT"
cd "$H" || { echo "no harness dir"; exit 1; }
echo "driver start $(date -u +%FT%TZ) DATA_DIR=$DATA_DIR STEPS=$STEPS" \
    | tee "$OUT/driver.log"

# --- environment manifest (once) ---
{
  echo "=== date ==="; date -u +%FT%TZ
  echo "=== nvidia-smi ==="; nvidia-smi
  echo "=== python ==="; $PY -V
  echo "=== torch ==="; $PY -c "import torch;print(torch.__version__, torch.version.cuda)"
  echo "=== traceml ==="; $PY -c "import traceml_ai as t;print(getattr(t,'__version__','?'))"
  echo "=== cpu ==="; nproc; lscpu | grep -E 'Model name|Socket|Core|Thread' || true
} > "$OUT/env_manifest.txt" 2>&1
$PY -m pip freeze > "$OUT/pip_freeze.txt" 2>&1

# util sampler: $1=label -> writes $OUT/util_$1.csv ; returns pid
start_util() {
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
    --format=csv,noheader -l 1 > "$OUT/util_$1.csv" 2>>"$OUT/driver.log" &
  echo $!
}

run_cfg() {  # $1 label ; rest = command
  local label="$1"; shift
  echo ">>> [$label] $(date -u +%FT%TZ): $*" | tee -a "$OUT/driver.log"
  local upid; upid=$(start_util "$label")
  local t0 t1
  t0=$(date +%s.%N)
  ( "$@" ) > "$OUT/${label}_stdout.log" 2>&1
  local ec=$?
  t1=$(date +%s.%N)
  kill "$upid" 2>/dev/null
  local wall; wall=$(awk "BEGIN{printf \"%.3f\", $t1 - $t0}")
  echo "$ec" > "$OUT/${label}.exit"
  echo "${label},wall_s,${wall},exit,${ec}" | tee -a "$OUT/wall.csv"
  echo "<<< [$label] wall_s=$wall exit=$ec" | tee -a "$OUT/driver.log"
}

# 1) BARE: wall-clock reference
run_cfg bare \
  $PY run_bare.py --profile baseline --data-dir "$DATA_DIR" \
  --batch-size "$BATCH" --max-steps "$STEPS" \
  --metrics-out "$OUT/bare_metrics.json"

# 2) TRACEML: always-on, via the real launcher
run_cfg traceml \
  traceml run --mode summary --html-report \
  --logs-dir "$OUT/traceml_logs" --run-name traceml_baseline \
  train_traceml.py --args --profile baseline --data-dir "$DATA_DIR" \
  --batch-size "$BATCH" --max-steps "$STEPS"

# 3) TORCH.PROFILER: windowed one-shot
run_cfg torch_profiler \
  $PY run_torch_profiler.py --profile baseline --data-dir "$DATA_DIR" \
  --batch-size "$BATCH" --active "$ACTIVE" --tag torch_profiler \
  --outdir "$OUT/torch_profiler_out" \
  --metrics-out "$OUT/torch_profiler_metrics.json"

# 4) CPROFILE: whole-run Python function profiling
run_cfg cprofile \
  $PY -m cProfile -o "$OUT/cprofile.prof" run_bare.py \
  --profile baseline --data-dir "$DATA_DIR" --batch-size "$BATCH" \
  --max-steps "$STEPS" --metrics-out "$OUT/cprofile_run_metrics.json"
# dump a human-readable top-30 by cumulative time
$PY -c "
import pstats
p=pstats.Stats('$OUT/cprofile.prof')
p.sort_stats('cumulative')
import sys
with open('$OUT/cprofile_top.txt','w') as f:
    p.stream=f; p.print_stats(30)
" 2>>"$OUT/driver.log"

# 5) COMBO: torch.profiler window, itself run under cProfile
#    (the profiler+cProfile workflow experts commonly use)
run_cfg combo \
  $PY -m cProfile -o "$OUT/combo_cprofile.prof" run_torch_profiler.py \
  --profile baseline --data-dir "$DATA_DIR" --batch-size "$BATCH" \
  --active "$ACTIVE" --tag combo --outdir "$OUT/combo_tp_out" \
  --metrics-out "$OUT/combo_tp_metrics.json"

# --- artifact sizes ---
{
  echo "config,bytes,note"
  echo "bare,0,no monitoring artifact (reference)"
  printf "traceml,%s,logs dir\n" "$(du -sb "$OUT/traceml_logs" 2>/dev/null | cut -f1)"
  printf "torch_profiler,%s,chrome trace + key_averages\n" "$(du -sb "$OUT/torch_profiler_out" 2>/dev/null | cut -f1)"
  printf "cprofile,%s,.prof\n" "$(du -sb "$OUT/cprofile.prof" 2>/dev/null | cut -f1)"
  printf "combo,%s,.prof + trace\n" "$(du -sb "$OUT/combo_cprofile.prof" "$OUT/combo_tp_out" 2>/dev/null | awk '{s+=$1} END{print s}')"
} > "$OUT/artifact_sizes.csv"

echo "driver done $(date -u +%FT%TZ)" | tee -a "$OUT/driver.log"
echo done > "$OUT/STUDY_DONE"
