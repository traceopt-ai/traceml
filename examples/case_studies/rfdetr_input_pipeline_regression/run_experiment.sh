#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Reproduce the RF-DETR non-JPEG input-pipeline regression.

Usage:
  run_experiment.sh [--image-format png|bmp]

The experiment runs RF-DETR 1.10.1, 1.11.0 and 1.11.1 in three alternating
orders. Every release receives one native and one TraceML-instrumented run per
repeat. PNG is the default; BMP requires substantially more disk space.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

image_format="png"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image-format)
      [[ $# -ge 2 ]] || die "--image-format requires png or bmp"
      image_format="$2"
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
[[ "$image_format" == "png" || "$image_format" == "bmp" ]] ||
  die "--image-format must be png or bmp"

readonly versions=("1.10.1" "1.11.0" "1.11.1")
readonly image_size=4096
readonly train_images=32
readonly val_images=8
readonly steps=50
readonly warmup_steps=10
readonly batch_size=4
readonly num_workers=0
readonly seed=1544

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
state_root="$repo_root/logs/rfdetr_input_pipeline_regression"
venv_root="$state_root/venv"
wheel_root="$state_root/wheels"
cache_root="$state_root/cache"
dataset_root="$state_root/dataset-${image_format}-${image_size}"
requirements_file="$script_dir/requirements.txt"

need() {
  command -v "$1" >/dev/null 2>&1 || die "$1 is required"
}

check_host() {
  [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] ||
    die "this reproduction requires Linux x86-64"
  need git
  need python3.11
  need nvidia-smi
  [[ "$(python3.11 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.11" ]] ||
    die "python3.11 must run Python 3.11"
  python3.11 -m ensurepip --version >/dev/null 2>&1 ||
    die "Python venv support is required"
  nvidia-smi >/dev/null || die "nvidia-smi cannot access an NVIDIA GPU"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  [[ "$CUDA_VISIBLE_DEVICES" != "-1" && "$CUDA_VISIBLE_DEVICES" != *","* ]] ||
    die "set CUDA_VISIBLE_DEVICES to exactly one GPU"
}

prepare_environment() {
  mkdir -p "$state_root" "$wheel_root" "$cache_root"
  export PIP_CACHE_DIR="$cache_root/pip"
  export TORCH_HOME="$cache_root/torch"
  export HF_HOME="$cache_root/huggingface"
  export PIP_DISABLE_PIP_VERSION_CHECK=1

  if [[ ! -e "$venv_root" ]]; then
    python3.11 -m venv "$venv_root" || die "could not create Python environment"
    "$venv_root/bin/python" -m pip install --upgrade pip setuptools wheel
  fi
  [[ -x "$venv_root/bin/python" ]] ||
    die "the reusable environment is incomplete: $venv_root"
  "$venv_root/bin/python" -m pip install -r "$requirements_file"
  "$venv_root/bin/python" -m pip install 'rfdetr[train]==1.11.1'
  "$venv_root/bin/python" -m pip install -e "$repo_root[lightning]"
  [[ -x "$venv_root/bin/traceml" ]] ||
    die "TraceML was not installed in the reusable environment"
  "$venv_root/bin/python" -m pip check
  "$venv_root/bin/python" - <<'PY' || die "PyTorch cannot access exactly one CUDA GPU"
import torch

assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
PY
}

prepare_wheels() {
  local version wheel
  for version in "${versions[@]}"; do
    wheel="$wheel_root/rfdetr-${version}-py3-none-any.whl"
    if [[ ! -f "$wheel" ]]; then
      "$venv_root/bin/python" -m pip download \
        --no-deps --only-binary=:all: --dest "$wheel_root" "rfdetr==$version"
    fi
    [[ -f "$wheel" ]] || die "missing published wheel for RF-DETR $version"
  done
}

prepare_dataset() {
  if [[ ! -e "$dataset_root" ]]; then
    "$venv_root/bin/python" "$script_dir/generate_dataset.py" \
      --output-dir "$dataset_root" \
      --image-format "$image_format" \
      --image-size "$image_size" \
      --train-images "$train_images" \
      --val-images "$val_images" \
      --seed "$seed"
  fi
  if ! "$venv_root/bin/python" - "$dataset_root" "$image_format" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {
    "image_format": sys.argv[2],
    "image_size": 4096,
    "train_images": 32,
    "val_images": 8,
    "seed": 1544,
}
manifest = json.loads((root / "manifest.json").read_text())
if manifest.get("generator") != expected:
    raise SystemExit(f"dataset does not match the experiment protocol: {manifest.get('generator')}")
PY
  then
    die "existing dataset does not match the protocol"
  fi
}

select_version() {
  local version="$1"
  local wheel="$wheel_root/rfdetr-${version}-py3-none-any.whl"
  "$venv_root/bin/python" -m pip install --quiet --no-deps --force-reinstall "$wheel"
  local installed
  installed="$("$venv_root/bin/python" -c 'from importlib.metadata import version; print(version("rfdetr"))')"
  [[ "$installed" == "$version" ]] ||
    die "selected RF-DETR $version but imported metadata reports $installed"
}

wheel_sha256() {
  "$venv_root/bin/python" - "$1" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
digest = hashlib.sha256()
with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
print(digest.hexdigest())
PY
}

run_measurement() {
  local repeat="$1"
  local version="$2"
  local mode="$3"
  local wheel="$wheel_root/rfdetr-${version}-py3-none-any.whl"
  local digest
  digest="$(wheel_sha256 "$wheel")"
  local output="$experiment_root/runs/repeat-${repeat}/${version}/${mode}"
  local run_name="repeat-${repeat}_rfdetr-${version}_${mode}"
  local disable=()
  if [[ "$mode" == "native" ]]; then
    disable=(--disable-traceml)
  fi

  echo "Running repeat $repeat, RF-DETR $version, $mode"
  "$venv_root/bin/traceml" run \
    --mode summary \
    --logs-dir "$experiment_root/telemetry" \
    --run-name "$run_name" \
    "${disable[@]}" \
    "$script_dir/train.py" \
    --args \
    --dataset-dir "$dataset_root" \
    --output-dir "$output" \
    --expected-rfdetr-version "$version" \
    --rfdetr-wheel-sha256 "$digest" \
    --batch-size "$batch_size" \
    --num-workers "$num_workers" \
    --steps "$steps" \
    --warmup-steps "$warmup_steps" \
    --seed "$seed" \
    2>&1 | tee "$experiment_root/terminal/${run_name}.log"
  [[ -f "$output/result.json" ]] || die "$run_name did not complete"
}

run_repeat() {
  local repeat="$1"
  shift
  local version mode
  for version in "$@"; do
    select_version "$version"
    if [[ "$repeat" == "2" ]]; then
      for mode in traced native; do
        run_measurement "$repeat" "$version" "$mode"
      done
    else
      for mode in native traced; do
        run_measurement "$repeat" "$version" "$mode"
      done
    fi
  done
}

record_protocol() {
  "$venv_root/bin/python" - "$experiment_root" "$dataset_root" "$image_format" "$wheel_root" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

experiment = Path(sys.argv[1])
dataset = Path(sys.argv[2])
image_format = sys.argv[3]
wheels = Path(sys.argv[4])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

payload = {
    "schema_version": 1,
    "started_at": datetime.now(timezone.utc).isoformat(),
    "versions": ["1.10.1", "1.11.0", "1.11.1"],
    "version_orders": [
        ["1.10.1", "1.11.0", "1.11.1"],
        ["1.11.1", "1.11.0", "1.10.1"],
        ["1.11.0", "1.10.1", "1.11.1"],
    ],
    "mode_orders": [["native", "traced"], ["traced", "native"], ["native", "traced"]],
    "dataset": {
        "path": str(dataset),
        "image_format": image_format,
        "manifest_sha256": sha256(dataset / "manifest.json"),
    },
    "wheels": {
        version: sha256(wheels / f"rfdetr-{version}-py3-none-any.whl")
        for version in ("1.10.1", "1.11.0", "1.11.1")
    },
}
(experiment / "protocol.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

check_host
prepare_environment
prepare_wheels
prepare_dataset

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
started_at="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$state_root/experiments"
experiment_root="$(mktemp -d "$state_root/experiments/${started_at}_XXXXXX")"
mkdir -p "$experiment_root/terminal" "$experiment_root/telemetry"
record_protocol

run_repeat 1 "1.10.1" "1.11.0" "1.11.1"
run_repeat 2 "1.11.1" "1.11.0" "1.10.1"
run_repeat 3 "1.11.0" "1.10.1" "1.11.1"

"$venv_root/bin/python" "$script_dir/analyze.py" \
  --experiment-dir "$experiment_root" \
  2>&1 | tee "$experiment_root/terminal/analysis.log"

echo "Experiment complete: $experiment_root"
echo "Report: $experiment_root/analysis/report.md"
