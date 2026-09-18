# RF-DETR PR #1489 validation

This experiment compares the commit immediately before RF-DETR PR #1489 with
its merged commit on four T4 GPUs and real COCO train2017 data. It is an
internal investigation, not part of the published case study.

TraceML provides phase, memory and rank attribution. The separate PyTorch
Profiler runs provide the optimizer, EMA, scheduler and explicit collective
kernel measurements. TraceML does not measure NCCL directly.

## Revisions

```bash
BEFORE=1ac74e7a25edf0d771f075a414e86cb8dae085ee
AFTER=5b39cbc0a0ef75ab9c9571ccb28b780c7da51432
RF_REPO="$HOME/rf-detr-source"
RF_WORKTREES="$HOME/rfdetr-1489-worktrees"

git clone https://github.com/roboflow/rf-detr.git "$RF_REPO"
git -C "$RF_REPO" fetch origin
mkdir -p "$RF_WORKTREES"
git -C "$RF_REPO" worktree add --detach "$RF_WORKTREES/before" "$BEFORE"
git -C "$RF_REPO" worktree add --detach "$RF_WORKTREES/after" "$AFTER"
```

Use the existing case-study environment. Both arms use that same environment;
`PYTHONPATH` selects the RF-DETR source without reinstalling dependencies.

```bash
source data/rfdetr-case-venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=2

CASE="$PWD/examples/case_studies/rfdetr_nano_training"
DATASET="$PWD/data/coco2017"
BATCH="$PWD/logs/rfdetr-pr1489-t4-ddp"
mkdir -p "$BATCH"
```

## Smoke runs

```bash
PYTHONPATH="$RF_WORKTREES/before/src" traceml run "$CASE/train.py" \
  --disable-traceml --nproc-per-node 4 --args \
  --dataset-dir "$DATASET" --output-dir "$BATCH/smoke-before" \
  --expected-rfdetr-commit "$BEFORE" --steps 3 --warmup-steps 1

PYTHONPATH="$RF_WORKTREES/after/src" traceml run "$CASE/train.py" \
  --disable-traceml --nproc-per-node 4 --args \
  --dataset-dir "$DATASET" --output-dir "$BATCH/smoke-after" \
  --expected-rfdetr-commit "$AFTER" --steps 3 --warmup-steps 1
```

## Native pairs

Run five pairs in AB, BA, AB, BA, AB order. Every invocation uses a fresh
process group and output directory.

```bash
for pair in 1 2 3 4 5; do
  arms="before after"
  if [ $((pair % 2)) -eq 0 ]; then arms="after before"; fi
  for arm in $arms; do
    if [ "$arm" = before ]; then
      commit="$BEFORE"
    else
      commit="$AFTER"
    fi
    PYTHONPATH="$RF_WORKTREES/$arm/src" traceml run "$CASE/train.py" \
      --disable-traceml --nproc-per-node 4 --args \
      --dataset-dir "$DATASET" \
      --output-dir "$BATCH/native-$pair-$arm" \
      --expected-rfdetr-commit "$commit"
  done
done
```

## TraceML attribution

```bash
for pair in 1 2 3; do
  arms="before after"
  if [ "$pair" = 2 ]; then arms="after before"; fi
  for arm in $arms; do
    if [ "$arm" = before ]; then
      commit="$BEFORE"
    else
      commit="$AFTER"
    fi
    PYTHONPATH="$RF_WORKTREES/$arm/src" traceml run "$CASE/train.py" \
      --mode summary --nproc-per-node 4 \
      --logs-dir "$BATCH/telemetry" \
      --run-name "trace-$pair-$arm" --args \
      --dataset-dir "$DATASET" \
      --output-dir "$BATCH/trace-$pair-$arm" \
      --expected-rfdetr-commit "$commit"
  done
done
```

## PyTorch Profiler

These runs are separate from throughput measurement. Each rank writes its own
trace and summary.

```bash
PYTHONPATH="$RF_WORKTREES/before/src" traceml run "$CASE/train.py" \
  --disable-traceml --nproc-per-node 4 --args \
  --dataset-dir "$DATASET" --output-dir "$BATCH/profile-before" \
  --expected-rfdetr-commit "$BEFORE" --profile

PYTHONPATH="$RF_WORKTREES/after/src" traceml run "$CASE/train.py" \
  --disable-traceml --nproc-per-node 4 --args \
  --dataset-dir "$DATASET" --output-dir "$BATCH/profile-after" \
  --expected-rfdetr-commit "$AFTER" --profile
```

## Report

```bash
python "$CASE/compare_revisions.py" \
  --native-pair "$BATCH/native-1-before" "$BATCH/native-1-after" \
  --native-pair "$BATCH/native-2-before" "$BATCH/native-2-after" \
  --native-pair "$BATCH/native-3-before" "$BATCH/native-3-after" \
  --native-pair "$BATCH/native-4-before" "$BATCH/native-4-after" \
  --native-pair "$BATCH/native-5-before" "$BATCH/native-5-after" \
  --traced-pair "$BATCH/trace-1-before" "$BATCH/trace-1-after" \
  --traced-pair "$BATCH/trace-2-before" "$BATCH/trace-2-after" \
  --traced-pair "$BATCH/trace-3-before" "$BATCH/trace-3-after" \
  --profile-before "$BATCH/profile-before" \
  --profile-after "$BATCH/profile-after" \
  --output "$BATCH/result.md"
```

If the fixed-resolution result is not a consistent improvement, repeat only
the five native pairs in a new batch directory with `--multi-scale`. Keep batch
four per rank. If this OOMs, record the configuration as unsupported instead of
changing the batch size.

Post a clear improvement or regression on RF-DETR PR #1489. Keep an
inconclusive result internal. This experiment does not establish mAP or
convergence parity.
