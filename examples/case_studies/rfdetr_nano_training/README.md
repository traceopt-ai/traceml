# RF-DETR Nano: where training time goes

**Status: baseline investigation; T4/A100 and four-GPU measurements pending.**

Measure where RF-DETR Nano training time goes using real COCO data and the
native training loop, for [Jiri's issue #1410](https://github.com/roboflow/rf-detr/issues/1410).
Start on one T4; the script also supports A100 and four-GPU DDP.

## Upstream reference

- Pin RF-DETR's `develop` revision
  [`0ed5be8e8d6762c4978a11671cbf34cfc0595e25`](https://github.com/roboflow/rf-detr/tree/0ed5be8e8d6762c4978a11671cbf34cfc0595e25).
  The script rejects another revision or tracked modifications to that checkout.
- [JESUSROYETH's first results](https://github.com/roboflow/rf-detr/issues/1410#issuecomment-5482257434)
  used `develop@6674d858`, synthetic training batches on a laptop 4060 and L4,
  and separate COCO loader measurements. This real-data experiment uses another
  revision and precision, so absolute timings are not directly comparable.
  The reported compile failures were addressed in
  [PR #1411](https://github.com/roboflow/rf-detr/pull/1411) and
  [PR #1436](https://github.com/roboflow/rf-detr/pull/1436), both included in our pin.
- [PR #1468](https://github.com/roboflow/rf-detr/pull/1468), from
  `perf/cuda-graph-training`, merged after our pin. This case measures eager
  training; the TraceML adapter does not support compilation or CUDA graphs.

## Install on the GPU machine

Use Linux, Python 3.11 and a CUDA 12.8-compatible driver. Run from the TraceML
repository root; PyTorch versions follow the
[official installation instructions](https://pytorch.org/get-started/previous-versions/).

```bash
python3.11 -m venv data/rfdetr-case-venv
source data/rfdetr-case-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[lightning]'
python -m pip install 'rfdetr[train] @ git+https://github.com/roboflow/rf-detr.git@0ed5be8e8d6762c4978a11671cbf34cfc0595e25'
python -m pip check
```

Alternatively, install a clean checkout at the same commit with
`python -m pip install -e '../intergate_git/rf-detr[train]'`.
The adapter's warning about versions outside `1.10.1` is expected: this case
pins a specific `1.11.0.dev0` commit, not any checkout with that version string.

RF-DETR downloads pretrained Nano weights before timing starts. Allow the
initial download before going offline.

## Prepare full COCO 2017

Use local storage with roughly 40 GB free for archives and extracted data.
Download the official COCO files below; skip completed downloads/extraction.

```bash
mkdir -p data/coco2017
curl -fL --retry 3 http://images.cocodataset.org/zips/train2017.zip -o data/coco2017/train2017.zip
curl -fL --retry 3 http://images.cocodataset.org/zips/val2017.zip -o data/coco2017/val2017.zip
curl -fL --retry 3 http://images.cocodataset.org/annotations/annotations_trainval2017.zip -o data/coco2017/annotations_trainval2017.zip
unzip -q data/coco2017/train2017.zip -d data/coco2017
unzip -q data/coco2017/val2017.zip -d data/coco2017
unzip -q data/coco2017/annotations_trainval2017.zip -d data/coco2017
```

Expected layout: `train2017/` (118,287 images), `val2017/` (5,000 images), and
`annotations/instances_{train,val}2017.json`. Native setup requires validation
data, but this experiment disables validation.

## Run the single-GPU experiment

Defaults: Nano at 384, `group_detr=13`, batch 4/GPU, accumulation 1,
2 loader workers/rank, seed 42, native fused AdamW and EMA. Resolution is fixed;
`multi_scale`, `expanded_scales` and `do_random_resize_via_padding` are false.
`augmentation_backend="torchvision"` keeps augmentation independent of optional
packages. External loggers and progress bars are disabled.

Precision `auto` selects native BF16 on A100 or FP16 on T4. FP16's gradient
scaler can skip updates, so the 50 steps count optimizer-update **attempts**.

Use **Bash** on an otherwise idle machine. This runs three traced/untraced pairs
in alternating order. Use a new `BATCH` directory for each rerun; keep storage
fixed and record cache conditions.

```bash
set -e
CASE=examples/case_studies/rfdetr_nano_training
DATASET="$PWD/data/coco2017"
BATCH="$PWD/logs/rfdetr-nano/t4-01"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2

for repeat in 1 2 3; do
  modes="baseline traced"
  if [ "$repeat" = 2 ]; then modes="traced baseline"; fi
  for mode in $modes; do
    disable=()
    if [ "$mode" = baseline ]; then disable=(--disable-traceml); fi
    traceml run "$CASE/train.py" --mode summary --nproc-per-node 1 \
      --logs-dir "$BATCH/telemetry" --run-name "pair-$repeat-$mode" \
      "${disable[@]}" --args \
      --dataset-dir "$DATASET" --output-dir "$BATCH/pair-$repeat-$mode"
  done
done

python "$CASE/summarize.py" \
  --pair "$BATCH/pair-1-baseline" "$BATCH/pair-1-traced" \
  --pair "$BATCH/pair-2-baseline" "$BATCH/pair-2-traced" \
  --pair "$BATCH/pair-3-baseline" "$BATCH/pair-3-traced" \
  --output "$BATCH/issue-1410-draft.md"
```

The reporter also accepts one pair. To change settings, pass the same
`--steps`, `--warmup-steps`, `--num-workers`, `--batch-size`, `--precision` or
`--seed` to both modes. Report changed configurations separately.
If input wait is substantial, repeat with `--num-workers 4`, holding everything
else fixed, and compare native throughput and input wait with the two-worker run.

## Four GPUs and profiler traces

For four GPUs, rerun the same paired loop with a new `BATCH`,
`CUDA_VISIBLE_DEVICES=0,1,2,3` and `--nproc-per-node 4`. This uses ordinary
single-node DDP: batch 4/rank means global batch 16, with eight loader workers
total. Report it separately from the single-GPU result; global batch differs.

Run the separate profiler on one GPU after the throughput experiment:

```bash
traceml run "$CASE/train.py" --disable-traceml --nproc-per-node 1 --args \
  --dataset-dir "$DATASET" --output-dir "$BATCH/profiler" --profile

python "$CASE/summarize.py" \
  --pair "$BATCH/pair-1-baseline" "$BATCH/pair-1-traced" \
  --pair "$BATCH/pair-2-baseline" "$BATCH/pair-2-traced" \
  --pair "$BATCH/pair-3-baseline" "$BATCH/pair-3-traced" \
  --profile-dir "$BATCH/profiler" --output "$BATCH/issue-1410-draft.md"
```

The profiler records steps 26–35 (`wait=20, warmup=5, active=10`) in
`profile-rank-0.json` and `profile-summary-rank-0.json`. The summary supplies the
criterion/matcher table; inspect the raw trace in Perfetto for host syncs and
kernel overlap. The table does not count synchronization events.

Keep the same arguments, environment and GPUs as the paired runs; use
`--nproc-per-node 4` for DDP. Profiler runs are excluded from throughput and
TraceML overhead calculations.

## What the report means

- **Wall throughput:** CUDA-synchronized elapsed time between completion of
  step 10 and completion of step 50, including the following input fetches.
  Synchronization occurs only at window boundaries. DDP aligns the start and
  reports the slowest rank. Initialization, downloads, validation, checkpoint
  writing and shutdown are excluded.
- **TraceML phases:** per-rank means for exactly steps 11–50, labeled with the
  CPU/GPU event clock. These intervals differ from end-to-end wall time.
- **Boundaries:** forward is the detector; backward includes DDP communication;
  the optimizer region can include scheduler/EMA callback work. Residual is
  unassigned time, not isolated criterion/matcher time. H2D can overlap other
  intervals. Input wait is exposed waiting, not all worker preprocessing.
- **Profiler scopes:** `rfdetr/criterion_including_matcher` includes
  `rfdetr/matcher` (batched `_match_many` and fallback `forward`). Target-side
  preparation outside matcher calls stays in criterion. CPU totals include
  child operations and host waits; CUDA totals sum attributed kernel work.
  Do not add nested scopes or CPU/CUDA totals.
- **Overhead:** `100 * (traced wall time / native wall time - 1)` for each
  matched pair, plus the median and range. A negative delta can be noise.

`run.json` records full configurations, source revisions, hardware, annotation
and weight checksums; `environment.txt` saves `pip freeze --all`. The report lists
non-default settings and rejects mismatched runs or missing steps/ranks.

## Share the result

Review `issue-1410-draft.md` and add an evidence-based interpretation, including
an inconclusive result if warranted. Post it manually to issue #1410 with the
TraceML commit link and storage/cache conditions. Share the whole `BATCH`
directory (compressed if needed) to preserve metadata, traces and relative paths.

Keep data, weights and raw outputs in ignored `data/` and `logs/`; commit only
the protocol and a small reviewed result. This baseline makes no accuracy claim.
Any later optimization needs before/after mAP on the same evaluation set.

### Validation status

- Passed locally: report validation, reduced Nano CPU smoke training, and native
  matcher tests for both paths, unchanged losses and ten active profiler steps.
- Pending: T4/A100 training loss, complete 40-step GPU telemetry, CUDA profiler
  export and four-rank DDP completion. CPU checks are not performance results.
