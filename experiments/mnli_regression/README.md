# MNLI training regression experiment

Fine-tune BERT-base on MNLI and investigate
[Transformers issue #49645](https://github.com/huggingface/transformers/issues/49645).
Its collator copied the batch while checking for empty input.
[PR #49942](https://github.com/huggingface/transformers/pull/49942) removes that
copy. The two pinned revisions differ only by that fix and its upstream test.

The reported CPU/tokenizer improvement has not been reproduced on this BERT
workload. Start with the pilot. Continue only if the change materially affects
training and TraceML helps explain it. A small or absent difference is a valid
result. This is a regression experiment, not a progressive-slowdown reproduction.

The workload follows the standard
[Hugging Face GLUE recipe](https://github.com/huggingface/transformers/blob/v4.57.1/examples/pytorch/text-classification/README.md):
FP16, dynamic padding up to 128 tokens, learning rate `2e-5`, and seed `42`.
Trainer manages DDP, loss logging and evaluation on both MNLI validation splits.
The default pilot uses a seeded 10% subset (about 39,000 examples) for one epoch.

## Prepare the GPU machine

Use Linux, Python 3.11 or 3.12, Git, and working CUDA-enabled PyTorch (2.6 or
newer). Start on one machine with four T4s if available. A100s are not required.
Install the CUDA PyTorch build appropriate for the machine before continuing.
Use the same environment and hardware for both variants.

From the TraceML repository root:

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install datasets==4.8.4 accelerate==1.14.0
python -m pip install 'transformers @ git+https://github.com/huggingface/transformers.git@50e35cc0ca18ec19b80653690de553e6decf74f5'
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())'
```

`train.py` checks the installed Transformers commit against `--variant`, so an
unrelated package cannot silently be labeled affected or fixed. The resolved
package versions and training configuration are saved with each run.

Download and tokenize the data and cache the model before measuring:

```bash
python experiments/mnli_regression/train.py --variant affected --prepare-only
```

Model and dataset revisions are pinned in `train.py`. Hugging Face downloads
them automatically into its normal cache. A 10% training subset can still
require downloading the complete source dataset. Later runs reuse the cache.

## Run the pilot

For a quick wiring check, run the command below with a fresh run name,
`--nproc-per-node=1`, and `--train-fraction 0.002`. Then use the four-GPU pilot.
The smoke check is not performance evidence.

```bash
RUN=mnli-affected-pilot-1
traceml run experiments/mnli_regression/train.py \
  --nproc-per-node=4 --mode=summary \
  --logs-dir=logs --run-name="$RUN" --history-retention=4h \
  --args --variant affected --output-dir "logs/$RUN/training" \
  --train-fraction 0.1 --epochs 1
```

Install the fixed revision into the same environment and repeat:

```bash
python -m pip install --no-deps --force-reinstall 'transformers @ git+https://github.com/huggingface/transformers.git@4ec6e54f0457942a89928097d47e220fcb635450'
RUN=mnli-fixed-pilot-1
traceml run experiments/mnli_regression/train.py \
  --nproc-per-node=4 --mode=summary \
  --logs-dir=logs --run-name="$RUN" --history-retention=4h \
  --args --variant fixed --output-dir "logs/$RUN/training" \
  --train-fraction 0.1 --epochs 1
```

Use a fresh run name every time. Keep the batch size (16 per GPU), workers
(2 per rank), input order, GPU count and other settings identical. Repeat the
pair in reverse order before treating a difference as a reliable result.
Do not reduce workers or disable prefetching just to create a bottleneck.

If the pilot qualifies, use `--train-fraction 1 --epochs 3` with new run names
for both variants. Prepare that fraction first with `--prepare-only
--train-fraction 1`. Training ends after the requested epochs, not a time limit.
Choose retention longer than the whole expected run; `4h` is a starting value,
not a storage bound or a promise about runtime.

For an overhead check, launch a short equivalent pilot using
`torchrun --standalone --nproc-per-node=4` instead of `traceml run`, keeping the
same training arguments and using a separate output directory.

For four separate GPU machines, use the existing
[multi-node launch instructions](../../docs/user_guide/distributed-training.md)
with `--nnodes=4 --nproc-per-node=1`. Prepare the same environment and caches on
every node and use the same absolute checkout/log paths. Node 0 owns the
aggregator database. Network topology is part of the experiment; do not mix
single-node and multi-node results in a paired comparison.

## Saved results

TraceML creates the run folder before starting the script. The command's
`--output-dir` places Trainer artifacts inside that same folder:

```text
logs/mnli-affected-pilot-1/
├── manifest.json
├── final_summary.json
├── final_summary.txt
├── aggregator/
│   └── telemetry                 # SQLite file, without an extension
├── nodes/node_0/
│   ├── training.stdout.log
│   └── training.stderr.log
└── training/
    ├── experiment.json          # Revisions, subset, settings and hardware
    ├── installed-packages.txt
    ├── trainer_state.json       # Loss/LR logs and validation history
    ├── train_results.json
    └── ...                      # Final model/tokenizer files
```

Loss and learning rate are logged every 50 optimizer updates, including the
first update. Validation loss and accuracy are logged after every epoch.
Trainer's `train_runtime` includes validation; TraceML step timing excludes
validation and final model saving. Compare these measurements with their
different boundaries in mind. Final weights are saved, but optimizer state
for resuming training is not.

## Copy a database snapshot for local analysis

Run this **on the GPU machine**, preferably after training and TraceML have
finished. It also works during a run to capture the currently committed data:

```bash
python - <<'PY'
from contextlib import closing
from pathlib import Path
import sqlite3

run = Path("logs/mnli-affected-pilot-1").resolve()
source = run / "aggregator/telemetry"
destination = run / "aggregator/telemetry.snapshot.sqlite"
with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as src:
    with closing(sqlite3.connect(destination)) as dst:
        src.backup(dst)
print(destination)
PY
```

Repeat for the fixed run. SQLite's
[backup API](https://www.sqlite.org/backup.html) includes committed WAL data in
one consistent snapshot. A mid-run snapshot cannot include later steps or rows
still waiting to be written. Do not copy just the active `telemetry` file or
open it through a network filesystem.

Copy the result directories into this checkout's local `logs/` folder, for
example from your laptop (replace the SSH host and remote path):

```bash
rsync -av --exclude='/aggregator/telemetry' \
  --exclude='/aggregator/telemetry-wal' --exclude='/aggregator/telemetry-shm' \
  --exclude='*.safetensors' --exclude='*.bin' \
  ubuntu@GPU_HOST:/absolute/path/to/traceml/logs/mnli-affected-pilot-1/ \
  logs/mnli-affected-pilot-1/
```

Repeat for the fixed run. The copied snapshot, configuration and logs are
enough for analysis; model weights are not required. Generated results stay
under the existing ignored `logs/` directory.

## Analyze locally

No GPU, database server or cloud connection is needed:

```bash
python -m pip install -e .
python -m pip install jupyterlab pandas matplotlib
jupyter lab experiments/mnli_regression/analyze.ipynb
```

Set `RUNS` in the first code cell to the copied run folders and choose **Run
All**. One run is enough to inspect a timeline; add its partner for comparison.
Use the same TraceML revision as the GPU run. The notebook opens SQLite
read-only and reuses TraceML's timing analyzer instead of recreating its
formulas. It reports retained coverage before plotting, preserves raw spikes,
and saves analysis tables/figures under each run's `analysis/` folder.

Trackio is optional: install it and add `--report-to trackio` to both training
commands for local learning curves through Trainer's built-in integration.
It is not needed to save loss logs or analyze TraceML. See
[Trackio's integration](https://huggingface.co/docs/trackio/transformers_integration).
