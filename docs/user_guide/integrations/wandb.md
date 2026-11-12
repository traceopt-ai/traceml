# Weights & Biases Integration

TraceML can publish its end-of-run diagnosis into the W&B run you already
use. The run then carries the bottleneck verdict next to your own metrics,
and the runs table can sort and compare runs by it.

TraceML can also stream each rank's step timings into W&B while training
runs, through the `wandb` exporter. See
[Send step time to Weights & Biases](#send-step-time-to-weights-biases).

## Install

This guide assumes your training framework is already installed.

```bash
pip install "traceml-ai[wandb]"
```

## Run

Launch your existing script through TraceML:

```bash
traceml run train.py
```

Hugging Face Trainer, PyTorch Lightning and RF-DETR training are
instrumented automatically. Other stacks use the setup in their
[integration guide](../integrations.md#choose-your-integration). Your W&B
setup stays as it is.

## Publish the diagnosis into your run

Call `log_training_report()` once after training finishes, while the W&B run
is still open. With no arguments it writes into the active `wandb.run`.

### Hugging Face Trainer

The Trainer's W&B callback keeps its run open after `train()` returns, so call
the helper right after training:

```python
from traceml_ai.integrations.wandb import log_training_report

trainer.train()
log_training_report()
```

### PyTorch Lightning

`WandbLogger` owns its run. Its `finalize()` does not call `wandb.finish()`,
so the run stays open after `trainer.fit()` returns. Call the helper then and
pass that run explicitly:

```python
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from traceml_ai.integrations.wandb import log_training_report

logger = WandbLogger(project="my-project")
trainer = Trainer(logger=logger)
trainer.fit(model, datamodule=datamodule)
log_training_report(run=trainer.logger.experiment)
```

### Custom PyTorch loop

Create and finish the run yourself, and call the helper in between:

```python
import traceml_ai as traceml
import wandb
from traceml_ai.integrations.wandb import log_training_report

traceml.init()
run = wandb.init(project="my-project")

for batch in dataloader:
    with traceml.trace_step(model):
        ...

log_training_report()
run.finish()
```

The [minimal example](https://github.com/traceopt-ai/traceml/blob/main/examples/wandb_report_minimal.py)
runs this loop end to end. Set `WANDB_MODE=offline` to try it without a W&B
account:

```bash
WANDB_MODE=offline traceml run examples/wandb_report_minimal.py --args --steps 20
```

## What lands in the run

- **Diagnosis card.** A compact HTML card is logged under
  `traceml/diagnosis_card`. W&B shows it as a media panel in the run
  workspace. The verdict comes first, because a workspace panel shows about
  330 by 180 px before you expand it. Below it are coverage, the phase bar
  and three Step Time averages.
- **Summary fields.** `traceml/primary/status`,
  `traceml/step_time/step_time_ms` and `traceml/step_time/input_wait_ms` are
  written to the run summary. They appear as columns in the runs table, so
  you can sort and filter runs by the verdict and the two timings. A field
  TraceML did not measure is left out rather than written as zero.

## How the helper behaves

- It never creates, configures or finishes a run. It reads no credentials
  and never calls `wandb.init()` or `wandb.login()`.
- It warns instead of raising. A missing `wandb` install, a missing run, a
  payload it cannot render or a failed write prints one `[TraceML] W&B:` line
  to stderr and returns `False`.
- It returns `True` when every write that had something to write succeeded.
  A summary with no diagnosis and no Step Time averages gets the card only
  and still returns `True`.
- When it fetches the summary itself, it writes from the primary rank only.
  On other ranks it returns `False` without requesting a summary, so you can
  call it on every rank.
- When it fetches the summary itself, it needs a mode that keeps local
  history. In `telemetry` mode TraceML keeps no history and generates no
  summary, so the helper prints one line and returns `False`.
- A payload passed with `final_summary=` skips both checks. It is written
  wherever the helper is called, so pass it from one rank only.
- Once it has a run, it requests the summary. When no TraceML session is
  available, or the summary request fails, it then raises the same
  `RuntimeError` as `traceml.final_summary()`. Pass `final_summary=` with a
  saved payload to publish an existing report.

## Send step time to Weights & Biases

The `wandb` exporter streams step time from the TraceML aggregator into W&B
while training runs. It is a built-in [live exporter](../live-export.md),
separate from `log_training_report()`: it creates runs of its own and never
touches the run your training script uses.

### Set it up

Install the extra and log in on the host that runs the aggregator. With
`traceml run` that is the host you launch from; with `traceml serve` it is
the host running `serve`; with Ray it is the aggregator actor's node.

```bash
pip install "traceml-ai[wandb]"
wandb login
```

Instead of `wandb login`, you can set `WANDB_API_KEY` or
`WANDB_IDENTITY_TOKEN_FILE` in the aggregator's environment: the shell you
launch from, the `serve` shell, or Ray's `runtime_env`. TraceML never reads
or passes the key itself, and it never prompts for one. For a dedicated or
self-hosted W&B server, set `WANDB_BASE_URL` as well.

`WANDB_ENTITY` and `WANDB_PROJECT` are both required:

```bash
export WANDB_ENTITY=my-team
export WANDB_PROJECT=my-project
traceml run train.py --mode telemetry --exporter wandb
```

`--mode summary --exporter wandb` works as well and keeps the local
analysis and final summary. With `traceml serve`, always pass `--run-name`:
without it, back-to-back jobs share one session and their runs collide.

### What lands in W&B

- **One run per rank.** Rank `r` gets a new run named `<run-name> rank <r>`.
  All of them sit in one group named after the TraceML run, with job type
  `traceml`, so the group page shows every rank as one line per chart.
  W&B therefore counts one run per rank for each training run.
- **One history row per step.** Each step a rank reports becomes one row
  in that rank's run, plotted against the hidden `traceml/step` axis.
  A step reported twice is logged twice.
- **The metrics.** The keys carry no rank, because each run is one rank:

| Key | Meaning |
|---|---|
| `traceml/step` | TraceML's step number, the x-axis of every chart |
| `traceml/timestamp_unix_ns` | when the training process measured the step |
| `traceml/<phase>_cpu_ms` | the phase's host-side time, summed over devices |
| `traceml/<phase>_gpu_ms` | the phase's GPU time, summed over devices |

`<phase>` is one of `input_wait`, `h2d`, `forward`, `backward`,
`optimizer_step` and `traced_step_time`. A measured zero is logged as zero.
A phase the step did not record, or a clock that was not measured, is left
out rather than written as zero; a CPU run has no `_gpu_ms` keys.

Plot against `traceml/step`, not W&B's wall-clock axis. Rows reach W&B a few
seconds after the step, so wall-clock charts show arrival time; use
`traceml/timestamp_unix_ns` to line a step up with your own logs.

Only the step-time stream is exported. Any other `--export-streams`
selection stops the aggregator at startup; use the `jsonl` exporter for
other streams. Regions you time with `trace_time()` are not exported. With
gradient accumulation, one step sums several forward and backward calls.

At the end of the run each run's summary gets three counts:
`traceml/export_written_records` (rows logged to that run),
`traceml/export_dropped_batches` and `traceml/export_skipped_records`
(records without a rank or step, or from a rank outside the run's world
size). The full loss report stays local in `finalization_warning.json`. A
written record means one handed to the W&B SDK.

### When it cannot start

In `telemetry` mode the exporter must start before training does. A missing
`wandb` install, a missing `WANDB_ENTITY` or `WANDB_PROJECT`, missing
credentials, a bad key, an unreachable server or a run W&B refuses to create
all stop the launch, and the launcher prints that training was not started.
This holds under the default `--on-missing-aggregator=raise`; with `warn`,
training starts without TraceML instead. Startup has a 10 second budget.
The launcher names the cause: `exporter 'wandb' failed to start: <reason>`,
and the run manifest records `telemetry_reason: exporter_startup_failed`.

In `summary`, `cli` and `dashboard` modes export is optional, like every
exporter: a startup failure prints one line, training runs without export,
and `finalization_warning.json` records the loss. `traceml serve` follows
its own `--mode` the same way.

The aggregator's W&B lines, including `[TraceML] W&B group: <url>`, go to
the terminal in every mode except `cli`, where the live display owns the
terminal and they go to `<logs-dir>/<run-name>/aggregator/process.stderr.log`.

The exporter logs online only. `WANDB_MODE` must be unset or `online`;
`offline`, `disabled` and any other value are rejected at startup. On a host
without W&B access, use the `jsonl` exporter.

### Shutdown and recovery

In `telemetry` mode each run's `finish()` gets up to 30 seconds at shutdown,
and the runs finish in parallel. A finish on a normal network takes about
2 seconds. The aggregator waits for the outcome before it writes
`finalization_warning.json`, so a finish that times out or fails is
recorded there as `exporter_failed`. The end-of-run deadline still bounds the
wait (`--finalize-timeout-sec`). On Ctrl+C the launcher gives the aggregator
5 seconds, and with Ray `TraceMLRayConfig(stop_timeout_sec=...)` must cover the
finish; see [Live Export](../live-export.md).

In `summary`, `cli` and `dashboard` modes each `finish()` is bounded to under
2 seconds, so a slow network cannot hold the aggregator open, and a finish
can be cut off. When it is, the run can show as crashed in W&B, and the run's
files stay under `<logs-dir>/<run-name>/aggregator/export/wandb/run-*`. The
same holds in telemetry mode when a finish runs out of time. Upload the files
with `wandb sync` and the explicit paths:

```bash
wandb sync --include-online logs/my-run/aggregator/export/wandb/run-*
```

On wandb 0.31 and later, add `--yes` to skip the confirmation prompt. A run
that received no row finishes as failed rather than finished.

### What the exporter changes and ignores

- It ignores inherited run settings in the aggregator process:
  `WANDB_RUN_ID`, `WANDB_RESUME`, `WANDB_RESUME_FROM`, `WANDB_FORK_FROM`,
  `WANDB_SWEEP_ID`, `WANDB_SWEEP_PARAM_PATH`, `WANDB_LAUNCH`,
  `WANDB_LAUNCH_CONFIG_PATH`, `WANDB_RUN_GROUP`, `WANDB_JOB_TYPE`,
  `WANDB_TAGS`, `WANDB_NOTES`, `WANDB_CONFIG_PATHS` and `WANDB_SERVICE`. The
  training process keeps its own environment.
- It sets `WANDB_ERROR_REPORTING=false` in the aggregator process unless you
  set it yourself.
- Its runs carry no code, git state, package list, console output, host
  metadata or system metrics. They describe TraceML's measurements only.
- The Stop button in the W&B UI does not stop training; the aggregator
  prints one line and keeps going. Stop the training run itself.
- Run files go under `<logs-dir>/<run-name>/aggregator/export/wandb/`, but
  wandb still keeps its own files in `~/.config/wandb` and `~/.cache/wandb`.

### Things to know

- A Hugging Face Trainer with `report_to="wandb"` keeps logging into its own
  run from the training process. Seeing both your run and TraceML's group
  in the project is expected. Linking the two is not supported yet.
- `--trace-max-steps N` is the volume control for long runs: each rank
  logs its first N steps. The runs stay open until training ends.
- A resumed training restarts TraceML's step at 1, so its charts start
  over on the same axis.
- A requeued Slurm job gets a new TraceML run name per attempt, so each
  attempt is a new group. The killed attempt's runs stay unfinished.
- With Ray, the credentials and both variables go into the aggregator
  actor's `runtime_env`. This path is documented, not tested, and the files
  for `wandb sync` sit on the actor's node.
- After a failed startup, Ctrl+C ends `traceml serve` while wandb shuts
  down. If it still does not stop, end it with `kill -9`.
- The exporter is a TraceML-maintained integration with the third-party
  `wandb` SDK (MIT license).

To check it by hand, run a short training script with
`--nproc-per-node 2 --mode telemetry --exporter wandb`, open the group link
the aggregator prints, and check that the group holds one run per rank and
that each chart follows `traceml/step`.

## Limitations

- The helper does not define training boundaries. Use automatic trainer
  instrumentation or the explicit setup for your loop.
- Call it after training finishes. The card is a finalized report, not a
  live metric.
- The repository tests the helper and the exporter against a stand-in
  `wandb` module. No real W&B client runs in CI.

## Next Steps

- [How to Read Output](../reading-output.md)
- [Compare Runs](../compare.md)
- [W&B, MLflow, and TensorBoard](wandb-mlflow.md)
- [Public API](../public-api.md#experiment-trackers)
