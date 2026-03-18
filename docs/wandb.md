# W&B Integration

Ship TraceML end-of-run summaries into your existing Weights & Biases experiment tracking workflow.

---

## Installation

```bash
pip install "traceml-ai[wandb]"
```

This pulls in `wandb>=0.16.0` as an optional dependency. TraceML's core works without it.

---

## How it works

`traceml run` launches **two separate processes**:

- **Executor** — runs your training script (this is where `wandb.run` lives)
- **Aggregator** — collects telemetry via TCP, maintains the session SQLite DB

Because the aggregator is a separate subprocess, `wandb.run` is always `None` there. The W&B upload **must happen inside your script** (the executor process), before `wandb.finish()`.

```
traceml run train.py
  ├─ Aggregator subprocess  ← writes telemetry DB continuously
  └─ Executor subprocess    ← your script runs here, wandb.run is active ✓
                               call upload_traceml_summary() here
```

---

## Usage

### Recommended — `upload_traceml_summary()` (works with `traceml run`)

Call this inside your script, **after training, before `wandb.finish()`**:

```python
import wandb
from traceml.integrations.wandb import upload_traceml_summary

wandb.init(project="my-project", name="my-run")

# ... training loop with trace_step(model) ...

# Upload TraceML summary BEFORE wandb.finish()
upload_traceml_summary(log_as_charts=True)

wandb.finish()
```

Run with:

```bash
traceml run train.py
```

`upload_traceml_summary()` reads the session DB (path derived automatically from `TRACEML_LOGS_DIR` / `TRACEML_SESSION_ID` env vars set by `traceml run`), generates the summary in-process, and pushes to W&B.

---

### `log_as_charts` — control where metrics appear

| `log_as_charts` | Where metrics appear |
|---|---|
| `False` (default) | **Overview → Summary** (scalar values, runs table, sweep comparison) |
| `True` | **Overview → Summary** + **Charts tab** (dot at `traceml_summary_step=1`) |

When `log_as_charts=True`:
- TraceML calls `wandb.define_metric("traceml/*", step_metric="traceml_summary_step")` so the metrics use their **own x-axis**, completely decoupled from the training step counter.
- All dots appear at `traceml_summary_step=1` (x=1), not at the last training step.
- Compare runs side-by-side by selecting multiple runs in the workspace — each run's dot shows its end-of-run value.

---

### Alternative — explicitly from a JSON file

If you have an existing `*_summary_card.json` (e.g. from a previous run):

```python
import wandb
from traceml.integrations.wandb import log_traceml_summary_to_wandb

with wandb.init(project="my-project") as run:
    log_traceml_summary_to_wandb(
        summary_json_path="./logs/<session>/aggregator/telemetry_summary_card.json",
        run=run,
        log_as_charts=True,
    )
```

---

## What gets logged

### Flat metrics → `run.summary`

| W&B metric key | Units | Description |
|---|---|---|
| `traceml/system/duration_s` | s | Total monitored duration |
| `traceml/system/cpu_avg_percent` | % | Avg CPU utilisation |
| `traceml/system/cpu_peak_percent` | % | Peak CPU utilisation |
| `traceml/system/ram_avg_gb` | GB | Avg RAM used |
| `traceml/system/ram_peak_gb` | GB | Peak RAM used |
| `traceml/system/ram_total_gb` | GB | Total RAM available |
| `traceml/system/gpu_available` | bool | GPU detected |
| `traceml/system/gpu_count` | int | Number of GPUs |
| `traceml/system/gpu_util_avg_percent` | % | Avg GPU utilisation |
| `traceml/system/gpu_util_peak_percent` | % | Peak GPU utilisation |
| `traceml/system/gpu_mem_avg_gb` | GB | Avg GPU memory used |
| `traceml/system/gpu_mem_peak_gb` | GB | Peak GPU memory used |
| `traceml/system/gpu_temp_avg_c` | °C | Avg GPU temperature |
| `traceml/system/gpu_temp_peak_c` | °C | Peak GPU temperature |
| `traceml/system/gpu_power_avg_w` | W | Avg GPU power draw |
| `traceml/system/gpu_power_peak_w` | W | Peak GPU power draw |
| `traceml/step_time/training_steps` | int | Total training steps |
| `traceml/step_time/ranks_seen` | int | DDP ranks observed |
| `traceml/step_time/worst_avg_step_ms` | ms | Slowest rank avg step time |
| `traceml/step_time/median_avg_step_ms` | ms | Median rank avg step time |
| `traceml/step_time/worst_vs_median_pct` | % | Straggler gap (worst vs median) |
| `traceml/step_time/median_dataloader_ms` | ms | Median rank: dataloader time |
| `traceml/step_time/median_forward_ms` | ms | Median rank: forward time |
| `traceml/step_time/median_backward_ms` | ms | Median rank: backward time |
| `traceml/step_time/median_optimizer_ms` | ms | Median rank: optimizer time |
| `traceml/step_time/worst_dataloader_ms` | ms | Worst rank: dataloader time |
| `traceml/step_time/worst_forward_ms` | ms | Worst rank: forward time |
| `traceml/step_time/worst_backward_ms` | ms | Worst rank: backward time |
| `traceml/step_time/worst_optimizer_ms` | ms | Worst rank: optimizer time |

Keys with `None` values (e.g. GPU metrics on CPU-only machines) are silently omitted.

When `log_as_charts=True`, a hidden `traceml_summary_step` key is also logged as the x-axis for all `traceml/*` panels. It is always `1` and can be ignored.

### Full JSON artifact

The full `*_summary_card.json` is uploaded as a W&B Artifact named `traceml_summary` (type `"traceml"`). Browse it in the **Artifacts** tab.

---

## Graceful failure

The integration is fully optional and **never crashes your run**:

- `wandb` not installed → warning logged, training continues
- No active W&B run → warning logged, upload skipped
- Any W&B API error → caught, logged as warning, returns `False`

---

## Examples

| File | Description |
|---|---|
| [`cnn_mnist_wandb.py`](../src/examples/advanced/cnn_mnist_wandb.py) | Summary metrics only (Overview tab) |
| [`cnn_mnist_wandb_charts.py`](../src/examples/advanced/cnn_mnist_wandb_charts.py) | Summary + Charts tab panels |

---

## FAQ

**The metrics show as dots, not lines — is that right?**
Yes. These are end-of-run aggregates — one value per run. W&B draws a line for time-series data; one run = one dot. Run multiple experiments and compare them in a bar chart or the runs table for a richer view.

**Why does the Charts tab x-axis say `traceml_summary_step` instead of `Step`?**
`traceml/*` metrics are decoupled from the training step counter via `wandb.define_metric()`. They always appear at `traceml_summary_step=1` on their own axis, avoiding the confusing placement at the last training step (e.g. x=500).

**Will this slow down training?**
No. The upload happens after the training loop exits, before `wandb.finish()`.

**Does this work with DDP / multi-GPU?**
Yes. TraceML's summary aggregates across all ranks. Call `upload_traceml_summary()` once on rank 0.

**Can I rename the `traceml/` metric keys?**
The keys are intentionally stable for cross-run comparison. Call `wandb.run.summary.update({...})` with custom keys after `upload_traceml_summary()` if you need aliases.
