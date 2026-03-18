# W&B Integration

Ship TraceML end-of-run summaries into your existing Weights & Biases experiment tracking workflow.

---

## Installation

```bash
pip install "traceml-ai[wandb]"
```

This pulls in `wandb>=0.16.0` as an optional dependency.  TraceML's core continues to work without it.

---

## What gets logged

### Flat metrics → `run.summary`

Every metric is logged to [`run.summary`](https://docs.wandb.ai/guides/track/log/log-summary/) so it is **queryable and comparable across runs** in the W&B UI.

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
| `traceml/system/gpu_util_peak_percent`| % | Peak GPU utilisation |
| `traceml/system/gpu_mem_avg_gb` | GB | Avg GPU memory used |
| `traceml/system/gpu_mem_peak_gb` | GB | Peak GPU memory used |
| `traceml/system/gpu_temp_avg_c` | °C | Avg GPU temperature |
| `traceml/system/gpu_temp_peak_c` | °C | Peak GPU temperature |
| `traceml/system/gpu_power_avg_w` | W | Avg GPU power draw |
| `traceml/system/gpu_power_peak_w` | W | Peak GPU power draw |
| `traceml/step_time/training_steps` | int | Total training steps |
| `traceml/step_time/ranks_seen` | int | DDP ranks observed |
| `traceml/step_time/worst_avg_step_ms` | ms | Slowest rank avg step time |
| `traceml/step_time/median_avg_step_ms`| ms | Median rank avg step time |
| `traceml/step_time/worst_vs_median_pct`| % | Straggler gap (worst vs median) |
| `traceml/step_time/median_dataloader_ms` | ms | Median rank: dataloader time |
| `traceml/step_time/median_forward_ms` | ms | Median rank: forward time |
| `traceml/step_time/median_backward_ms`| ms | Median rank: backward time |
| `traceml/step_time/median_optimizer_ms`| ms | Median rank: optimizer time |
| `traceml/step_time/worst_dataloader_ms`| ms | Worst rank: dataloader time |
| `traceml/step_time/worst_forward_ms` | ms | Worst rank: forward time |
| `traceml/step_time/worst_backward_ms` | ms | Worst rank: backward time |
| `traceml/step_time/worst_optimizer_ms`| ms | Worst rank: optimizer time |

Keys with `None` values (e.g. GPU metrics on a CPU-only machine) are silently omitted.

### Full JSON artifact

The complete `*_summary_card.json` file (including per-rank breakdowns) is uploaded as a **W&B Artifact** named `traceml_summary` with type `"traceml"`.  You can browse it in the **Artifacts** tab of your W&B run.

---

## Usage

### Option A — Automatic (recommended with `traceml run`)

Set `TRACEML_WANDB_AUTO=1` and call `wandb.init()` before training.  TraceML will pick up the active run and export the summary automatically at shutdown.

```bash
TRACEML_WANDB_AUTO=1 traceml run train.py
```

```python
# train.py
import wandb

wandb.init(project="my-project", name="my-run")

# ... your training loop with trace_step(model) ...

wandb.finish()
```

No other changes to your code are needed.

---

### Option B — Explicit API

Call `generate_summary()` with `wandb_run` after training:

```python
import wandb
from traceml.aggregator.final_summary import generate_summary

wandb.init(project="my-project")

# ... training ...

generate_summary(
    db_path="./logs/session_xyz.db",
    wandb_run=wandb.run,
)
wandb.finish()
```

---

### Option C — Post-run, from file

If you already have a `*_summary_card.json` file, upload it to any W&B run:

```python
import wandb
from traceml.integrations.wandb import log_traceml_summary_to_wandb

with wandb.init(project="my-project", name="post-hoc-upload") as run:
    log_traceml_summary_to_wandb(
        summary_json_path="./logs/session_xyz.db_summary_card.json",
        run=run,
    )
```

---

## Graceful failure

The W&B integration is **fully optional** and **never crashes your run**:

- If `wandb` is not installed → warning is printed, training continues normally.
- If no active W&B run is found → warning is printed, nothing is uploaded.
- If any W&B API call raises → exception is caught, logged as a warning, `False` is returned.

---

## Example

See [`src/examples/advanced/cnn_mnist_wandb.py`](../src/examples/advanced/cnn_mnist_wandb.py) for a complete example combining TraceML step profiling with live W&B loss logging and end-of-run summary upload.

---

## FAQ

**Does TraceML auto-upload to W&B without me doing anything?**
No.  You must either set `TRACEML_WANDB_AUTO=1` (and call `wandb.init()` in your script) or call `log_traceml_summary_to_wandb()` / pass `wandb_run=` to `generate_summary()` explicitly.

**Will this slow down training?**
No.  The export happens only at shutdown, after the training loop finishes.

**Can I rename the metrics?**
The metric names are intentionally stable for cross-run comparison.  If you need custom names, call `wandb.run.summary.update({...})` with your preferred keys after `log_traceml_summary_to_wandb()`.

**Does this work with DDP / multi-GPU?**
Yes.  TraceML's summary already aggregates across ranks.  The W&B export runs only on the process that calls `generate_summary()` (rank 0 in normal operation).
