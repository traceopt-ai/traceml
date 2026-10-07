# Hugging Face Accelerate Integration

## Install

This guide assumes PyTorch and Hugging Face Accelerate are already installed.

```bash
pip install traceml-ai
```

## Add the step boundary

For a custom Accelerate loop, initialize TraceML once and wrap each training
step with `trace_step()`. Unlike Hugging Face Trainer, this path requires
explicit setup in your script.

Create the `Accelerator` first, initialize TraceML, then prepare your existing
model, optimizer, and DataLoader:

```python
import traceml_ai as traceml
from accelerate import Accelerator

accelerator = Accelerator()
traceml.init(mode="auto")

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
traced_model = accelerator.unwrap_model(model)

for batch_x, batch_y in dataloader:
    with traceml.trace_step(traced_model):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        accelerator.backward(loss)
        optimizer.step()
```

Pass the unwrapped model to TraceML, while continuing to call the prepared
`model` for training. This identifies the underlying module when Accelerate
wraps it for distributed execution.

## Run

```bash
traceml run train.py
```

TraceML starts the collector and runs your instrumented script. Use this
command as the launcher rather than nesting `accelerate launch` inside it.
For direct launches with a separate collector, see
[Direct Launch](../public-api.md#direct-launch-with-traceml-serve).

## Read the result

When training finishes, TraceML prints a diagnosis of the likely bottleneck
and saves the report for comparison or CI.

The timing breakdown shows available input waiting, forward, backward, and
optimizer measurements. CUDA runs also show step memory. The default
Accelerate-prepared loader transfers batches before the illustrated
`trace_step()` block, so those transfers are not reported as H2D.
See [Measuring GPU transfers](#measuring-gpu-transfers) to include them.

See [How to Read Output](../reading-output.md) for an example report and
explanations.

## How TraceML measures Accelerate training

Accelerate manages device placement and distributed execution. TraceML's
`init()` installs timing hooks, and your `trace_step()` block defines the
work included in one reported step.

```text
Accelerate training loop             TraceML measurement
──────────────────────────────────────────────────────────────────
Prepared DataLoader fetch            Observed PyTorch fetch waiting
+ automatic device transfer          Outside the illustrated step
          ↓
trace_step(unwrapped_model)          Open step capture
          ↓
model(batch)                         Forward
          ↓
accelerator.backward(loss)           Backward
          ↓
optimizer.step()                     Optimizer
          ↓
Exit trace_step()                    Complete one reported step
```

In the example, each batch performs one optimizer update and produces one
TraceML step. CUDA memory reports the peak within that block. Input waiting
is reported separately from traced training time.

With gradient accumulation, each completed `trace_step()` block still
produces one record. TraceML does not automatically group Accelerate
microbatches into optimizer steps; the placement of the block determines
what a reported step contains.

## Multi-GPU training

For single-node multi-GPU DDP:

```bash
traceml run train.py --nproc-per-node=4
```

Accelerate reads the worker environment created by the launcher. This recipe
uses a standard `Accelerator()` configuration. For multi-node launch commands,
see [Distributed Training](../distributed-training.md).

## Advanced options

### Measuring GPU transfers

To include batch transfers in H2D timing, disable automatic device placement
for the DataLoader and move the tensors inside `trace_step()`:

```python
model, optimizer, dataloader = accelerator.prepare(
    model,
    optimizer,
    dataloader,
    device_placement=[True, True, False],
)
traced_model = accelerator.unwrap_model(model)

for batch_x, batch_y in dataloader:
    with traceml.trace_step(traced_model):
        batch_x = batch_x.to(accelerator.device)
        batch_y = batch_y.to(accelerator.device)
        # Keep forward, backward, and optimizer work inside this block.
```

Use this preparation call instead of the earlier one. `device_placement`
needs one value for each object passed to `prepare()`. Adapt the tensor
transfers to your batch structure.

### Runnable example

The [minimal example](https://github.com/traceopt-ai/traceml/blob/main/examples/integrations/accelerate_minimal.py)
trains a small MLP on synthetic data with the explicit setup shown above.
From the repository root:

```bash
traceml run examples/integrations/accelerate_minimal.py
```

This is a smoke workload, not a performance baseline. Its small compute
workload can make overhead proportionally large.

## Limitations

- **Step boundaries.** Accumulation grouping is defined by your `trace_step()`
  blocks, not inferred from Accelerate's optimizer-update boundaries.
- **Transfers.** Automatic prepared-loader transfers happen outside the basic
  example's step boundary. They are not included in its H2D measurement.
- **Distributed strategies.** This recipe covers ordinary CPU/CUDA training
  and DDP. It does not establish support for DeepSpeed or FSDP configurations
  routed through Accelerate. See the
  [support matrix](../integrations.md#integration-support-matrix) for validation
  evidence.
- **Memory.** Step memory is a CUDA measurement; CPU runs do not report it.

## Next Steps

- [How to Read Output](../reading-output.md)
- [Compare Runs](../compare.md)
- [Catch Regressions in CI](../regression-guard.md)
- [Distributed Training](../distributed-training.md)
- [Hugging Face Trainer](huggingface.md)
- [Public API](../public-api.md)
- [Open an issue](https://github.com/traceopt-ai/traceml/issues)
