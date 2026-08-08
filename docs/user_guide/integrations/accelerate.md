# Hugging Face Accelerate Integration

Use TraceML with a custom Hugging Face `Accelerate` training loop — for
users who write their own loop with `Accelerator.prepare(...)` and
`accelerator.backward(...)` instead of using `Trainer`.

The integration is not a callback or a new API. It is the same
`traceml.trace_step(...)` primitive used in any plain PyTorch loop, wrapped
around the unwrapped model that `accelerator.prepare()` produces.

## 1. Install

```bash
pip install "traceml-ai[hf]"
```

`accelerate` ships as part of the existing `hf` extra alongside
`transformers` — there is no separate `accelerate` extra, and TraceML does
not add `accelerate` as a core dependency.

## 2. Initialize TraceML And Wrap The Prepared Model

Create the `Accelerator` first, then call `traceml.init()`, then hand the
model, optimizer, and dataloader to `accelerator.prepare(...)` before
wrapping the *result* with `trace_step`:

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

`accelerator.unwrap_model(model)` matters even for a single-process run: it
is what makes this code correct unchanged under multi-GPU too. TraceML's
step-memory tracker keys instrumentation off `id(model)` and reads device
placement from `model.parameters()`. If `accelerator.prepare()` wraps the
model (which it does under distributed launch), passing the wrapped object
into `trace_step` would hook an object that isn't the one actually running
forward/backward. `unwrap_model()` is a no-op on CPU/single-GPU and strips
the wrapper under distributed — so this line is what keeps the same code
correct in both cases, not just a distributed-only concern.

Note also what's *not* here: no `model.to(device)`, no `batch.to(device)`.
`accelerator.prepare()` handles both — that's the actual value Accelerate
adds over writing raw `DistributedDataParallel` by hand.

## 3. Launch The Run

CPU or single GPU:

```bash
traceml run examples/integrations/accelerate_minimal.py --mode=summary
```

Single-node multi-GPU:

```bash
traceml run examples/integrations/accelerate_minimal.py --mode=summary --nproc-per-node=2
```

For multi-node DDP launch commands, see
[Distributed Training](../distributed-training.md).

**Use `traceml run`, not `accelerate launch`.** `traceml run` is itself a
launcher — under the hood it invokes `torch.distributed.run` with the
`--nproc-per-node` you give it, the same mechanism `accelerate launch` uses.
Running both would mean two launchers fighting over the same rendezvous.
Wrapping `accelerate launch` directly is not supported in this first version;
see Limitations.

## Limitations

- **Gradient accumulation is not modeled in the minimal example.** One call
  to `accelerator.backward()` maps to one `trace_step`. If you accumulate
  over several micro-batches before `optimizer.step()`, decide whether you
  want each micro-batch to be its own TraceML step or the whole accumulation
  window to be one step, and place `trace_step` accordingly — this guide
  does not prescribe one, since the right choice depends on what you're
  trying to measure.
- **DeepSpeed and FSDP configurations routed through Accelerate are out of
  scope for this first version.** This guide covers the plain `Accelerator()`
  path only. Tune DeepSpeed/FSDP independently of TraceML for now.

## Troubleshooting

### The verdict says RESIDUAL-HEAVY (CRITICAL) on the minimal example

This is expected on this specific example, not a sign of a problem. The
synthetic `TinyMLP` finishes each step in roughly a millisecond, so ordinary
Python loop overhead (the periodic `accelerator.print`, function-call
overhead, condition checks) becomes proportionally large next to the actual
compute. Don't use this example's verdict as a reference point for
interpreting a real training run — it's an artifact of the workload being
deliberately tiny.

In general, converting a loss tensor to a Python value (`float(loss)` or
`loss.item()`) forces the host to wait for the device inside the traced
region. Keep any running total on-device and convert it only when you
actually print or log, outside `trace_step`, as this example does. On the
GPU-selected clock this does not change reported residual time, since that
comes from CUDA event timestamps rather than CPU wall time.

### Step Memory shows NO DATA

Step memory is only populated when running on GPU. On a CPU-only run this
section is expected to be empty.

### Multi-GPU run only shows one rank

Make sure you launched through TraceML with `--nproc-per-node`, not plain
`python`:

```bash
traceml run examples/integrations/accelerate_minimal.py --mode=summary --nproc-per-node=2
```

### I want a baseline without TraceML

```bash
traceml run examples/integrations/accelerate_minimal.py --disable-traceml
```

This launches the script natively through `torchrun` without TraceML
telemetry.

## Full Examples

A complete, runnable version of the loop above lives at
`examples/integrations/accelerate_minimal.py`.
It trains a small MLP on synthetic data using the exact
`Accelerator()` → `traceml.init()` → `accelerator.prepare()` →
`accelerator.unwrap_model()` → `trace_step` sequence shown above, and
produces a `TraceML Run Summary` at the end of the run.

```bash
traceml run examples/integrations/accelerate_minimal.py --mode=summary
```

## Reference

`traceml.init(mode="auto")` installs TraceML's process-wide instrumentation
(`DataLoader` fetch timing, H2D `Tensor.to`, forward/backward/optimizer
auto-timers). Accepts `mode="auto"|"manual"|"selective"`. `"auto"` (the
default, and what this integration uses) auto-installs forward/backward/
optimizer timing, so unlike `manual` mode you don't need to explicitly wrap
each phase — `trace_step(...)` around the loop is enough. Idempotent; call
once, before `accelerator.prepare(...)`.

`traceml.trace_step(model)` brackets one training step. Pass the object your
loop actually runs forward/backward on — with Accelerate, that means
`accelerator.unwrap_model(model)` after `accelerator.prepare(...)`, not the
pre-`prepare()` model.

This guide introduces no new TraceML API. `accelerator.prepare(...)` and
`accelerator.unwrap_model(...)` are standard Accelerate methods; see
[Accelerate's own documentation](https://huggingface.co/docs/accelerate)
for their full behavior.

## Next Steps

- [How to Read Output](../reading-output.md)
- [Distributed Training](../distributed-training.md)
- [Hugging Face Trainer](huggingface.md)
- [Open an issue](https://github.com/traceopt-ai/traceml/issues)
