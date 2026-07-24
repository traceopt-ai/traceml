# DeepSpeed

Use TraceML with DeepSpeed to find training bottlenecks without changing how
your DeepSpeed job runs.

DeepSpeed leaves the training loop to you: you call `model_engine(...)`,
`model_engine.backward(loss)`, and `model_engine.step()` yourself. So the
integration is the same recipe used for plain PyTorch, DDP, and FSDP — wrap
each step with `traceml.trace_step(...)`. There is no DeepSpeed-specific
callback or wrapper to install.

## 1. Install

TraceML does not depend on DeepSpeed. Install DeepSpeed yourself:

```bash
pip install "traceml-ai[deepspeed]"
```

or follow the [DeepSpeed getting-started guide](https://www.deepspeed.ai/getting-started/).
DeepSpeed requires a CUDA GPU.

## 2. Wrap The Step

Call `traceml.init(mode="auto")` once, then wrap each DeepSpeed step with
`traceml.trace_step(...)`. Pass `model_engine.module` (the unwrapped model),
exactly like `model.module` for DDP and `base_model` for FSDP.

```python
import traceml_ai as traceml

# ... deepspeed.initialize(...) returns model_engine ...

traceml.init(mode="auto")

for batch_x, batch_y in loader:
    with traceml.trace_step(model_engine.module):
        batch_x = batch_x.to(model_engine.device, non_blocking=True)
        batch_y = batch_y.to(model_engine.device, non_blocking=True)

        logits = model_engine(batch_x)      # forward
        loss = criterion(logits, batch_y)

        model_engine.backward(loss)         # backward
        model_engine.step()                 # optimizer step
```

`traceml.init(mode="auto")` installs TraceML's process-wide auto-timers, so it
records DataLoader fetch timing (when you iterate a real PyTorch `DataLoader`,
as above), host-to-device (H2D) copies, forward, backward, optimizer, and step
timing, plus GPU and process memory, and writes an end-of-run summary.
Forward is timed on `model_engine.module`; backward is captured because
`model_engine.backward(loss)` reaches `torch.Tensor.backward()`;
optimizer time is captured from the underlying torch optimizer step inside
`model_engine.step()`. You do not need to add anything else per step.

## 3. Launch The Run

DeepSpeed reads `RANK` / `LOCAL_RANK` / `WORLD_SIZE` from the environment, and
`traceml run` launches your script through torchrun, so the two work together
directly.

Single GPU:

```bash
traceml run train.py --mode=summary
```

Single-node multi-GPU (e.g. 4 GPUs):

```bash
traceml run train.py --nproc-per-node=4 --mode=summary
```

For multi-node launch commands, see
[Distributed Training](../distributed-training.md).

## Limitations

- **No explicit NCCL / collective timing.** With ZeRO, gradient reduce-scatter
  runs during the backward pass and parameter all-gather runs inside the
  optimizer step. That communication overlaps the phases TraceML measures, so
  its cost is folded into backward / optimizer / step time and reported as
  residual/proxy timing rather than as a separate collective number.
- **Optimizer timing is "where available".** TraceML times the underlying torch
  optimizer step reached inside `model_engine.step()`. DeepSpeed's own work
  around that step (loss scaling, gradient clipping, ZeRO partitioning) is
  outside the traced phases and shows up as residual.
- **Gradient accumulation.** The example config uses
  `gradient_accumulation_steps: 1`, so each `trace_step` brackets one optimizer
  step. With accumulation greater than 1, `model_engine.step()` runs the
  optimizer only on boundary micro-steps, so backward is timed every step while
  optimizer time appears only on those boundaries, and TraceML's step count
  follows micro-steps rather than optimizer steps.

## Troubleshooting

### Multi-GPU run only shows one rank

Make sure you launched through TraceML with `--nproc-per-node`, not plain
`python`:

```bash
traceml run train.py --nproc-per-node=4
```

### I want a baseline without TraceML

Run the same script with TraceML disabled:

```bash
traceml run train.py --disable-traceml
```

This launches your script natively through `torchrun` without TraceML telemetry.

## Full Example

A runnable example and a minimal ZeRO stage-2 config live in the repo:

- `examples/integrations/deepspeed_minimal.py`
- `examples/integrations/deepspeed_config_minimal.json`

Run it with:

```bash
traceml run examples/integrations/deepspeed_minimal.py --mode=summary
```

The example exits cleanly when DeepSpeed or a CUDA GPU is unavailable.

## Next Steps

- [How to Read Output](../reading-output.md)
- [Distributed Training](../distributed-training.md)
- [Quickstart](../quickstart.md)
- [Open an issue](https://github.com/traceopt-ai/traceml/issues)
