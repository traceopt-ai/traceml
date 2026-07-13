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

## 2. Wrap the step

Pass `model_engine.module` (the unwrapped model) to `traceml.trace_step(...)`,
exactly like `model.module` for DDP and `base_model` for FSDP. Everything else
in your DeepSpeed loop stays the same.

```python
import traceml_ai as traceml

# ... deepspeed.initialize(...) returns model_engine ...

traceml.init(mode="auto")

for batch_x, batch_y in loader:
    with traceml.trace_step(model_engine.module):
        batch_x = batch_x.to(model_engine.device, non_blocking=True)
        batch_y = batch_y.to(model_engine.device, non_blocking=True)

        logits = model_engine(batch_x)          # forward
        loss = criterion(logits, batch_y)

        model_engine.backward(loss)             # backward
        model_engine.step()                     # optimizer step
```

`traceml.init(mode="auto")` enables DataLoader, forward, backward, optimizer,
and H2D timing for the process. `trace_step(...)` defines the per-step
boundary that TraceML measures.

## 3. Launch the run

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

## What TraceML can measure today

For a DeepSpeed run, TraceML reports:

- DataLoader / input timing
- host-to-device (H2D) transfer timing
- step timing
- forward, backward, and optimizer phase timing *where available*
- GPU and process memory
- final end-of-run summaries

TraceML uses this to tell you whether a run is input-bound, compute-bound,
residual-heavy, straggler-heavy, or drifting in memory.

## What TraceML does not measure yet

TraceML does **not** provide explicit NCCL / collective timing for DeepSpeed.

With ZeRO, gradient reduce-scatter runs via autograd hooks *during* the
backward pass, and parameter all-gather happens *inside* the optimizer step.
That communication overlaps the phases TraceML measures, so its cost is folded
into backward / optimizer / step time rather than reported as a separate
collective number. Today that communication shows up as **residual / proxy**
timing, not as an explicit NCCL breakdown.

## How it works

TraceML instruments the DeepSpeed loop through the same auto-timers it uses for
plain PyTorch:

- **Forward** is timed on `model_engine.module`, so it targets your model even
  though `model_engine` wraps it.
- **Backward** is timed because `model_engine.backward(loss)` ultimately calls
  `torch.Tensor.backward()` on the (scaled) loss, which TraceML's backward
  auto-timer captures.
- **Optimizer** is timed because `model_engine.step()` calls the underlying
  torch optimizer's `step()`, which TraceML's optimizer hook captures. This is
  why optimizer timing is reported "where available": DeepSpeed's own
  bookkeeping and collectives around that inner step are not separated out.

Because forward is scoped to `model_engine.module`, do not also wrap
`model_engine` itself in a second `trace_step`; one bracket per step is enough.

## Baseline without TraceML

To measure the same run without TraceML telemetry:

```bash
traceml run train.py --disable-traceml
```

This launches your script natively through torchrun.

## Full example

A runnable example and a minimal ZeRO stage-2 config live in the repo:

- `examples/integrations/deepspeed_minimal.py`
- `examples/integrations/deepspeed_config_minimal.json`

Run it with:

```bash
traceml run examples/integrations/deepspeed_minimal.py --mode=summary
```

The example exits cleanly when DeepSpeed or a CUDA GPU is unavailable.

## Next steps

- Read the [Quickstart](../quickstart.md) for plain PyTorch loops
- Read [Distributed Training](../distributed-training.md) for multi-GPU runs
- Open an issue if you hit a problem: https://github.com/traceopt-ai/traceml/issues
