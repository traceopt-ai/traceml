# DeepSpeed Integration

## Install

This guide assumes PyTorch and DeepSpeed are already installed. The existing
DeepSpeed example requires a CUDA GPU.

```bash
pip install traceml-ai
```

## Add TraceML to your loop

Initialize TraceML once, then mark the training region with `trace_step()`.
Pass `model_engine.module` so forward timing targets the underlying model.
DeepSpeed continues to manage backward, gradient accumulation, and updates.

```python
import traceml_ai as traceml

# Keep your existing deepspeed.initialize(...) setup.
# It returns model_engine, optimizer, loader, and scheduler.
traceml.init(mode="auto")

for batch_x, batch_y in loader:
    with traceml.trace_step(model_engine.module):
        batch_x = batch_x.to(model_engine.device, non_blocking=True)
        batch_y = batch_y.to(model_engine.device, non_blocking=True)

        logits = model_engine(batch_x)
        loss = criterion(logits, batch_y)
        model_engine.backward(loss)
        model_engine.step()
```

Both `init()` and `trace_step()` are required for this setup. There is no
DeepSpeed-specific callback or automatic attachment.

## Run

Launch the instrumented script:

```bash
traceml run train.py
```

TraceML starts the telemetry collector and launches the script through
`torchrun`. Keep your existing DeepSpeed distributed initialization; the
repository example uses `deepspeed.init_distributed()` with the launcher's
rank environment.

## Read the result

When training finishes, TraceML prints a diagnosis of the likely bottleneck
and saves the report for comparison or CI.

The report shows available input waiting, forward, backward, optimizer,
host-to-device, and memory measurements. DeepSpeed-specific work may not have
its own phase measurement; see [Limitations](#limitations).

See [How to Read Output](../reading-output.md) for an example report and
explanations.

## How TraceML measures DeepSpeed training

TraceML uses its PyTorch timing hooks inside the region you mark. It does not
replace DeepSpeed's training loop or time the entire engine API as separate
forward, backward, and optimizer calls.

```text
Your DeepSpeed loop                  TraceML measurement
──────────────────────────────────────────────────────────────────
PyTorch DataLoader fetch             Input Wait
          ↓
trace_step(model_engine.module)      Open step capture
          ↓
Batch Tensor.to()                    GPU transfer
          ↓
model_engine(batch)                  Underlying model forward
          ↓
model_engine.backward(loss)          Observed PyTorch backward calls
          ↓
model_engine.step()                  Observed PyTorch optimizer calls
          ↓
Exit trace_step()                    Complete one reported step
```

**One reported step is one completed `trace_step()` block.** In the loop above,
that means one microbatch. The repository example sets
`gradient_accumulation_steps=1`, so each block also contains an optimizer update
attempt.

With accumulation of four, the same loop reports four TraceML steps for one
DeepSpeed optimizer update. TraceML does not automatically combine those
microbatches into one record. Input Wait remains separate from traced training
time, and CUDA memory reports the peak within each block.

## Multi-GPU training

For single-node training on four GPUs:

```bash
traceml run train.py --nproc-per-node=4
```

Initialize TraceML in every worker, as in the loop above. DeepSpeed reads the
rank environment supplied by the launcher. For multi-node commands, see
[Distributed Training](../distributed-training.md).

## Advanced options

### Runnable example

The repository includes a small training script and a ZeRO stage-2 config:

- [DeepSpeed example](https://github.com/traceopt-ai/traceml/blob/main/examples/integrations/deepspeed_minimal.py)
- [Example configuration](https://github.com/traceopt-ai/traceml/blob/main/examples/integrations/deepspeed_config_minimal.json)

From the repository root:

```bash
traceml run examples/integrations/deepspeed_minimal.py --args --steps 20
```

The example exits without training if DeepSpeed or a CUDA GPU is unavailable.

### Direct launches

For a direct launch, start a TraceML aggregator with `traceml serve` and
configure workers to connect to it. Keep the same `init()` and `trace_step()`
setup. See [Direct Launch](../public-api.md#direct-launch-with-traceml-serve).

## Limitations

- **Phase coverage.** Backward timing observes patched PyTorch backward entry
  points. Optimizer timing observes PyTorch optimizer step hooks. DeepSpeed
  implementations that bypass these hooks may leave those signals unavailable.
  An observed optimizer event does not measure the whole `model_engine.step()`
  call. Other engine work can contribute to Residual.
- **Distributed communication.** TraceML does not separately time NCCL
  collectives. Communication can occur within observed phases or elsewhere in
  the traced region; the report does not isolate its cost.
- **Step boundaries.** The illustrated setup reports microbatch steps when
  accumulation is enabled. Evaluation or other work placed inside `trace_step()`
  is also included; keep it outside if you want training-only measurements.
- **Input timing.** Automatic fetch timing covers a normal PyTorch DataLoader.
  Other input sources need explicit fetch wrapping; see the
  [core API](../public-api.md#manual-instrumentation-helpers).
- **Validation.** This is a documented recipe, not a real-DeepSpeed CI-tested
  integration. The repository has no real DeepSpeed CUDA or distributed signal
  validation job. See the
  [support matrix](../integrations.md#integration-support-matrix).

## Next Steps

- [How to Read Output](../reading-output.md)
- [Compare Runs](../compare.md)
- [Catch Regressions in CI](../regression-guard.md)
- [Distributed Training](../distributed-training.md)
- [Open an issue](https://github.com/traceopt-ai/traceml/issues)
