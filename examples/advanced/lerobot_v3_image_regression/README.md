# LeRobot v3 image loading regression

This example reproduces [LeRobot issue #2282](https://github.com/huggingface/lerobot/issues/2282):
image training became input-bound after the v3 dataset changed how rows were
read. [PR #2408](https://github.com/huggingface/lerobot/pull/2408) changed that
access path. The runner trains the same ACT workload before and after the fix,
then asks TraceML to compare them.

This is a reproduction, not a published benchmark. It intentionally contains
no performance claim.

## Run it

You need Linux x86-64, Python 3.10, one NVIDIA GPU, and network access. The
environment uses PyTorch 2.7.1 with CUDA 12.8. A newer NVIDIA driver is fine.

From the TraceML repository root:

```bash
bash examples/advanced/lerobot_v3_image_regression/run_reproduction.sh
```

That runs one broken/fixed pair. For publication evidence, run two pairs so the
second pair reverses the order:

```bash
bash examples/advanced/lerobot_v3_image_regression/run_reproduction.sh --pairs 2
```

The first run installs the pinned environment and downloads the public dataset.
Later runs reuse both. Every run writes to a new directory under:

```text
logs/lerobot_v3_image_regression/runs/<experiment-id>/
```

The comparison is in `compare/`; raw TraceML summaries are in `traceml/`.

## What is fixed

- Broken LeRobot: `f6b16f6d97155e3ce34ab2a1ec145e9413588197`
- Fixed LeRobot: `67a6c8dfef90351830ad02afc5bf1bd299f9a521`
- Dataset: `imstevenpmwork/aloha_sim_transfer_cube_human_image`
- Dataset revision: `13e0d3bff90f02ec417761b58199eb1d33a72efb`
- ACT on CUDA for 200 steps
- LeRobot defaults for batch size, workers, and seed

Exact dependency versions live in `requirements.txt`.

## TraceML boundary

The same small patch is applied to both LeRobot commits. It keeps data loading
outside `trace_step` and keeps preprocessing plus the unchanged training update
inside it. TraceML selectively instruments DataLoader fetches, backward, and
host-to-device copies. After `accelerator.prepare`, it wraps the unwrapped
policy's `forward` method and the prepared optimizer.

The forward wrapper is necessary because this LeRobot version calls
`policy.forward(batch)` directly. The patch measures that call without changing
it to `policy(batch)` or changing LeRobot's model, optimizer, DataLoader,
checkpointing, or training arguments.

TraceML reports time at this boundary; it does not time LeRobot's private
dataset query function directly. W&B runs offline. Raw local artifacts may
contain normal runtime paths and machine identity, so review them before public
upload.
