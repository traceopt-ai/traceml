# LeRobot v3 image-dataset input regression

This example reproduces the image-dataset training regression reported in
[LeRobot issue #2282](https://github.com/huggingface/lerobot/issues/2282) and
the dataset access change merged in
[LeRobot PR #2408](https://github.com/huggingface/lerobot/pull/2408). It runs
the same ACT training workload at the pinned commit before and after the fix,
then compares the resulting TraceML summaries.

This directory contains a reproduction foundation, not benchmark results. The
upstream change is expected to improve image-dataset input access. TraceML
measures input wait and complete training-step time during each local run. No
performance outcome is claimed here until the controlled GPU experiment has
been completed.

## Controlled inputs

| Input | Pinned value |
|---|---|
| LeRobot before fix | [`f6b16f6`](https://github.com/huggingface/lerobot/commit/f6b16f6d97155e3ce34ab2a1ec145e9413588197) |
| LeRobot after fix | [`67a6c8d`](https://github.com/huggingface/lerobot/commit/67a6c8dfef90351830ad02afc5bf1bd299f9a521) |
| Dataset | [`imstevenpmwork/aloha_sim_transfer_cube_human_image`](https://huggingface.co/datasets/imstevenpmwork/aloha_sim_transfer_cube_human_image) |
| Dataset revision | `13e0d3bff90f02ec417761b58199eb1d33a72efb` |
| Policy and device | ACT on CUDA |
| Training length | 200 steps |
| LeRobot defaults | batch size 8, 4 workers, seed 1000 |
| PyTorch stack | PyTorch 2.7.1 / CUDA 12.8, torchvision 0.22.1, torchcodec 0.5 |
| Data and launch stack | datasets 4.1.1, Accelerate 1.11.0 |

The runner verifies that the two LeRobot commits differ only in
`src/lerobot/datasets/lerobot_dataset.py`. The training script and the TraceML
patch are otherwise identical.

## Requirements

- Linux x86-64.
- Python 3.10 available as `python3.10`.
- One NVIDIA GPU visible to the process and a driver that supports CUDA 12.8.
- At least 10 GiB of free disk space.
- Network access to GitHub, Python package indexes, the PyTorch CUDA wheel
  index, and the public Hugging Face dataset.

The dataset does not require authentication. An optional `HF_TOKEN` can still
be set to avoid anonymous Hub rate limits; the runner does not print or copy
it into its environment record.

## Run

From the TraceML repository root, run one A/B pair:

```bash
bash examples/advanced/lerobot_v3_image_regression/run_reproduction.sh
```

For evidence intended for publication, run two pairs. The second pair reverses
the order from fixed-after-broken to broken-after-fixed so cache and execution
order are visible rather than silently confounded:

```bash
bash examples/advanced/lerobot_v3_image_regression/run_reproduction.sh --pairs 2
```

The first invocation creates a reusable Python environment, clones LeRobot
once, and downloads the pinned public dataset. Later invocations reuse the
environment, source clone, and Hugging Face cache. Every invocation creates a
new result directory and retains earlier work.

## Outputs

All runner-managed state is written below the repository's ignored `logs/`
directory:

```text
logs/lerobot_v3_image_regression/
├── cache/
├── source/lerobot/
├── venv/
└── runs/<experiment-id>/
    ├── compare/
    ├── environment.txt
    ├── lerobot/
    ├── terminal/
    ├── traceml/
    ├── wandb/
    └── worktrees/
```

Each pair produces `pair_<n>_broken_vs_fixed.json` and `.txt` in `compare/`.
The underlying TraceML summaries remain in
`traceml/<run-name>/final_summary.json`. W&B is forced into offline mode and
uses the invocation-specific `wandb/` directory.

The environment record is deliberately narrow: it includes only public
revision identifiers, package versions, OS/kernel architecture, and GPU model,
memory, and driver. It does not enumerate environment variables or record a
username or hostname.

## Instrumentation boundary

The same [`traceml.patch`](traceml.patch) is applied to both detached LeRobot
worktrees. It only:

- imports and initializes TraceML after the `Accelerator` exists;
- obtains the unwrapped policy after `accelerator.prepare(...)`;
- leaves `next(dl_iter)` outside `traceml.trace_step(...)`; and
- places preprocessing and LeRobot's unchanged `update_policy(...)` call inside
  the traced step.

This boundary lets TraceML associate the already measured DataLoader wait with
the following step while retaining complete step timing. It does not alter the
dataset query, model, optimizer, DataLoader, checkpoint behavior, or training
arguments.

## Limitations and deviations from upstream

- TraceML can show whether input wait and end-to-end step time change; it does
  not attribute time directly to LeRobot's private dataset query function.
- This LeRobot version calls `policy.forward(...)` directly. TraceML's automatic
  module-call instrumentation therefore may not report forward timing. The
  example preserves that upstream behavior rather than rewriting the call.
- TraceML's raw local run artifacts retain its normal runtime identity fields,
  including hostname and absolute local paths. The narrow `environment.txt`
  record omits them, but raw artifacts need a privacy review before publication.
- The dataset is pinned to an immutable revision, each run receives a unique
  output directory and name, and W&B runs offline. These are reproducibility
  and isolation controls added around the upstream command.
- Model, optimizer, DataLoader settings, checkpoint behavior, batch size,
  workers, seed, and training length otherwise remain the upstream workload.
- One pair is suitable for checking that the reproduction runs. Two alternating
  pairs reduce obvious order bias, but they are not a substitute for inspecting
  variance before making a measured claim.
