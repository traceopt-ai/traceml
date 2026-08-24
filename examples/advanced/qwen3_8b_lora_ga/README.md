# Qwen3-8B LoRA gradient accumulation on one GPU

This workload measures how physical batch size changes TRL LoRA training
throughput while the effective batch and processed-token capacity stay fixed.
It is intended for one NVIDIA L40S 48 GB GPU, such as an AWS `g6e.2xlarge`.

The workload uses Qwen3-8B, the `train_sft` split of UltraChat 200k, BF16,
2,048-token packed sequences, assistant-only loss, LoRA and gradient
checkpointing. Model and dataset revisions are pinned in `train.py`.

This is a production-shaped throughput benchmark, not a complete production
fine-tune. Evaluation and checkpoint saving are intentionally disabled so they
do not distort steady training measurements.

## Environment

Start from an AWS PyTorch Deep Learning AMI with one L40S GPU. From the TraceML
repository root, install TraceML from the checkout and the workload packages:

```bash
python -m pip install -e .
python -m pip install -r examples/advanced/qwen3_8b_lora_ga/requirements.txt
```

This workload does not use `torchao`. If the AMI contains `torchao<0.16` and
PEFT reports that it is incompatible, remove that unused package before
running:

```bash
python -m pip uninstall -y torchao
```

The first run downloads approximately 16 GB of model weights and the public
dataset. Use an authenticated `HF_TOKEN` to avoid anonymous Hub rate limits.

## Run the matrix

The default command performs a five-step fit check with physical batch 8,
followed by four 500-update runs:

```bash
bash examples/advanced/qwen3_8b_lora_ga/run_matrix.sh --steps 500
```

The matrix keeps the effective batch at 8:

| Physical batch | Accumulation | Effective batch |
|---:|---:|---:|
| 1 | 8 | 8 |
| 2 | 4 | 8 |
| 4 | 2 | 8 |
| 8 | 1 | 8 |

If the physical batch 8 preflight runs out of memory, rerun with batch 4 as
the largest configuration:

```bash
bash examples/advanced/qwen3_8b_lora_ga/run_matrix.sh \
  --steps 500 \
  --max-batch 4
```

For an order-reversed second measurement:

```bash
bash examples/advanced/qwen3_8b_lora_ga/run_matrix.sh \
  --steps 500 \
  --order reverse \
  --repeat 2 \
  --skip-preflight
```

Each configuration is a separate process. Trainer runtime and throughput are
printed before TraceML's final phase and memory summary. Outputs are written
under:

```text
logs/qwen3_8b_lora_ga/<experiment-id>/
```

The runner stops instead of continuing if TraceML reports an end-of-run
rank-finalization timeout.

## Run one configuration

For example, run only physical batch 2 with eight effective samples per
optimizer update:

```bash
traceml run \
  --mode summary \
  --summary-window-rows 500 \
  --run-name qwen3_8b_bs2_ga4 \
  examples/advanced/qwen3_8b_lora_ga/train.py \
  --args \
  --batch-size 2 \
  --max-steps 500 \
  --run-name qwen3_8b_bs2_ga4
```

Use Trainer runtime as the primary end-to-end measurement. TraceML phase
timing explains where the step time was spent, but it is not kernel-level
root-cause attribution. Because gradient checkpointing is enabled, the
backward region includes activation recomputation.

## Validate runtime without TraceML

`train_no_traceml.py` uses the same dataset preparation, model, LoRA and
`SFTConfig` as `train.py`, but it does not import or initialize TraceML. It
records every optimizer step with non-blocking CUDA events and synchronizes
only at the boundaries of the complete training loop.

Run the two endpoint configurations in reverse order from the TraceML matrix:

```bash
bash examples/advanced/qwen3_8b_lora_ga/run_no_traceml_endpoints.sh \
  --steps 500
```

Each run writes:

- `trainer_metrics.json`: Hugging Face Trainer runtime and throughput.
- `optimizer_step_times.csv`: GPU and host interval for every optimizer step.
- `baseline_timing.json`: timing method and aggregate statistics.
- A terminal log under the experiment's `terminal` directory.

The baseline deliberately keeps BF16, SDPA, packing and gradient checkpointing
because the TraceML runs used them. It does not enable `torch.compile`, Liger,
FlashAttention, quantization or another optimization. The script uses a
single-process `torchrun` launch to match the distributed environment created
by `traceml run` as closely as possible.

The workload follows the documented TRL paths for
[Qwen3 assistant-only loss](https://huggingface.co/docs/trl/main/sft_trainer#train-on-assistant-messages-only),
[packing](https://huggingface.co/docs/trl/main/sft_trainer#packing) and
[PEFT adapters](https://huggingface.co/docs/trl/main/sft_trainer#train-adapters-with-peft).
