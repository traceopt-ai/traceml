# Hugging Face Trainer Integration

## Install

This guide assumes PyTorch and Hugging Face Transformers are already installed.
Use Transformers 4.46.1 or newer for training input timing.

```bash
pip install traceml-ai
```

## Run

```bash
traceml run train.py
```

Run your existing script with TraceML. Standard `Trainer` training is
instrumented automatically, with no code changes required. For custom training loops or direct
`python`/`torchrun` launches, see
[Advanced: manual setup](#advanced-manual-setup).

## Read the result

When training finishes, TraceML prints a diagnosis of the likely bottleneck
and saves the report for comparison or CI.

The timing breakdown shows input waiting, forward, backward, and optimizer
work. CUDA runs also show available host-to-device and memory measurements.

See [How to Read Output](../reading-output.md) for an example report and
explanations.

## How TraceML measures Hugging Face training

TraceML automatically attaches its Trainer callback and timing hooks.
Hugging Face continues to manage batch collection, gradient accumulation,
and optimizer updates.

```text
Hugging Face Trainer                 TraceML measurement
─────────────────────────────────────────────────────────
get_batch_samples()                  Input Wait + GPU transfer
          ↓
on_step_begin()                       Open the step capture
          ↓
Forward → Backward                    Record compute timings
(repeated for each microbatch)
          ↓
Optimizer                            Record optimizer timing
          ↓
on_step_end()                         Complete one reported step
```

One reported step covers one optimizer update attempt. With
`gradient_accumulation_steps=4`, four microbatches contribute to that step.
TraceML adds their timings and records peak CUDA memory across the group.
The final group can contain fewer microbatches.

Batch fetching appears separately as **Input Wait**; GPU transfer and compute
contribute to traced training time. Evaluation and prediction are excluded.

**Checkpoint resume:** Continue using Hugging Face's normal
`trainer.train(resume_from_checkpoint=...)`. TraceML records resumed training
with step numbering local to the new run, which may differ from Hugging
Face's `global_step`. See [Limitations](#limitations) for iterable-dataset
resume handling, or the
[developer guide](../../developer_guide/step-time-pipeline-contract.md#hugging-face-steps)
for exact timing boundaries.

## Multi-GPU training

For single-node multi-GPU DDP, specify the number of processes:

```bash
traceml run train.py --nproc-per-node=4
```

For multi-node DDP launch commands, see
[Distributed Training](../distributed-training.md).

## Advanced: manual setup

Existing scripts using `init()` and `TraceMLTrainerCallback()` also work with
`traceml run`; TraceML reuses compatible setup without adding another callback.

Use this path for a direct `python`/`torchrun` launch or a custom Trainer loop
that still dispatches Hugging Face callbacks. A direct launch needs a running
aggregator; start one with `traceml serve` first (see
[Direct Launch](../public-api.md#direct-launch-with-traceml-serve)). Call
`init()` once before constructing the `Trainer`, then register
`TraceMLTrainerCallback`:

```python
from traceml_ai.integrations import huggingface as traceml_hf
from transformers import Trainer, TrainingArguments

traceml_hf.init()

training_args = TrainingArguments(
    output_dir="./output",
    report_to="none",
    disable_tqdm=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[traceml_hf.TraceMLTrainerCallback()],
)

trainer.train()
```

Use both: the callback groups measurements into completed steps, and `init()`
installs the process-wide timing patches. Custom loops may still bypass input
timing, as described in [Limitations](#limitations). You do not need to add
`traceml.trace_step(...)` to your code.
If a custom loop does not dispatch Hugging Face callbacks, instrument that
loop with the [core API](../public-api.md) instead.

### Full manual examples

Use these examples when you want a complete runnable manual setup. If you
already have a standard Hugging Face training script, start with `traceml run`.

Install the examples' optional dependencies first:

```bash
pip install datasets torchvision
```

<details>
<summary>NLP classification example</summary>

Save as `fine_tune_nlp.py`:

```python
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from traceml_ai.integrations import huggingface as traceml_hf


def main():
    traceml_hf.init()

    model_name = "prajjwal1/bert-mini"
    output_dir = "./hf_nlp_output"
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4,
    ).to(device)

    raw_dataset = load_dataset("ag_news", split="train[:2000]")

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=64,
        )

    dataset = raw_dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="no",
        use_cpu=(device == "cpu"),
        report_to="none",
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[traceml_hf.TraceMLTrainerCallback()],
    )

    trainer.train()


if __name__ == "__main__":
    main()
```

Run with:

```bash
traceml run fine_tune_nlp.py
```

</details>

<details>
<summary>Vision classification example</summary>

Save as `fine_tune_vision.py`:

```python
import os

import torch
from datasets import load_dataset
from torchvision.transforms import Compose, Normalize, RandomResizedCrop, ToTensor
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

from traceml_ai.integrations import huggingface as traceml_hf


def main():
    traceml_hf.init()

    model_name = "google/vit-base-patch16-224-in21k"
    output_dir = "./hf_vision_output"
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=10,
    ).to(device)

    dataset = load_dataset("cifar10", split="train[:2000]")
    transform = Compose(
        [
            RandomResizedCrop(
                image_processor.size["height"],
                scale=(0.8, 1.0),
            ),
            ToTensor(),
            Normalize(
                mean=image_processor.image_mean,
                std=image_processor.image_std,
            ),
        ]
    )

    def preprocess(example):
        image = example["img"].convert("RGB")
        example["pixel_values"] = transform(image)
        example["labels"] = example["label"]
        return example

    dataset = dataset.map(preprocess)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        num_train_epochs=2,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DefaultDataCollator(),
        callbacks=[traceml_hf.TraceMLTrainerCallback()],
    )

    trainer.train()


if __name__ == "__main__":
    main()
```

Run with:

```bash
traceml run fine_tune_vision.py
```

</details>

### API reference

For manual setup, `init()` takes no arguments. Call it once before
constructing the `Trainer` to install TraceML's process-wide patches
(`DataLoader` fetch timing, H2D `Tensor.to`, and the forward/backward/optimizer
auto-timers). It is idempotent and returns the effective `TraceMLInitConfig`.

`TraceMLTrainerCallback()` takes no TraceML-specific arguments and records
standard step-level timing and memory.

## Limitations

- **Training input timing.** The input collection hook requires
  `transformers>=4.46.1`, where `Trainer.get_batch_samples` is available. If an
  older version is installed manually, TraceML warns once and lets training
  continue, but omits training Input Wait and pre-step H2D measurements.
- **Lifecycle guard.** Failure/retry cleanup and duplicate-callback handling
  require automatic launch or `traceml_hf.init()` to install the Trainer
  lifecycle guard. If installation fails, TraceML reports the error and
  training continues without those guarantees. Custom `_inner_training_loop`
  overrides must call the guarded parent implementation to receive this handling.
- **Custom batch collection.** A Trainer subclass that overrides
  `get_batch_samples` bypasses the standard training-input hook. TraceML warns
  once for the affected Trainer class and continues without training Input
  Wait or pre-step H2D signals rather than reporting partial measurements.
- **Iterable checkpoint resume.** When Accelerate lazily consumes skipped
  batches, the entire first resumed optimizer group is omitted from step
  telemetry. It still trains normally; subsequent groups are recorded.
- **Memory window.** Temporary allocation peaks before the callback starts
  a step are outside its memory measurement.
- **Custom loops.** Automatic attachment requires the standard
  `_inner_training_loop`. Register the callback before `trainer.train()` for
  custom loops that drive Hugging Face callbacks themselves.

## Next Steps

- [How to Read Output](../reading-output.md)
- [Compare Runs](../compare.md)
- [Catch Regressions in CI](../regression-guard.md)
- [Distributed Training](../distributed-training.md)
- [W&B / MLflow](wandb-mlflow.md)
- [Open an issue](https://github.com/traceopt-ai/traceml/issues)
