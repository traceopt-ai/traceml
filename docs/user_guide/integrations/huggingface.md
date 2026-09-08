# Hugging Face Trainer Integration

Use TraceML with Hugging Face `Trainer` without rewriting your training loop.

The preferred integration is two steps: call
`traceml_ai.integrations.huggingface.init()` once, then pass
`TraceMLTrainerCallback` (a standard `transformers.TrainerCallback`) to your
existing `Trainer`. The legacy `TraceMLTrainer` subclass is still supported. It
is now a thin wrapper that installs the same callback under the hood.

## 1. Install

```bash
pip install "traceml-ai[hf]"
```

If you are running the full examples below, install their optional dependencies:

```bash
pip install datasets torchvision
```

## 2. Initialize TraceML And Add `TraceMLTrainerCallback`

Call `init()` once before constructing the `Trainer`, then register the
callback alongside `transformers.Trainer`:

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

`traceml_hf.init()` enables automatic timing. The callback groups timing and
memory measurements into training steps. Use both; you do not need to add
`traceml.trace_step(...)` to your training code.

### Legacy `TraceMLTrainer`

The `TraceMLTrainer(Trainer)` subclass remains supported for users who already
adopted it. It is now a thin wrapper that auto-installs
`TraceMLTrainerCallback` on construction and accepts `traceml_enabled` to turn
step-level instrumentation on or off:

```python
from traceml_ai.integrations import huggingface as traceml_hf

traceml_hf.init()

trainer = traceml_hf.TraceMLTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    traceml_enabled=True,
)
trainer.train()
```

New code should prefer the direct callback registration shown above.

## 3. Launch The Run

Single GPU:

```bash
traceml run fine_tune.py
```

Single-node multi-GPU DDP:

```bash
traceml run fine_tune.py --nproc-per-node=4
```

For multi-node DDP launch commands, see
[Distributed Training](../distributed-training.md).

## Recommended `TrainingArguments`

These settings are optional, but they make local TraceML diagnostic runs easier
to read:

| Setting | Why it helps |
|---|---|
| `disable_tqdm=True` | Prevents the Hugging Face progress bar from fighting with the TraceML live CLI. |
| `report_to="none"` | Keeps tracker output out of the terminal during local diagnosis. |
| `save_strategy="no"` | Avoids checkpoint files during short diagnostic runs. |

TraceML can still run alongside W&B, MLflow, and TensorBoard. For tracker
logging patterns, see [W&B / MLflow](wandb-mlflow.md).

## What one step includes

One TraceML step covers the microbatches used for one HF optimizer update
attempt. With `gradient_accumulation_steps=4`, four microbatches share one step
number. A final group can contain fewer microbatches.

Forward and backward times are added across the group. CUDA memory reports
the peak PyTorch allocated/reserved memory on the tracked device during that
group, rather than adding memory values. CPU runs report this memory metric
as unavailable.

TraceML follows HF's completed-step count even when mixed-precision overflow
skips a parameter update. Step numbers are local to the process, so they may
differ from HF's `global_step` after checkpoint resume. See the
[developer guide](../../developer_guide/step-time-pipeline-contract.md#hugging-face-steps)
for the exact timing boundaries and optimizer behavior.

## Limitations

- **Input timing.** Accelerate can transfer batches to the GPU before the
  callback starts a step. Those H2D copies are currently missed, so Step Time
  can omit pre-step transfers. Evaluation loader fetches can also be attributed
  to the next training step.
- **Memory window.** Temporary allocation peaks before the callback starts
  a step are outside its memory measurement.
- **Interrupted training.** If training raises, tracing can remain active
  until a later cleanup call, which may record the unfinished group as a
  completed step. The legacy wrapper uses the same callback and has the same
  limitation.
- **Callback registration.** Register the callback before `trainer.train()`
  so it receives the training events from the start.

## Troubleshooting

### Terminal output overlaps with TraceML

Set `disable_tqdm=True` in `TrainingArguments`.

If output is still noisy, use browser dashboard mode on single-node runs:

```bash
traceml run fine_tune.py --mode=dashboard
```

### Multi-GPU run only shows one rank

Make sure you launched through TraceML with `--nproc-per-node`, not plain
`python`:

```bash
traceml run fine_tune.py --nproc-per-node=4
```

### I want a baseline without TraceML

Run the same script with TraceML disabled:

```bash
traceml run fine_tune.py --disable-traceml
```

This launches your script natively through `torchrun` without TraceML telemetry.

## Full Examples

Use these examples when you want a complete runnable script. If you already
have a Hugging Face training script, start with the smaller replacement pattern
above.

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

## Reference

`init()` takes no arguments. Call it once before constructing the `Trainer` to
install TraceML's process-wide patches (`DataLoader` fetch timing, H2D
`Tensor.to`, and the forward/backward/optimizer auto-timers). It is idempotent
and returns the effective `TraceMLInitConfig`.

`TraceMLTrainerCallback()` takes no TraceML-specific arguments and records
standard step-level timing and memory.

`TraceMLTrainer` (legacy thin wrapper) accepts:

- everything that normal `transformers.Trainer` accepts
- `traceml_enabled=True|False`

## Next Steps

- [How to Read Output](../reading-output.md)
- [Distributed Training](../distributed-training.md)
- [W&B / MLflow](wandb-mlflow.md)
- [Open an issue](https://github.com/traceopt-ai/traceml/issues)
