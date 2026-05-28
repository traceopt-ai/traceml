# Hugging Face Trainer Integration

Use TraceML with Hugging Face `Trainer` without rewriting your training loop.

The preferred integration is `TraceMLTrainerCallback`: a standard
`transformers.TrainerCallback` you pass to your existing `Trainer`. The legacy
`TraceMLTrainer` subclass is still supported. It is now a thin wrapper that
installs the same callback under the hood.

## 1. Install

```bash
pip install "traceml-ai[hf]"
```

If you are running the full examples below, install their optional dependencies:

```bash
pip install datasets torchvision
```

## 2. Add `TraceMLTrainerCallback` To Your `Trainer`

Register the callback alongside `transformers.Trainer`:

```python
from traceml_ai.integrations.huggingface import TraceMLTrainerCallback
from transformers import Trainer, TrainingArguments

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
    callbacks=[TraceMLTrainerCallback()],
)

trainer.train()
```

You do not need to add `traceml.trace_step(...)` manually. The callback opens
and closes `trace_step` around each optimizer step, and TraceML's existing
auto-timers capture forward, backward, h2d, and optimizer phases inside that
bracket.

### Legacy `TraceMLTrainer`

The `TraceMLTrainer(Trainer)` subclass remains supported for users who already
adopted it. It is now a thin wrapper that auto-installs
`TraceMLTrainerCallback` on construction and accepts the same `traceml_enabled`
and `traceml_kwargs` arguments as before:

```python
from traceml_ai.integrations.huggingface import TraceMLTrainer

trainer = TraceMLTrainer(
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

## Limitations

The callback path is the right default for most users, but it has structural
trade-offs vs. the legacy subclass that are worth knowing:

- **Step granularity.** One TraceML step equals one optimizer step. With
  `gradient_accumulation_steps=N`, forward and backward times from all `N`
  accumulated micro-batches fold into a single TraceML step. The pre-refactor
  `TraceMLTrainer.training_step` override counted each micro-batch as its own
  TraceML step. If you need per-micro-batch attribution under gradient
  accumulation, open an issue.
- **Optimizer timing.** Captured by TraceML's global optimizer hooks, which
  are installed automatically only when running under the default
  `traceml.init(mode="auto")` path (or with no explicit `init` call). Under
  `manual` or `selective` modes optimizer events are not emitted, but this
  applies consistently to every step, so dashboard step alignment still holds.
- **Exception safety.** If `training_step` raises, Hugging Face does not call
  `on_step_end`. The callback defensively closes the trace_step context on
  the next `on_step_begin` and on `on_train_end`, but the step where the
  exception occurred may be reported with incomplete timing or memory. The
  legacy subclass's `with trace_step(model): super().training_step(...)`
  pattern ran the `finally` cleanup deterministically. If you need precise
  attribution on failing steps, the legacy `TraceMLTrainer` path is stricter.
- **Wrapped models (DDP / FSDP).** Hugging Face passes the unwrapped
  `trainer.model` to callbacks, not `trainer.model_wrapped`. For plain DDP
  this is harmless (submodule forward hooks still fire), but for FSDP the
  flattened parameter layout can change which forwards the deep-profile
  hooks observe. Use the legacy `TraceMLTrainer` for FSDP deep-mode runs.
- **Callback registration timing.** Pass `TraceMLTrainerCallback()` at
  `Trainer(...)` construction. Callbacks added via `trainer.add_callback(...)`
  *after* `trainer.train()` has started will not receive `on_train_begin`,
  which means deep-mode lazy hook attachment will be skipped for the first
  step.

## Troubleshooting

### Terminal output overlaps with TraceML

Set `disable_tqdm=True` in `TrainingArguments`.

If output is still noisy, use browser dashboard mode on single-node runs:

```bash
pip install "traceml-ai[dashboard]"
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

from traceml_ai.integrations.huggingface import TraceMLTrainerCallback


def main():
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
        callbacks=[TraceMLTrainerCallback()],
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

from traceml_ai.integrations.huggingface import TraceMLTrainerCallback


def main():
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
        callbacks=[TraceMLTrainerCallback()],
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

`TraceMLTrainerCallback(traceml_kwargs=None)` accepts:

- `traceml_kwargs`: optional dict forwarded to `trace_model_instance` for
  deep-profile layer-level tracing. `None` (the default) means standard
  step-level instrumentation only.

`TraceMLTrainer` (legacy thin wrapper) accepts:

- everything that normal `transformers.Trainer` accepts
- `traceml_enabled=True|False`
- `traceml_kwargs={...}`: forwarded to the installed callback

## Next Steps

- [How to Read Output](../reading-output.md)
- [Distributed Training](../distributed-training.md)
- [W&B / MLflow](wandb-mlflow.md)
- [Open an issue](https://github.com/traceopt-ai/traceml/issues)
