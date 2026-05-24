# Hugging Face Trainer

Use TraceML with Hugging Face `Trainer` to find training bottlenecks without rewriting your training loop.

`TraceMLTrainer` is a drop-in replacement for `transformers.Trainer`. It adds step-aware diagnosis so you can quickly tell whether a run is input-bound, compute-bound, straggler-heavy, or showing memory drift.

> Start with the [Quickstart](../quickstart.md) if you have not used TraceML yet.

---

## Install

Install TraceML with Hugging Face support:

```bash
pip install "traceml-ai[hf]"
```

If your example uses datasets:

```bash
pip install datasets
```

For vision examples:

```bash
pip install torchvision
```

---

## Basic usage

Replace `Trainer` with `TraceMLTrainer`. Everything else stays the same.

```python
from traceml.integrations.huggingface import TraceMLTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=32,
    num_train_epochs=3,
    report_to="none",
    disable_tqdm=True,
)

trainer = TraceMLTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    traceml_enabled=True,
)

trainer.train()
```

Run with:

```bash
traceml run fine_tune.py
```

Or open the local UI:

```bash
traceml run fine_tune.py --mode=dashboard
```

Dashboard mode is intended for single-node runs, including single-node
multi-GPU.

---

## What TraceML will show

In Hugging Face runs, TraceML helps you spot:

- input-bound training
- compute-bound steps
- DDP rank stragglers
- wait-heavy behavior
- memory creep over time

You keep the normal Hugging Face workflow. TraceML adds diagnosis around the training step.

---

## How it works

`TraceMLTrainer` subclasses `transformers.Trainer` and wraps the training step automatically.

That means you do not need to add `traceml.trace_step(...)` manually in your
Hugging Face training loop.

If `traceml_enabled=False`, it behaves like a normal `Trainer`.

---

## Use with your existing tracking stack

TraceML is designed to work alongside tools like W&B, MLflow, and TensorBoard.

A common setup is:

- Hugging Face Trainer for training
- W&B / TensorBoard for experiment tracking
- TraceML for bottleneck diagnosis

For the cleanest terminal experience, you can set:

```python
report_to="none"
disable_tqdm=True
```

This is optional. TraceML does not require you to replace your existing logger stack.

---

## Multi-GPU DDP

`TraceMLTrainer` inherits DDP support from Hugging Face `Trainer`.

For a single-node run, launch with:

```bash
traceml run fine_tune.py --nproc-per-node=4
```

For a multi-node summary run, use the same `--run-name`, `--nnodes`,
`--nproc-per-node`, and `--master-addr` on every node, changing only
`--node-rank`:

```bash
traceml run fine_tune.py \
  --nnodes=2 \
  --node-rank=0 \
  --nproc-per-node=4 \
  --master-addr=<node0-ip> \
  --run-name=my-run
```

Run the same command on each node, changing only `--node-rank`.
`--session-id` remains accepted as a backward-compatible alias for
`--run-name`.

TraceML can help surface:

- rank imbalance
- input stragglers
- compute stragglers
- memory skew

> Multi-node runs currently produce end-of-run summary reports. Live CLI and
> dashboard views are intended for single-node runs.

---

## Full NLP example

Save as `fine_tune_nlp.py`:

```python
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)

from traceml.integrations.huggingface import TraceMLTrainer


def main():
    model_name = "prajjwal1/bert-mini"
    output_dir = "./hf_nlp_output"
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

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

    trainer = TraceMLTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        traceml_enabled=True,
    )

    trainer.train()
    print("Done.")


if __name__ == "__main__":
    main()
```

Run with:

```bash
traceml run fine_tune_nlp.py
```

---

## Full vision example

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
    TrainingArguments,
)

from traceml.integrations.huggingface import TraceMLTrainer


def main():
    model_name = "google/vit-base-patch16-224-in21k"
    output_dir = "./hf_vision_output"
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

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

    trainer = TraceMLTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DefaultDataCollator(),
        traceml_enabled=True,
    )

    trainer.train()
    print("Done.")


if __name__ == "__main__":
    main()
```

Run with:

```bash
traceml run fine_tune_vision.py
```

---

## Recommended `TrainingArguments` settings

These settings make the terminal output cleaner when using TraceML:

| Setting | Recommended value | Why |
|---|---|---|
| `disable_tqdm=True` | Yes | Prevents tqdm from fighting with the TraceML CLI |
| `report_to="none"` | Optional | Keeps W&B / TensorBoard output out of the terminal for local diagnosis |
| `save_strategy="no"` | Optional | Useful for short local diagnostic runs |

---

## Troubleshooting

### Terminal output overlaps with TraceML

Set:

```python
disable_tqdm=True
```

This gives the TraceML CLI cleaner terminal control.

### I still want W&B or TensorBoard

That is fine. TraceML is designed to work alongside them.

If terminal output gets noisy, use:

```bash
traceml run fine_tune.py --mode=dashboard
```

Dashboard mode is intended for single-node runs. For multi-node runs, use the
default final summary path.

### Multi-GPU run only shows one rank

Make sure you launch through TraceML, not plain `python`:

```bash
traceml run fine_tune.py --nproc-per-node=4
```

### I want a baseline without TraceML

Run:

```bash
traceml run fine_tune.py --disable-traceml
```

This launches your script natively through `torchrun` without TraceML telemetry.

---

## Reference

`TraceMLTrainer` accepts:

- everything that normal `transformers.Trainer` accepts
- `traceml_enabled=True|False`

---

## Next steps

- Read the [Quickstart](../quickstart.md) for plain PyTorch training loops
- Read [lightning.md](lightning.md) for the PyTorch Lightning integration
- Open an issue if you hit a problem: https://github.com/traceopt-ai/traceml/issues
