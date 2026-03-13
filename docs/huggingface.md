# Hugging Face Trainer Integration

TraceML provides `TraceMLTrainer`, a drop-in replacement for `transformers.Trainer` that adds step-level timing and memory visibility to any HF training run.

> **Prerequisites:** Follow the [Quickstart](quickstart.md) first to confirm `traceml run` works with a plain PyTorch loop.

---

## Table of Contents

1. [Install](#1-install)
2. [How it works](#2-how-it-works)
3. [Basic usage](#3-basic-usage)
4. [Enable Deep-Dive mode](#4-enable-deep-dive-mode)
5. [NLP example: BERT fine-tuning on AG News](#5-nlp-example-bert-fine-tuning-on-ag-news)
6. [Vision example: ViT fine-tuning on CIFAR-10](#6-vision-example-vit-fine-tuning-on-cifar-10)
7. [Multi-GPU DDP](#7-multi-gpu-ddp)
8. [TrainingArguments tips](#8-trainingarguments-tips)
9. [TraceMLTrainer reference](#9-tracemltrainer-reference)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Install

Install TraceML with the `[hf]` extra. This pulls in `transformers` and `accelerate` automatically:

```bash
pip install "traceml-ai[hf]"
```

The examples in this guide also use `datasets` to load training data. Install it separately:

```bash
pip install datasets
```

For vision tasks (Section 6), you also need `torchvision`:

```bash
pip install torchvision
```

---

## 2. How it works

`TraceMLTrainer` subclasses `transformers.Trainer` and overrides `training_step`. It wraps each step with `trace_step(model)` automatically, so you do not need to change your training loop at all.

```
Your script
    TraceMLTrainer.train()
        TraceMLTrainer.training_step()      <-- overridden
            with trace_step(model):         <-- injected by TraceML
                Trainer.training_step()     <-- original HF logic unchanged
```

If `traceml_enabled=False`, the class becomes a transparent pass-through to the original `Trainer` with zero overhead.

---

## 3. Basic usage

Replace `Trainer` with `TraceMLTrainer`. Everything else stays the same.

```python
from traceml.integrations.huggingface import TraceMLTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=32,
    num_train_epochs=3,
    report_to="none",       # Disables wandb / tensorboard for this example
    disable_tqdm=True,      # Lets the TraceML dashboard own the terminal output
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

Then launch with the CLI:

```bash
traceml run fine_tune.py
```

Or with the web dashboard:

```bash
traceml run fine_tune.py --mode=dashboard
```

---

## 4. Enable Deep-Dive mode

Deep-Dive attaches per-layer hooks to the model and surfaces forward/backward timing and memory signals for each layer. Pass a `traceml_kwargs` dict to enable it.

```python
trainer = TraceMLTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    traceml_enabled=True,
    traceml_kwargs={
        "sample_layer_memory": True,          # Track parameter memory per layer
        "trace_layer_forward_memory": True,   # Activation memory per layer (forward)
        "trace_layer_forward_time": True,     # Forward pass time per layer
        "trace_layer_backward_time": True,    # Backward pass time per layer
    },
)
```

Hooks are attached **lazily on the first step**, so they see the final model state (after any internal HF wrapping).

> **Overhead note:** Deep-Dive hooks add a small amount of overhead per layer per step. Use it for diagnostic runs; for long production runs you can keep `traceml_kwargs=None` (the default) and rely on step-level signals only.

---

## 5. NLP example: BERT fine-tuning on AG News

A complete example using `bert-mini` on the AG News text classification dataset.

**Install dependencies:**

```bash
pip install "traceml-ai[hf]" datasets
```

**Save as `fine_tune_nlp.py`:**

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

    # Model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=4
    ).to(device)

    # Dataset (small subset for a quick demo)
    raw_dataset = load_dataset("ag_news", split="train[:2000]")

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=64,
        )

    dataset = raw_dataset.map(tokenize, batched=True)

    # Training arguments
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

    # Trainer with Deep-Dive enabled
    trainer = TraceMLTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        traceml_enabled=True,
        traceml_kwargs={
            "sample_layer_memory": True,
            "trace_layer_forward_memory": True,
            "trace_layer_forward_time": True,
            "trace_layer_backward_time": True,
        },
    )

    trainer.train()
    print("Done.")


if __name__ == "__main__":
    main()
```

**Run it:**

```bash
traceml run fine_tune_nlp.py
```

---

## 6. Vision example: ViT fine-tuning on CIFAR-10

A complete example using `vit-base-patch16-224` on CIFAR-10.

**Install dependencies:**

```bash
pip install "traceml-ai[hf]" datasets torchvision
```

**Save as `fine_tune_vision.py`:**

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

    # Model and image processor
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=10,
        id2label={i: str(i) for i in range(10)},
        label2id={str(i): i for i in range(10)},
        ignore_mismatched_sizes=True,
    ).to(device)

    # Dataset (small subset for a quick demo)
    dataset = load_dataset("cifar10", split="train[:500]")

    # Image transforms
    normalize = Normalize(
        mean=image_processor.image_mean,
        std=image_processor.image_std,
    )
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

    def apply_transforms(examples):
        examples["pixel_values"] = [
            _transforms(img.convert("RGB")) for img in examples["img"]
        ]
        del examples["img"]
        return examples

    dataset = dataset.with_transform(apply_transforms)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        num_train_epochs=1,
        logging_steps=5,
        save_strategy="no",
        use_cpu=(device == "cpu"),
        report_to="none",
        disable_tqdm=True,
        remove_unused_columns=False,  # Required for vision datasets
    )

    # Trainer with Deep-Dive enabled
    trainer = TraceMLTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DefaultDataCollator(),
        traceml_enabled=True,
        traceml_kwargs={
            "sample_layer_memory": True,
            "trace_layer_forward_memory": True,
            "trace_layer_forward_time": True,
            "trace_layer_backward_time": True,
        },
    )

    trainer.train()
    print("Done.")


if __name__ == "__main__":
    main()
```

**Run it:**

```bash
traceml run fine_tune_vision.py
```

---

## 7. Multi-GPU DDP

`TraceMLTrainer` inherits DDP support directly from HF `Trainer`. No code changes are needed. Just pass `--nproc-per-node` to the CLI.

```bash
# 4-GPU run
traceml run fine_tune_nlp.py --nproc-per-node=4
```

In the TraceML dashboard you will see:

- Per-step metrics aggregated **across all ranks**
- **Median** step time (typical behavior)
- **Worst rank** (slowest GPU)
- **Skew %** (imbalance between ranks)

> **Limitation:** Single-node DDP only. Multi-node is not yet supported.

---

## 8. TrainingArguments tips

A few `TrainingArguments` settings interact with TraceML.

| Setting | Recommended value | Why |
|---------|-------------------|-----|
| `disable_tqdm=True` | Yes | Prevents tqdm from fighting with the Rich terminal dashboard for output. |
| `report_to="none"` | Yes (for local runs) | Disables wandb / tensorboard if you do not need them alongside TraceML. |
| `logging_steps` | Any | TraceML steps are independent of HF logging steps. Both work simultaneously. |
| `use_cpu` | Set based on `torch.cuda.is_available()` | Controls which backend HF uses internally; TraceML will pick up GPU signals automatically. |

---

## 9. TraceMLTrainer reference

`TraceMLTrainer` is a subclass of `transformers.Trainer`. It accepts all standard `Trainer` arguments plus two extra parameters.

```python
TraceMLTrainer(
    # All standard Trainer arguments (model, args, train_dataset, ...)
    ...,

    # TraceML-specific
    traceml_enabled: bool = True,
    traceml_kwargs: dict | None = None,
)
```

### `traceml_enabled`

Set to `False` to disable all TraceML instrumentation without removing `TraceMLTrainer` from your code. Useful for A/B runs or CI environments.

```python
trainer = TraceMLTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    traceml_enabled=False,   # Transparent pass-through to Trainer
)
```

### `traceml_kwargs`

Each key maps directly to a parameter of `trace_model_instance` in `traceml/decorators.py`. All keys are optional.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `sample_layer_memory` | `bool` | `True` | Enqueue model for parameter memory sampling per layer. |
| `trace_layer_forward_memory` | `bool` | `True` | Attach activation hooks to capture forward-pass memory per layer. |
| `trace_layer_backward_memory` | `bool` | `True` | Attach gradient hooks to capture backward-pass memory per layer (module + param). |
| `trace_layer_forward_time` | `bool` | `True` | Attach forward timing hooks (pre + post) per layer. |
| `trace_layer_backward_time` | `bool` | `True` | Attach backward timing hooks (pre + post) per layer. |
| `trace_execution` | `bool` | `True` | Attach execution entry hooks to the model. |
| `include_names` | `list[str] or None` | `None` | If set, only trace layers whose names appear in this list. |
| `exclude_names` | `list[str] or None` | `None` | If set, skip layers whose names appear in this list. |
| `leaf_only` | `bool` | `True` | When `True`, hooks are attached only to leaf modules, not to container layers. |

---

## 10. Troubleshooting

### "transformers is not installed"

```
ImportError: TraceMLTrainer requires 'transformers' to be installed.
```

Fix:

```bash
pip install transformers
```

---

### tqdm output overlaps with TraceML dashboard

Set `disable_tqdm=True` in your `TrainingArguments`. This gives the TraceML dashboard clean control over terminal output.

---

### Hooks show `None` or empty layer names

This usually means `traceml_kwargs` was passed but the model was not yet moved to the correct device when hooks were attached. `TraceMLTrainer` attaches hooks **lazily on the first step**, so this should be rare. If you see it, open an issue with your model class name and PyTorch version.

---

### Multi-GPU run: only rank 0 shows metrics

This is expected if you launched with plain `python` instead of `traceml run`. TraceML uses `torchrun` internally to coordinate ranks. Always launch with:

```bash
traceml run fine_tune_nlp.py --nproc-per-node=4
```

---

## Next steps

- Read the [Quickstart](quickstart.md) for plain PyTorch training loops
- Browse full example scripts in [`src/examples/`](../src/examples/)
- [Open an issue](https://github.com/traceopt-ai/traceml/issues) if you hit a problem not covered here
