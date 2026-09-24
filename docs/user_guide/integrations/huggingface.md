# Hugging Face Trainer Integration

Use TraceML with Hugging Face `Trainer` without rewriting your training loop.

The integration is two steps: call
`traceml_ai.integrations.huggingface.init()` once, then pass
`TraceMLTrainerCallback` (a standard `transformers.TrainerCallback`) to your
existing `Trainer`.

## 1. Install

```bash
pip install "traceml-ai[hf]"
```

The `hf` extra installs `transformers>=4.46.1`, the minimum supported
Transformers version for automatic Trainer timing.

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

Accelerate prepares each training batch before the Trainer's step callback.
TraceML measures those host-to-device transfers as short parts of Traced Step
Time and combines them with the later forward, backward, and optimizer work.
DataLoader waiting remains separate Input Wait, including blocking look-ahead
fetches, and is not included in the transfer regions. The timing scope opens
only while the standard Trainer requests training batches; evaluation and
prediction input work is excluded, including direct `trainer.evaluate()` and
`trainer.predict(...)` calls made outside training.

TraceML follows HF's completed-step count even when mixed-precision overflow
skips a parameter update. Step numbers are local to the process, so they may
differ from HF's `global_step` after checkpoint resume. See the
[developer guide](../../developer_guide/step-time-pipeline-contract.md#hugging-face-steps)
for the exact timing boundaries and optimizer behavior.

When resuming an iterable dataset, Accelerate may consume checkpoint-skipped
batches and the first real batch inside one iterator call. TraceML does not
attribute that mixed work to the real training step: the entire first resumed
optimizer group is omitted from step timing and memory telemetry, including
all its accumulation microbatches. Training executes normally; recording starts
with the next group's input collection. No partial step or zero-valued
placeholder is published. Fresh runs, map-style sampler skipping, and
`ignore_data_skip=True` record their first group normally.

If training stops before an accumulation group completes, TraceML discards
that group. Cleanup happens before an automatic batch-size retry, and the same
callback can be reused by a later Trainer run.

## Limitations

- **Transformers version.** Automatic Trainer timing requires
  `transformers>=4.46.1`, where `Trainer.get_batch_samples` is available. If an
  older version is installed manually, TraceML warns once and lets training
  continue, but omits training Input Wait and pre-step H2D measurements.
- **Lifecycle guard.** Failure/retry cleanup and duplicate-callback handling
  require `traceml_hf.init()` to install the Trainer lifecycle guard. If guard
  installation fails, TraceML reports the error and training continues without
  those guarantees. Custom `_inner_training_loop` overrides must call the
  guarded parent implementation to receive this handling.
- **Custom batch collection.** A Trainer subclass that overrides
  `get_batch_samples` bypasses the standard training-input hook. TraceML warns
  once for the affected Trainer class and continues without training Input
  Wait or pre-step H2D signals rather than reporting partial measurements.
- **Iterable checkpoint resume.** When Accelerate lazily consumes skipped
  batches, the entire first resumed optimizer group is omitted from step
  telemetry. It still trains normally; subsequent groups are recorded.
- **Memory window.** Temporary allocation peaks before the callback starts
  a step are outside its memory measurement.
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

## Migration

`TraceMLTrainer` was intentionally removed because it only installed
`TraceMLTrainerCallback`. Replace it with `transformers.Trainer`, call
`traceml_hf.init()` once, and add `traceml_hf.TraceMLTrainerCallback()` to the
Trainer's callbacks, as shown above. Keep the rest of your Trainer arguments.

The `traceml_enabled` argument was removed with the wrapper. For optional
tracing, control initialization and callback registration in your code:

```python
enable_tracing = True
callbacks = []  # Add any other Trainer callbacks here.
if enable_tracing:
    traceml_hf.init()
    callbacks.append(traceml_hf.TraceMLTrainerCallback())

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=callbacks,
)
trainer.train()
```

To disable TraceML for an entire launched run, use `--disable-traceml` as
shown in Troubleshooting.

## Next Steps

- [How to Read Output](../reading-output.md)
- [Distributed Training](../distributed-training.md)
- [W&B / MLflow](wandb-mlflow.md)
- [Open an issue](https://github.com/traceopt-ai/traceml/issues)
