"""Minimal Hugging Face Trainer example with TraceMLTrainerCallback.

Run with:

    traceml run examples/integrations/huggingface_trainer_minimal.py

Use ``--steps`` to change the number of optimizer steps::

    traceml run examples/integrations/huggingface_trainer_minimal.py \
        --args --steps 20
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from traceml_ai.integrations import huggingface as traceml_hf

SEED = 42
INPUT_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = 10
NUM_SAMPLES = 4096
BATCH_SIZE = 64
MAX_STEPS = 200


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--steps",
        type=positive_int,
        default=MAX_STEPS,
        help="Number of optimizer steps (Trainer max_steps) to run.",
    )
    return parser.parse_args()


class SyntheticClassificationDataset(Dataset):
    def __init__(self, num_samples: int):
        self.x = torch.randn(num_samples, INPUT_DIM)
        self.y = torch.randint(0, NUM_CLASSES, (num_samples,))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "inputs": self.x[idx],
            "labels": self.y[idx],
        }


class TinyMLPForTrainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs=None, labels=None):
        logits = self.net(inputs)
        loss = None

        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
        }


def main() -> None:
    args = parse_args()
    torch.manual_seed(SEED)

    traceml_hf.init()

    model = TinyMLPForTrainer()
    train_dataset = SyntheticClassificationDataset(NUM_SAMPLES)

    training_args = TrainingArguments(
        output_dir="./hf_minimal_output",
        per_device_train_batch_size=BATCH_SIZE,
        max_steps=args.steps,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[traceml_hf.TraceMLTrainerCallback()],
    )

    trainer.train()


if __name__ == "__main__":
    main()
