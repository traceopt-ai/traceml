import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import TrainerCallback, TrainingArguments

from traceml.integrations.huggingface import TraceMLTrainer

SEED = 42
INPUT_DIM = 1024
HIDDEN_DIM = 2048
NUM_CLASSES = 10
NUM_SAMPLES = 50000
BATCH_SIZE = 256
MAX_STEPS = 600
PAUSE_BETWEEN_STEPS = 0.05


class SyntheticClassificationDataset(Dataset):
    def __init__(self, num_samples: int):
        self.x = torch.randn(num_samples, INPUT_DIM)
        self.y = torch.randint(0, NUM_CLASSES, (num_samples,))

    def __len__(self):
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
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
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


class SlowDownCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        time.sleep(PAUSE_BETWEEN_STEPS)
        return control


def main():
    torch.manual_seed(SEED)

    model = TinyMLPForTrainer()
    train_dataset = SyntheticClassificationDataset(NUM_SAMPLES)

    training_args = TrainingArguments(
        output_dir="./hf_minimal_output",
        per_device_train_batch_size=BATCH_SIZE,
        max_steps=MAX_STEPS,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
        remove_unused_columns=False,
    )

    trainer = TraceMLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        traceml_enabled=True,
        callbacks=[SlowDownCallback()],
    )

    trainer.train()


if __name__ == "__main__":
    main()
