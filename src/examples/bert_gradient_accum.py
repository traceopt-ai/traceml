import math
import os
import random

import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from traceml.decorators import trace_step

SEED = 42
MODEL_NAME = "bert-base-uncased"

MAX_TRAIN_EXAMPLES = 10000
MAX_VAL_EXAMPLES = 0

BATCH_SIZE = 32
GRAD_ACC_STEPS = 4

EPOCHS = 10
LR = 2e-5
WARMUP_RATIO = 0.06


def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == labels).sum()
    total = labels.size(0)
    return correct / max(1, total)


def prepare_data():
    raw = load_dataset("ag_news")

    train_raw = raw["train"].select(range(min(MAX_TRAIN_EXAMPLES, len(raw["train"]))))
    val_raw = raw["test"].select(range(min(MAX_VAL_EXAMPLES, len(raw["test"]))))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    def tok(examples):
        return tokenizer(examples["text"], truncation=True)

    train_ds = train_raw.map(tok, batched=True, remove_columns=["text"])
    val_ds = val_raw.map(tok, batched=True, remove_columns=["text"])

    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator
    )
    return tokenizer, train_loader, val_loader


def load_batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def main():
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    dtype = torch.float16 if use_amp else torch.float32

    tokenizer, train_loader, _ = prepare_data()

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=4
    ).to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=LR)

    # Scheduler should step per optimizer-step (not per micro-batch)
    opt_steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACC_STEPS)
    total_opt_steps = EPOCHS * opt_steps_per_epoch
    warmup_steps = int(WARMUP_RATIO * total_opt_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_opt_steps,
    )

    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    global_step = 0

    # Turn DataLoader into an iterator so we can pull GRAD_ACC_STEPS micro-batches per optimizer step
    for epoch in range(EPOCHS):
        running_loss = 0.0
        running_acc = 0.0

        dl_iter = iter(train_loader)
        micro_seen = 0

        while True:
            # Try to build one optimizer step from up to GRAD_ACC_STEPS micro-batches
            micro_batches = []
            for _ in range(GRAD_ACC_STEPS):
                try:
                    micro_batches.append(next(dl_iter))
                except StopIteration:
                    break

            if not micro_batches:
                break  # epoch done

            optimizer.zero_grad(set_to_none=True)

            # ONE TraceML step == GRAD_ACC_STEPS micro-steps + optimizer step
            with trace_step(model):
                last_logits = None
                last_labels = None
                total_step_loss = 0.0

                for mb in micro_batches:
                    mb = load_batch_to_device(mb, device)

                    with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):
                        out = model(**mb)
                        # scale loss down so accumulated grad matches non-accum
                        loss = out.loss / GRAD_ACC_STEPS
                        logits = out.logits

                    scaler.scale(loss).backward()

                    total_step_loss += loss.detach()
                    last_logits = logits
                    last_labels = mb["labels"]
                    micro_seen += 1

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            # Logging on optimizer steps (use last micro-batch logits/labels)
            # Report unscaled loss (multiply back)
            step_loss_unscaled = (total_step_loss * GRAD_ACC_STEPS).item()
            acc = accuracy_from_logits(last_logits.detach(), last_labels).item()

            running_loss += step_loss_unscaled
            running_acc += acc
            global_step += 1

            if global_step % 50 == 0:
                avg_loss = running_loss / 50
                avg_acc = running_acc / 50
                print(
                    f"[Train] epoch {epoch+1} step {global_step} "
                    f"| loss {avg_loss:.4f} | acc {avg_acc:.4f}"
                )
                running_loss = 0.0
                running_acc = 0.0

        print(f"Finished epoch {epoch + 1}")

    save_dir = "./bert_agnews_gradacc"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved model to {save_dir}")


if __name__ == "__main__":
    main()