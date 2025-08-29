import os
import random
import math
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

# TraceML imports
from traceml.decorator import trace_model_instance

# Optional: if you added StepTimer earlier
try:
    from traceml.utils.gradient_time import StepTimer
except Exception:
    StepTimer = None  # timing is optional


SEED = 42
MODEL_NAME = "distilbert-base-uncased"  # light & laptop-friendly
MAX_TRAIN_EXAMPLES = 2000  # keep small for quick runs
MAX_VAL_EXAMPLES = 800
BATCH_SIZE = 32
EPOCHS = 2
LR = 2e-5
WARMUP_RATIO = 0.06


def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def prepare_data():
    """
    Load a tiny AG News subset and tokenize.
    AG News has fields: 'text' and 'label'.
    """
    raw = load_dataset("ag_news")
    # Trim to a tiny subset for quick laptop runs
    train_raw = raw["train"].select(range(min(MAX_TRAIN_EXAMPLES, len(raw["train"]))))
    val_raw = raw["test"].select(range(min(MAX_VAL_EXAMPLES, len(raw["test"]))))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok(examples):
        return tokenizer(examples["text"], truncation=True)

    train_ds = train_raw.map(tok, batched=True, remove_columns=["text"])
    val_ds = val_raw.map(tok, batched=True, remove_columns=["text"])

    # Rename 'label' → 'labels' for HF models
    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator
    )
    return tokenizer, train_loader, val_loader


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == labels).sum().item()
    return correct / max(1, labels.size(0))


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer, train_loader, val_loader = prepare_data()

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,  # AG News has 4 classes
    ).to(device)

    # Attach TraceML hooks so your samplers/loggers receive events
    trace_model_instance(
        model,
        sample_layer_memory=True,
        trace_activations=True,
        trace_gradients=True,
    )

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = EPOCHS * math.ceil(len(train_loader))
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    model.train()
    global_step = 0
    for epoch in range(EPOCHS):
        running_loss = 0.0
        running_acc = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Optional timing for your GradientTimeSampler if available
            if StepTimer is not None:
                with StepTimer(model, label="train") as t:
                    if hasattr(t, "mark_backward_start"):
                        t.mark_backward_start()
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(
                        enabled=torch.cuda.is_available(), dtype=dtype
                    ):
                        out = model(**batch)
                        loss = out.loss
                        logits = out.logits

                    if hasattr(t, "mark_backward_done"):
                        pass
                    scaler.scale(loss).backward()
                    if hasattr(t, "mark_backward_done"):
                        t.mark_backward_done()

                    if hasattr(t, "mark_optimizer_step_start"):
                        t.mark_optimizer_step_start()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    if hasattr(t, "mark_optimizer_step_done"):
                        t.mark_optimizer_step_done()
            else:
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(
                    enabled=torch.cuda.is_available(), dtype=dtype
                ):
                    out = model(**batch)
                    loss = out.loss
                    logits = out.logits
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            acc = accuracy_from_logits(logits.detach(), batch["labels"])
            running_loss += loss.item()
            running_acc += acc
            global_step += 1

            if global_step % 50 == 0:
                print(
                    f"[Train] epoch {epoch + 1} step {global_step} | loss {running_loss / 50:.4f} | acc {running_acc / 50:.4f}"
                )
                running_loss = 0.0
                running_acc = 0.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.cuda.amp.autocast(
                    enabled=torch.cuda.is_available(), dtype=dtype
                ):
                    out = model(**batch)
                    loss = out.loss
                    logits = out.logits
                val_loss += loss.item()
                val_acc += accuracy_from_logits(logits, batch["labels"])
                n_batches += 1

        print(
            f"[Val] epoch {epoch + 1} | loss {val_loss / max(1, n_batches):.4f} | acc {val_acc / max(1, n_batches):.4f}"
        )
        model.train()

    # Save a tiny checkpoint for demo
    save_dir = "./distilbert_agnews_small"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved model to {save_dir}")


if __name__ == "__main__":
    main()
