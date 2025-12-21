import os
import random
import math

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
from traceml.decorators import trace_model_instance, trace_timestep

SEED = 42
MODEL_NAME = "bert-base-uncased"

# Increase these to generate a LOT of profiling data
MAX_TRAIN_EXAMPLES = 5000
MAX_VAL_EXAMPLES   = 200
BATCH_SIZE         = 32
EPOCHS             = 2
LR = 2e-5
WARMUP_RATIO = 0.06


def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == labels).sum().item()
    return correct / max(1, labels.size(0))


def prepare_data():
    """Load & tokenize AG News (bigger subset for better logs)."""
    raw = load_dataset("ag_news")

    train_raw = raw["train"].select(range(min(MAX_TRAIN_EXAMPLES, len(raw["train"]))))
    val_raw   = raw["test"].select(range(min(MAX_VAL_EXAMPLES,   len(raw["test"]))))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok(examples):
        return tokenizer(examples["text"], truncation=True)

    train_ds = train_raw.map(tok, batched=True, remove_columns=["text"])
    val_ds   = val_raw.map(tok, batched=True, remove_columns=["text"])

    train_ds = train_ds.rename_column("label", "labels")
    val_ds   = val_ds.rename_column("label", "labels")

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator
    )
    val_loader   = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator
    )

    return tokenizer, train_loader, val_loader


# --- TraceML Wrappers ---------------------------------------------------------

@trace_timestep("dataloader_fetch", use_gpu=False)
def fetch_batch(batch):
    """Count wait time — simple wrapper for TraceML."""
    return batch


@trace_timestep("data_loading", use_gpu=False)
def load_batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@trace_timestep("forward", use_gpu=True)
def forward_pass(model, batch, dtype):
    with torch.cuda.amp.autocast(enabled=False, dtype=dtype):
        return model(**batch)


@trace_timestep("backward", use_gpu=True)
def backward_pass(loss, scaler):
    scaler.scale(loss).backward()


@trace_timestep("optimizer_step", use_gpu=True)
def optimizer_step(scaler, optimizer, scheduler):
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()


@trace_timestep("validation", use_gpu=True)
def run_validation(model, val_loader, dtype, device):
    model.eval()
    val_loss = 0.0
    val_acc  = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = load_batch_to_device(batch, device)
            with torch.cuda.amp.autocast(
                enabled=torch.cuda.is_available(), dtype=dtype
            ):
                out = model(**batch)
                loss = out.loss
                logits = out.logits
            val_loss += loss.item()
            val_acc  += accuracy_from_logits(logits, batch["labels"])
            n_batches += 1

    model.train()
    return val_loss / max(1, n_batches), val_acc / max(1, n_batches)


# MAIN TRAINING LOOP

def main():
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer, train_loader, val_loader = prepare_data()

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=4
    ).to(device)

    # Attach TraceML model-level hooks:
    trace_model_instance(model)

    optimizer = AdamW(model.parameters(), lr=LR)

    total_steps = EPOCHS * math.ceil(len(train_loader))
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler(device="cuda", enabled=torch.cuda.is_available())

    model.train()
    global_step = 0

    #  TRAINING LOOP
    for epoch in range(EPOCHS):
        running_loss = 0.0
        running_acc  = 0.0

        for step, batch in enumerate(train_loader):
            # Wrap batch fetch for TraceML timing
            batch = fetch_batch(batch)

            batch = load_batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)

            out = forward_pass(model, batch, dtype)
            loss = out.loss
            logits = out.logits

            backward_pass(loss, scaler)
            optimizer_step(scaler, optimizer, scheduler)

            acc = accuracy_from_logits(logits.detach(), batch["labels"])
            running_loss += loss.item()
            running_acc  += acc
            global_step  += 1

            if global_step % 50 == 0:
                print(
                    f"[Train] epoch {epoch+1} step {global_step} "
                    f"| loss {running_loss/50:.4f} | acc {running_acc/50:.4f}"
                )
                running_loss = 0.0
                running_acc  = 0.0

        # ---- VALIDATION ----
        val_loss, val_acc = run_validation(model, val_loader, dtype, device)
        print(f"[Val] epoch {epoch+1} | loss={val_loss:.4f} | acc={val_acc:.4f}")

    # Save model
    save_dir = "./distilbert_agnews_full"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved model to {save_dir}")


if __name__ == "__main__":
    main()
