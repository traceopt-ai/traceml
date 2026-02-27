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

# =========================
# TraceML imports
# =========================
# trace_model_instance:
#   Attaches model-level hooks (activation memory, gradient memory, timings, etc.)
# trace_step:
#   Defines a training-step boundary (flushes TraceML buffers at step end)
# trace_timestep:a
#   Optional fine-grained timers for user-defined code sections
from traceml.decorators import trace_model_instance, trace_step

SEED = 42
MODEL_NAME = "bert-base-uncased"
# MODEL_NAME = "prajjwal1/bert-mini"


# Increase these to generate a LOT of profiling data
MAX_TRAIN_EXAMPLES = 10000
MAX_VAL_EXAMPLES = 0
BATCH_SIZE = 32
EPOCHS = 10
LR = 2e-5
WARMUP_RATIO = 0.06


def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == labels).sum()
    total = labels.size(0)
    return correct / max(1, total)


def prepare_data():
    """Load & tokenize AG News (bigger subset for better logs)."""
    raw = load_dataset("ag_news")

    train_raw = raw["train"].select(
        range(min(MAX_TRAIN_EXAMPLES, len(raw["train"])))
    )
    val_raw = raw["test"].select(
        range(min(MAX_VAL_EXAMPLES, len(raw["test"])))
    )

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


# ============================================================
# TraceML: Optional fine-grained user-defined timers
# ============================================================
# These are NOT required for TraceML to work.
# They add extra visibility into specific code regions.


def load_batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def forward_pass(model, batch, dtype):
    with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
        return model(**batch)


def backward_pass(loss, scaler):
    scaler.scale(loss).backward()


def optimizer_step(scaler, optimizer, scheduler):
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()


def run_validation(model, val_loader, dtype, device):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
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
            val_acc += accuracy_from_logits(logits, batch["labels"])
            n_batches += 1

    model.train()
    return val_loss / max(1, n_batches), val_acc / max(1, n_batches)


# ============================================================
# MAIN TRAINING LOOP
# ============================================================


def main():
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer, train_loader, val_loader = prepare_data()

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=4
    ).to(device)

    # ========================================================
    # TraceML: Attach model-level instrumentation
    # ========================================================
    # This attaches hooks for:
    #  - forward pass memory
    #  - backward pass memory
    #  - execution context
    #  - forward pass / backward timing
    # No changes to training loop required.
    # trace_model_instance(
    #     model,
    #     # sample_layer_memory=False,
    #     # trace_layer_forward__memory=False,
    #     # trace_layer_backward_memory=False,
    #     # trace_execution=False,
    # )

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = EPOCHS * math.ceil(len(train_loader))
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler(device="cuda", enabled=True)

    model.train()
    global_step = 0

    #  TRAINING LOOP
    for epoch in range(EPOCHS):
        running_loss = 0.0
        running_acc = 0.0

        for batch in train_loader:
            # ====================================================
            # TraceML: Step boundary
            # ====================================================
            # Defines ONE training step for TraceML.
            # Guarantees:
            #  - per-step flushing of buffers
            #  - crash-safe observability
            with trace_step(model):

                ## Load batch
                batch = load_batch_to_device(batch, device)

                optimizer.zero_grad(set_to_none=True)

                out = forward_pass(model, batch, dtype)
                loss = out.loss
                logits = out.logits

                backward_pass(loss, scaler)
                optimizer_step(scaler, optimizer, scheduler)

                acc = accuracy_from_logits(logits.detach(), batch["labels"])
                running_loss += loss.detach()
                running_acc += acc.detach()
                global_step += 1

                if global_step % 50 == 0:
                    avg_loss = (running_loss / 50).item()  # ONE sync
                    avg_acc = (running_acc / 50).item()

                    print(
                        f"[Train] epoch {epoch + 1} step {global_step} "
                        f"| loss {avg_loss:.4f} | acc {avg_acc:.4f}"
                    )

                    running_loss.zero_()
                    running_acc.zero_()

        # ---- VALIDATION ----
        # val_loss, val_acc = run_validation(model, val_loader, dtype, device)
        # print(f"[Val] epoch {epoch+1} | loss={val_loss:.4f} | acc={val_acc:.4f}")

    # Save model
    save_dir = "./distilbert_agnews_full"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved model to {save_dir}")


if __name__ == "__main__":
    main()
