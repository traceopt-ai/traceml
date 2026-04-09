import os
import random

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from traceml.decorators import trace_model_instance, trace_step

SEED = 42
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 96

MAX_TRAIN_EXAMPLES = 40000
MAX_VAL_EXAMPLES = 0

BATCH_SIZE = 128
EPOCHS = 12
LR = 2e-6
WARMUP_RATIO = 0.06


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    preds = torch.argmax(logits, dim=-1)
    return (preds == labels).float().mean()


def prepare_data(rank: int, world_size: int):
    raw = load_dataset("fancyzhx/ag_news")

    train_raw = raw["train"].select(
        range(min(MAX_TRAIN_EXAMPLES, len(raw["train"])))
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    train_ds = train_raw.map(tok, batched=True, remove_columns=["text"])
    train_ds = train_ds.rename_column("label", "labels")
    train_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        pin_memory=True,
    )

    val_loader = None
    if MAX_VAL_EXAMPLES > 0:
        val_raw = raw["test"].select(
            range(min(MAX_VAL_EXAMPLES, len(raw["test"])))
        )
        val_ds = val_raw.map(tok, batched=True, remove_columns=["text"])
        val_ds = val_ds.rename_column("label", "labels")
        val_ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            pin_memory=True,
        )

    return tokenizer, train_loader, val_loader, train_sampler


def main() -> None:
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        amp_dtype = torch.float16
    else:
        device = torch.device("cpu")
        amp_dtype = torch.float32

    set_seed(SEED + rank)

    tokenizer, train_loader, _val_loader, train_sampler = prepare_data(
        rank, world_size
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,
        output_hidden_states=True,
    ).to(device)

    trace_model_instance(model)

    if use_cuda:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    else:
        model = torch.nn.parallel.DistributedDataParallel(model)

    optimizer = AdamW(model.parameters(), lr=LR)

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_RATIO * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler(
        enabled=use_cuda,
        device="cuda" if use_cuda else "cpu",
    )

    retained_debug_cache = []

    model.train()
    global_step = 0

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)

        running_loss = 0.0
        running_acc = 0.0

        for batch in train_loader:
            with trace_step(model.module):
                batch = {
                    key: value.to(device, non_blocking=True)
                    for key, value in batch.items()
                }

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    device_type="cuda" if use_cuda else "cpu",
                    enabled=use_cuda,
                    dtype=amp_dtype,
                ):
                    out = model(**batch)

                loss = out.loss
                logits = out.logits

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # Intentional memory-creep bug for testing.
                retained_debug_cache.append(
                    {
                        "loss": loss,
                        "logits": logits,
                        "hidden": out.hidden_states[-1],
                    }
                )

                acc = accuracy_from_logits(logits.detach(), batch["labels"])

                running_loss += float(loss.detach())
                running_acc += float(acc.detach())
                global_step += 1

                if rank == 0 and global_step % 50 == 0:
                    print(
                        f"[Train] epoch {epoch + 1} step {global_step} "
                        f"| loss {running_loss / 50:.4f} "
                        f"| acc {running_acc / 50:.4f} "
                        f"| retained={len(retained_debug_cache)}"
                    )
                    running_loss = 0.0
                    running_acc = 0.0

    if rank == 0:
        save_dir = "./bert_agnews_ddp_memory_creep"
        os.makedirs(save_dir, exist_ok=True)
        model.module.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Saved model to {save_dir}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
