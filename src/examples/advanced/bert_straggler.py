import os
import random
import time

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)

from traceml.decorators import trace_model_instance, trace_step

SEED = 42
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128

MAX_TRAIN_EXAMPLES = 40000
MAX_VAL_EXAMPLES = 0

BATCH_SIZE = 16
EPOCHS = 8
LR = 2e-6
WARMUP_RATIO = 0.06

# Straggler controls
STRAGGLER_RANK = 0
STRAGGLER_SLEEP_S = 0.8
STRAGGLER_EVERY_N_STEPS = 1
STRAGGLER_PHASE = (
    "backward"  # one of: dataloader, forward, backward, optimizer
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    preds = torch.argmax(logits, dim=-1)
    return (preds == labels).float().mean()


def maybe_inject_straggler(rank: int, global_step: int, phase: str) -> None:
    """
    Inject a deterministic delay on one rank to create a clean DDP straggler.

    This is intentionally synthetic and more stable than relying on slow disk
    or noisy input pipelines.
    """
    if rank != STRAGGLER_RANK:
        return
    if phase != STRAGGLER_PHASE:
        return
    if global_step % STRAGGLER_EVERY_N_STEPS != 0:
        return
    time.sleep(STRAGGLER_SLEEP_S)


def prepare_data(rank: int, world_size: int):
    raw = load_dataset("fancyzhx/ag_news")

    train_raw = raw["train"].select(
        range(min(MAX_TRAIN_EXAMPLES, len(raw["train"])))
    )
    val_raw = raw["test"].select(
        range(min(MAX_VAL_EXAMPLES, len(raw["test"])))
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
    val_ds = val_raw.map(tok, batched=True, remove_columns=["text"])

    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")

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
        collate_fn=default_data_collator,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=default_data_collator,
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

    model.train()
    global_step = 0

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)

        running_loss = 0.0
        running_acc = 0.0

        for batch in train_loader:
            global_step += 1

            with trace_step(model.module):
                maybe_inject_straggler(rank, global_step, phase="dataloader")

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
                    maybe_inject_straggler(rank, global_step, phase="forward")
                    out = model(**batch)

                loss = out.loss
                logits = out.logits

                maybe_inject_straggler(rank, global_step, phase="backward")
                scaler.scale(loss).backward()

                maybe_inject_straggler(rank, global_step, phase="optimizer")
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                acc = accuracy_from_logits(logits.detach(), batch["labels"])

                running_loss += float(loss.detach())
                running_acc += float(acc.detach())

                if rank == 0 and global_step % 50 == 0:
                    print(
                        f"[Train] epoch {epoch + 1} step {global_step} "
                        f"| loss {running_loss / 50:.4f} "
                        f"| acc {running_acc / 50:.4f} "
                        f"| straggler_rank={STRAGGLER_RANK} "
                        f"| phase={STRAGGLER_PHASE} "
                        f"| sleep={STRAGGLER_SLEEP_S:.3f}s"
                    )
                    running_loss = 0.0
                    running_acc = 0.0

    if rank == 0:
        save_dir = "./bert_agnews_ddp_straggler"
        os.makedirs(save_dir, exist_ok=True)
        model.module.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Saved model to {save_dir}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
