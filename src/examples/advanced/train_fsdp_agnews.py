import os
import random

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from traceml.decorators import trace_model_instance, trace_step, trace_time

SEED = 42
MODEL_NAME = "distilbert-base-uncased"

MAX_TRAIN_EXAMPLES = 10000
MAX_VAL_EXAMPLES = 0

BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
WARMUP_RATIO = 0.06


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
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
    val_raw = raw["test"].select(
        range(min(MAX_VAL_EXAMPLES, len(raw["test"])))
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok(examples):
        return tokenizer(examples["text"], truncation=True)

    train_ds = train_raw.map(tok, batched=True, remove_columns=["text"])
    val_ds = val_raw.map(tok, batched=True, remove_columns=["text"])

    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")

    collator = DataCollatorWithPadding(tokenizer)

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
        collate_fn=collator,
        pin_memory=True,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        pin_memory=True,
        num_workers=2,
    )

    return tokenizer, train_loader, val_loader, train_sampler


def load_batch_to_device(batch, device):
    with trace_time("load_batch"):
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def build_fsdp_model(device, use_cuda):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,
    )

    trace_model_instance(model)

    model = model.to(device)

    mp_policy = None
    if use_cuda:
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    # DistilBERT uses TransformerBlock internally; auto-wrap those blocks.
    from transformers.models.distilbert.modeling_distilbert import (
        TransformerBlock,
    )

    auto_wrap = transformer_auto_wrap_policy(
        transformer_layer_cls={TransformerBlock}
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device if use_cuda else None,
        use_orig_params=True,
    )

    return model


def main():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        amp_dtype = torch.float16
    else:
        device = torch.device("cpu")
        amp_dtype = torch.float32

    set_seed(SEED + rank)

    tokenizer, train_loader, _, train_sampler = prepare_data(rank, world_size)

    model = build_fsdp_model(device, use_cuda)

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
    running_loss = 0.0
    running_acc = 0.0

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)

        for batch in train_loader:
            with trace_step(model):
                batch = load_batch_to_device(batch, device)

                optimizer.zero_grad(set_to_none=True)

                with trace_time("forward"):
                    with torch.cuda.amp.autocast(
                        enabled=use_cuda, dtype=amp_dtype
                    ):
                        out = model(**batch)
                        loss = out.loss
                        logits = out.logits

                with trace_time("backward"):
                    scaler.scale(loss).backward()

                with trace_time("optimizer"):
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                acc = accuracy_from_logits(logits.detach(), batch["labels"])
                running_loss += loss.detach()
                running_acc += acc.detach()
                global_step += 1

                if rank == 0 and global_step % 50 == 0:
                    print(
                        f"[FSDP] epoch {epoch+1} step {global_step} "
                        f"| loss {(running_loss / 50).item():.4f} "
                        f"| acc {(running_acc / 50).item():.4f}"
                    )
                    running_loss.zero_()
                    running_acc.zero_()

    # Simple save on rank 0 only; for a comparison post you can also skip saving.
    if rank == 0:
        print("FSDP run finished")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
