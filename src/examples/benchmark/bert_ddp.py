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
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

# =========================
# TraceML imports
# =========================
# trace_model_instance:
#   Attaches model-level hooks (forward pass, backward pass, timings, etc.)
# trace_step:
#   Defines a training-step boundary (flushes TraceML buffers at step end)
# trace_time:
#   Optional fine-grained timers for user-defined code sections
from traceml.decorators import trace_model_instance, trace_step


SEED = 42
MODEL_NAME = "distilbert-base-uncased"
# MODEL_NAME = "prajjwal1/bert-mini"

# Increase these to generate a LOT of profiling data
MAX_TRAIN_EXAMPLES = 10000
MAX_VAL_EXAMPLES = 0

BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-6
WARMUP_RATIO = 0.06


def set_seed(seed: int):
    """
    Ensure deterministic behavior where possible.
    In DDP, each rank should have a slightly different seed.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """
    Simple per-batch accuracy (rank-local).
    """
    preds = torch.argmax(logits, dim=-1)
    return (preds == labels).float().mean()


def prepare_data(rank: int, world_size: int):
    """
    Load dataset and create a DistributedSampler so that:
      - each rank sees a unique shard of the dataset
      - all samples are seen exactly once per epoch
    """
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

    # DistributedSampler shards the dataset across ranks.
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,  # sampler replaces shuffle=True
        collate_fn=collator,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    return tokenizer, train_loader, val_loader, train_sampler


# ============================================================
# TraceML: Optional fine-grained user-defined timers
# ============================================================
# These are NOT required for TraceML to work.
# They add extra visibility into specific code regions.


def load_batch_to_device(batch, device):
    """
    Measure CPU → GPU transfer time.
    """
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def forward_pass(model, batch, dtype):
    """
    Measure forward pass time (with AMP).
    """
    use_cuda = torch.cuda.is_available()
    with torch.cuda.amp.autocast(enabled=use_cuda, dtype=dtype):
        return model(**batch)


def backward_pass(loss, scaler):
    """
    Measure backward pass time.
    """
    scaler.scale(loss).backward()


def optimizer_step(scaler, optimizer, scheduler):
    """
    Measure optimizer + scheduler step.
    """
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()


# ============================================================
# MAIN TRAINING ENTRY POINT
# ============================================================
def main():

    # --------------------------------------------------------
    # DDP ENVIRONMENT (provided automatically by torchrun)
    # --------------------------------------------------------
    # torchrun sets these OS environment variables per process
    rank = int(os.environ.get("RANK", 0))  # global rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # GPU index on this node
    world_size = int(
        os.environ.get("WORLD_SIZE", 1)
    )  # total number of processes

    # --------------------------------------------------------
    # Initialize distributed communication
    # --------------------------------------------------------
    # This creates the distributed system (NCCL backend for GPUs)
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    # --------------------------------------------------------
    # Bind this process to ONE GPU
    # --------------------------------------------------------
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    # Different seed per rank (important for shuffling etc.)
    set_seed(SEED + rank)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    tokenizer, train_loader, val_loader, train_sampler = prepare_data(
        rank, world_size
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=4
    ).to(device)

    # --------------------------------------------------------
    # TraceML: attach hooks to the *real* model
    # --------------------------------------------------------
    # Do this BEFORE wrapping with DistributedDataParallel
    trace_model_instance(model)

    # Wrap model with DDP
    if use_cuda:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    else:
        model = torch.nn.parallel.DistributedDataParallel(model)

    # --------------------------------------------------------
    # Optimizer, scheduler, scaler
    # --------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=LR)

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_RATIO * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler(
        enabled=use_cuda, device="cuda" if use_cuda else "cpu"
    )

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------
    model.train()
    global_step = 0

    for epoch in range(EPOCHS):

        # IMPORTANT:
        # Ensures all ranks shuffle identically before sharding
        train_sampler.set_epoch(epoch)

        running_loss = 0.0
        running_acc = 0.0

        for batch in train_loader:

            # ------------------------------------------------
            # TraceML: define ONE training step boundary
            # ------------------------------------------------
            with trace_step(model.module):

                # Load batch to GPU
                batch = load_batch_to_device(batch, device)

                optimizer.zero_grad(set_to_none=True)

                # Forward
                out = forward_pass(model, batch, dtype)
                loss = out.loss
                logits = out.logits

                # Backward + optimizer
                backward_pass(loss, scaler)

                optimizer_step(scaler, optimizer, scheduler)

                # Rank-local metrics
                acc = accuracy_from_logits(logits.detach(), batch["labels"])

                running_loss += loss.detach()
                running_acc += acc.detach()
                global_step += 1

                # ------------------------------------------------
                # Only rank 0 prints/logs
                # ------------------------------------------------
                if rank == 0 and global_step % 50 == 0:
                    print(
                        f"[Train] epoch {epoch+1} step {global_step} "
                        f"| loss {(running_loss/50).item():.4f} "
                        f"| acc {(running_acc/50).item():.4f}"
                    )
                    running_loss.zero_()
                    running_acc.zero_()

    # --------------------------------------------------------
    # SAVE (rank 0 only)
    # --------------------------------------------------------
    if rank == 0:
        save_dir = "./bert_agnews_ddp"
        os.makedirs(save_dir, exist_ok=True)
        model.module.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Saved model to {save_dir}")

    # --------------------------------------------------------
    # Clean shutdown
    # --------------------------------------------------------
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
