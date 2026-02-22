import os
import random

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

# =========================
# TraceML imports
# =========================
from traceml.decorators import trace_model_instance, trace_step, trace_time

# =========================
# Config
# =========================
SEED = 42

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
DATASET_NAME = "tatsu-lab/alpaca"

# Training + demo knobs
MAX_LENGTH = 1024
MAX_TRAIN_EXAMPLES = (
    20000  # raise for longer runs; keep smaller for quick demo
)
BATCH_SIZE = 1  # QLoRA-friendly on T4
GRAD_ACCUM_STEPS = 8
EPOCHS = 1
LR = 2e-4
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.03
MAX_STEPS = 1200  # cap steps for demo

NUM_WORKERS = 2
PIN_MEMORY = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16  # T4 friendly


# =========================
# Utils
# =========================
def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# TraceML: Optional fine-grained user-defined timers
# ============================================================


@trace_time("data_transfer", use_gpu=False)
def load_batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@trace_time("forward", use_gpu=True)
def forward_pass(model, batch, dtype):
    # For LLMs, autocast fp16 is standard on T4
    with torch.cuda.amp.autocast(
        enabled=torch.cuda.is_available(), dtype=dtype
    ):
        return model(**batch)


@trace_time("backward", use_gpu=True)
def backward_pass(loss, scaler):
    scaler.scale(loss).backward()


@trace_time("optimizer_step", use_gpu=True)
def optimizer_step(scaler, optimizer, scheduler):
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


# =========================
# Data
# =========================
def prepare_data():
    """
    Loads Alpaca, formats to a single text field, tokenizes, and returns a DataLoader.
    We intentionally use padding='longest' at batch-time to preserve variability,
    which is useful for observing dataloader/step-time behavior.
    """
    ds = load_dataset(DATASET_NAME, split="train")

    if MAX_TRAIN_EXAMPLES is not None and MAX_TRAIN_EXAMPLES > 0:
        ds = ds.select(range(min(MAX_TRAIN_EXAMPLES, len(ds))))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_alpaca(ex):
        instr = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        out = (ex.get("output") or "").strip()

        text = (
            "### Instruction:\n" + instr + "\n\n"
            "### Input:\n" + (inp if inp else "(none)") + "\n\n"
            "### Response:\n" + out
        )
        return {"text": text}

    ds = ds.map(format_alpaca, remove_columns=ds.column_names)

    def tok(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,  # we pad in collator for dynamic padding
        )
        # causal LM labels = input_ids
        enc["labels"] = enc["input_ids"].copy()
        return enc

    ds_tok = ds.map(tok, batched=True, remove_columns=["text"])

    # Dynamic padding collator (keeps variability and avoids excessive padding)
    def collate_fn(features):
        # Pad to longest sequence in THIS batch
        batch = tokenizer.pad(
            features,
            padding="longest",
            return_tensors="pt",
        )
        # Ensure labels exist and are padded correctly
        # tokenizer.pad already padded labels if present
        return batch

    train_loader = DataLoader(
        ds_tok,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
    )

    return tokenizer, train_loader


# =========================
# Model (QLoRA)
# =========================
def build_model():
    # QLoRA: load base model in 4-bit, train LoRA adapters
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=DTYPE,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",  # important for quantized load; keeps it on GPU if available
    )

    # Prep for k-bit training (layer norms, etc.)
    model = prepare_model_for_kbit_training(model)

    # Attach LoRA adapters (trainable)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)

    # Helpful for training stability
    model.config.use_cache = False

    model.print_trainable_parameters()
    return model


# ============================================================
# MAIN TRAINING LOOP (TraceML format)
# ============================================================
def main():
    set_seed()

    assert (
        torch.cuda.is_available()
    ), "This script is intended for CUDA GPUs (e.g., T4)."
    dtype = DTYPE

    tokenizer, train_loader = prepare_data()
    model = build_model()

    # ========================================================
    # TraceML: Attach model-level instrumentation
    # ========================================================
    trace_model_instance(
        model,
        trace_layer_forward_memory=False,
        trace_layer_forward_time=False,
        # trace_layer_backward_memory=False,
        # trace_execution=False,
    )

    # Optimizer sees only trainable params (LoRA) because base.py weights are frozen
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # Scheduler steps happen on optimizer steps (after grad accumulation)
    # Compute total optimizer steps
    steps_per_epoch = min(MAX_STEPS, len(train_loader))
    total_optimizer_steps = max(
        1, (EPOCHS * steps_per_epoch) // GRAD_ACCUM_STEPS
    )
    warmup_steps = int(WARMUP_RATIO * total_optimizer_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    scaler = torch.amp.GradScaler(device="cuda", enabled=True)

    model.train()
    global_step = 0
    opt_step = 0

    running_loss = 0.0

    for epoch in range(EPOCHS):
        for batch in train_loader:

            # ====================================================
            # TraceML: Step boundary
            # ====================================================
            with trace_step(model):

                # Move batch
                batch = load_batch_to_device(batch, DEVICE)

                # Forward
                out = forward_pass(model, batch, dtype)
                loss = out.loss / GRAD_ACCUM_STEPS

                # Backward
                backward_pass(loss, scaler)

                running_loss += loss.detach().float().item() * GRAD_ACCUM_STEPS
                global_step += 1

                # Optimizer step only every GRAD_ACCUM_STEPS
                if global_step % GRAD_ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer_step(scaler, optimizer, scheduler)
                    opt_step += 1

                    if opt_step % 10 == 0:
                        avg_loss = running_loss / 10.0
                        print(
                            f"[Train] epoch {epoch+1} step {global_step:04d} "
                            f"(opt_step {opt_step:04d}) | loss {avg_loss:.4f} "
                        )
                        running_loss = 0.0

            if global_step >= MAX_STEPS:
                break

        if global_step >= MAX_STEPS:
            break

    # Save adapter + tokenizer (small)
    save_dir = "./tinylamma_qlora_adapter"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved adapter to {save_dir}")


if __name__ == "__main__":
    main()
