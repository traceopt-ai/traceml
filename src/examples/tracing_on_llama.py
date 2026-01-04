import os
import time
import torch

from datasets import load_dataset
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# -------------------------
# Config
# -------------------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
DATASET_NAME = "tatsu-lab/alpaca"

MAX_LENGTH = 512
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8

LR = 2e-4
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.03
MAX_STEPS = 1200  # keep short for demo; raise for longer runs

NUM_WORKERS = 2
PIN_MEMORY = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16  # T4-friendly; bf16 usually not supported on T4


# -------------------------
# 1) Load dataset (ready-to-use)
# -------------------------
ds = load_dataset(DATASET_NAME, split="train")

def format_alpaca(example):
    # Alpaca fields: instruction, input, output
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")

    text = (
        "### Instruction:\n" + instruction.strip() + "\n\n"
        "### Input:\n" + (inp.strip() if inp else "(none)") + "\n\n"
        "### Response:\n" + output.strip()
    )
    return {"text": text}

ds = ds.map(format_alpaca, remove_columns=ds.column_names)


# -------------------------
# 2) Tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# Many LLaMA-like tokenizers don't have pad_token by default
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_batch(batch):
    # padding="longest" preserves step-to-step variability (good for observing stalls)
    out = tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="longest",
        return_tensors=None,
    )
    # Standard causal LM objective: labels = input_ids (ignore padding tokens)
    out["labels"] = out["input_ids"].copy()
    return out

ds_tok = ds.map(tokenize_batch, batched=True, remove_columns=["text"])


# -------------------------
# 3) DataLoader
# -------------------------
def collate_fn(features):
    # Convert lists to tensors and pad dynamically to max length in the batch
    # Here batch_size=1, but collate still ensures tensors are present.
    batch = {}
    for k in ["input_ids", "attention_mask", "labels"]:
        batch[k] = torch.tensor([f[k] for f in features], dtype=torch.long)
    return batch

loader = DataLoader(
    ds_tok,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    collate_fn=collate_fn,
)


# -------------------------
# 4) Load model in 4-bit (QLoRA)
# -------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # standard QLoRA choice
    bnb_4bit_use_double_quant=True,     # helps accuracy
    bnb_4bit_compute_dtype=DTYPE,       # fp16 compute on T4
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepares layer norms and gradient checkpointing compatibility for k-bit finetuning
model = prepare_model_for_kbit_training(model)


# -------------------------
# 5) Attach LoRA adapters (the "trainable part")
# -------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # Common targets for LLaMA-like models
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Optional: reduce memory / stabilize
model.config.use_cache = False  # important for training


# -------------------------
# 6) Optimizer + Scheduler
# -------------------------
# Only LoRA params require grad, so optimizer sees a small parameter set
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Rough steps estimate for scheduler: we’ll stop at MAX_STEPS anyway
total_train_steps = MAX_STEPS
warmup_steps = int(WARMUP_RATIO * total_train_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_train_steps,
)


# -------------------------
# 7) Train loop (custom PyTorch)
# -------------------------
model.train()
step = 0
optimizer.zero_grad(set_to_none=True)

for batch in loader:

    # Move to GPU
    batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}

    # Forward + loss
    outputs = model(**batch)
    loss = outputs.loss / GRAD_ACCUM_STEPS
    loss.backward()

    if (step + 1) % GRAD_ACCUM_STEPS == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    if step % 20 == 0:
        # This prints a useful live signal without spamming
        print(
            f"step={step:04d} "
            f"loss={loss.item() * GRAD_ACCUM_STEPS:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

    step += 1
    if step >= MAX_STEPS:
        break

print("Done.")

# -------------------------
# 8) Save LoRA adapter (small file)
# -------------------------
os.makedirs("qlora_adapter", exist_ok=True)
model.save_pretrained("qlora_adapter")
tokenizer.save_pretrained("qlora_adapter")
print("Saved adapter to ./qlora_adapter")
