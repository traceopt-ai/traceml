"""Large model fine-tuning example — designed to require 48 GB+ VRAM.

This script fine-tunes Llama-2-7B in fp16 (full fine-tuning, no LoRA/QLoRA).

Run traceml suggest-gpu to see the VRAM estimate:

    traceml suggest-gpu src/examples/large_model_example.py --target-batch-size 4

Expected output: 80–120 GB total VRAM estimate → recommends A100 80GB multi-GPU.

Why so much VRAM?
  - Weights (fp16):        7B × 2 bytes ≈  14 GB
  - Gradients (fp16):      7B × 2 bytes ≈  14 GB
  - AdamW states (fp32):   7B × 8 bytes ≈  56 GB  ← always fp32!
  - Activations (bs=4):    ~28 GB (heuristic)
  - Total:                 ~112 GB

This is why QLoRA / LoRA adapters exist — they freeze the base weights and only
train a small fraction of parameters, reducing training VRAM dramatically.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM

from traceml.decorators import trace_model_instance, trace_step

# =============================================================================
# Config
# =============================================================================

MODEL_NAME = "meta-llama/Llama-2-7b-hf"
SEQ_LEN = 512
BATCH_SIZE = 4  # target batch size for production training
NUM_DUMMY_SAMPLES = (
    64  # only used for shape; script won't actually run without HF login
)
EPOCHS = 1
LR = 2e-5
WEIGHT_DECAY = 0.01


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Model — full fine-tuning (all 7B parameters are trainable)
    # -------------------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,  # fp16 weights
    ).to(device)

    trace_model_instance(model)

    # All parameters are trainable (no LoRA, no frozen layers)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # -------------------------------------------------------------------------
    # Dummy dataset (shape only — demonstrates memory, not real training)
    # -------------------------------------------------------------------------
    dummy_input_ids = torch.randint(0, 32000, (NUM_DUMMY_SAMPLES, SEQ_LEN))
    dummy_labels = dummy_input_ids.clone()
    dataset = TensorDataset(dummy_input_ids, dummy_labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    model.train()
    for epoch in range(EPOCHS):
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with trace_step(model):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            print(f"Epoch {epoch} | loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
