"""Large custom DL model example — requires 48 GB+ VRAM.

A deep CNN image classifier with a large backbone, designed to stress-test
suggest-gpu with a non-LLM, non-HuggingFace custom model.

Run traceml suggest-gpu to see the VRAM estimate:

    traceml suggest-gpu tests/large_dl_example.py --target-batch-size 64

Expected output: 48–80 GB total VRAM → recommends L40S / A100.

Why so much VRAM?
  - This model has ~50M+ parameters in fp32 across many Conv + Linear layers
    defined with explicit literal dimensions (AST-scannable).
  - With batch_size=64 the activation memory dominates.
  - Full AdamW training (no weight-sharing, no frozen layers).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from traceml.decorators import trace_model_instance, trace_step

# =============================================================================
# Config — all dimensions are literal integers so AST scanner resolves them
# =============================================================================
IMAGE_SIZE = 224
IN_CHANNELS = 3
BATCH_SIZE = 64
NUM_CLASSES = 10000
NUM_SAMPLES = 512
EPOCHS = 1
LR = 1e-3
WEIGHT_DECAY = 1e-4


# =============================================================================
# Model — stages defined with explicit literal dims (no helper method)
# All Conv2d / Linear args are directly resolvable from module constants.
# =============================================================================
class LargeCNNClassifier(nn.Module):
    """Deep CNN backbone, ConvNeXt / EfficientNet scale (~60M parameters)."""

    def __init__(self):
        super().__init__()

        # Stem: 3 → 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        # Stage 1: 64 → 128  (3 conv blocks)
        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        # Stage 2: 128 → 256  (4 conv blocks)
        self.stage2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # Stage 3: 256 → 512  (6 conv blocks)
        self.stage3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )

        # Stage 4: 512 → 1024  (4 conv blocks)
        self.stage4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.GELU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.GELU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.GELU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.GELU(),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Classifier head: 1024 → 4096 → 4096 → NUM_CLASSES
        self.classifier = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 4096),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 10000),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# =============================================================================
# Training loop
# =============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LargeCNNClassifier().to(device)
    trace_model_instance(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(NUM_SAMPLES, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    y = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(EPOCHS):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            with trace_step(model):
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch} | loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
