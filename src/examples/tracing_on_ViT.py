import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from transformers import ViTForImageClassification

from traceml.decorators import (
    trace_model_instance,
    trace_time,
    trace_step,
)

# -------------------------
# TraceML wrappers
# -------------------------

@trace_time("forward")
def forward_step(model, x):
    return model(x).logits


@trace_time("backward")
def backward_step(loss):
    loss.backward()


@trace_time("optimizer_step")
def optimizer_step(opt):
    opt.step()


# -------------------------
# Train
# -------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])

    dataset = datasets.CIFAR100(
        root="./cifar100",
        train=True,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,      # ← great to show DL bottleneck
        pin_memory=True,
    )

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=100,
    ).to(device)

    # 🔍 Deep instrumentation (short runs only)
    # trace_model_instance(
    #     model,
    # )

    opt = optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    print("Starting ViT + CIFAR-100 training...")
    for step, (xb, yb) in enumerate(loader):

        with trace_step(model):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)

            logits = forward_step(model, xb)
            loss = loss_fn(logits, yb)

            backward_step(loss)
            optimizer_step(opt)

        if step % 20 == 0:
            print(f"Step {step}, loss={loss.item():.4f}")

        if step == 300:
            break


if __name__ == "__main__":
    main()
