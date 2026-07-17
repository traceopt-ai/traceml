"""Shared ResNet-18 / Imagenette training core.

Single source of truth for the workload under test, so every tool
(bare, TraceML, torch.profiler, cProfile) measures the SAME compute.
Derived from the data-loading bottleneck example notebook: ResNet-18,
full-res Imagenette, Adam, AMP off, drop_last. The only knob is `--profile`:
  baseline  = num_workers=0 (the INPUT-BOUND / GPU-starved run)
  optimized = workers matched to cores + pin_memory + persistent_workers

The training step body here is instrumentation-free. Tool-specific wrappers
(TraceML trace_step, torch.profiler) are applied by the runner scripts, never
by this file, so the workload stays identical across configs.
"""

import os
from contextlib import nullcontext

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build(data_dir, profile, batch_size):
    """Build model + loader + optimizer for the requested profile."""
    optimized = profile == "optimized"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The ONLY thing that changes between profiles: data loading.
    num_workers = min(4, os.cpu_count() or 2) if optimized else 0
    transform = T.Compose(
        [
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ]
    )
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=optimized,
        persistent_workers=(optimized and num_workers > 0),
        drop_last=True,
    )
    # AMP off on purpose in both profiles: it would speed up compute and
    # confound a demo whose whole point is isolating the data-loading change.
    model = torchvision.models.resnet18(
        weights=None, num_classes=len(dataset.classes)
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return {
        "model": model,
        "loader": loader,
        "criterion": criterion,
        "optimizer": optimizer,
        "device": device,
        "optimized": optimized,
        "num_workers": num_workers,
    }


def train_loop(c, max_steps, step_ctx=None, on_step=None):
    """Run the training loop.

    step_ctx: zero-arg callable returning a context manager wrapped around
      each step body (TraceML's trace_step goes here; bare uses nullcontext).
    on_step: optional callback(step) after each step (torch.profiler.step()).
    """
    if step_ctx is None:
        step_ctx = nullcontext
    model = c["model"]
    loader = c["loader"]
    criterion = c["criterion"]
    optimizer = c["optimizer"]
    device = c["device"]
    optimized = c["optimized"]

    model.train()
    step = 0
    while step < max_steps:
        for images, labels in loader:
            with step_ctx():
                images = images.to(device, non_blocking=optimized)
                labels = labels.to(device, non_blocking=optimized)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
            step += 1
            if on_step is not None:
                on_step(step)
            if step >= max_steps:
                break
    if c["device"].type == "cuda":
        torch.cuda.synchronize()
    return step
