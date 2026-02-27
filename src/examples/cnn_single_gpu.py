import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from traceml.decorators import trace_model_instance, trace_step

# -------------------------
# Medium CNN for MNIST
# -------------------------


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# TraceML wrappers
# -------------------------

def forward_step(model, x):
    return model(x)


def backward_step(loss):
    loss.backward()


def optimizer_step(opt):
    opt.step()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            # slow_op,
        ]
    )

    dataset = datasets.MNIST(
        root="./mnist", train=True, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    model = MNISTCNN().to(device)

    # ⚠️ Deep instrumentation (use only for detailed debugging / profiling)
    # Enables per-layer memory + timing hooks.
    # Can add overhead in long training runs.
    # Recommended for short runs, diagnosis, or one-off investigations.
    trace_model_instance(
        model,
        trace_layer_forward_memory=True,
        trace_layer_backward_memory=True,
        trace_layer_forward_time=True,
        trace_layer_backward_time=True,
    )

    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print("Starting training...")
    for step, (xb, yb) in enumerate(loader):

        with trace_step(model):
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad(set_to_none=True)

            out = forward_step(model, xb)
            loss = loss_fn(out, yb)

            backward_step(loss)
            optimizer_step(opt)

        if step % 50 == 0:
            print(f"Step {step}, loss={loss.item():.4f}")

        if step == 500:
            break


if __name__ == "__main__":
    main()
