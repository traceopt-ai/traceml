import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from traceml.decorators import trace_model_instance, trace_timestep

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


@trace_timestep("dataloader")
def get_next_batch(it):
    return next(it)


@trace_timestep("forward")
def forward_step(model, x):
    return model(x)


@trace_timestep("backward")
def backward_step(loss):
    loss.backward()


@trace_timestep("optimizer_step")
def optimizer_step(opt):
    opt.step()


# def slow_op(x):
#     # time.sleep(0.0005)  # 0.5ms per image
#     # return x

# -------------------------
# Train
# -------------------------


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
        root="./mnist", train=True, download=False, transform=transform
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    model = MNISTCNN().to(device)

    print("I am here")
    trace_model_instance(model)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    loader_iter = iter(loader)

    print("Starting training...")
    for i in range(len(loader)):
        # for i in range(500):
        xb, yb = get_next_batch(loader_iter)
        xb, yb = xb.to(device), yb.to(device)

        opt.zero_grad()
        out = forward_step(model, xb)
        loss = loss_fn(out, yb)

        backward_step(loss)
        optimizer_step(opt)

        if i % 50 == 0:
            print(f"Step {i}, loss={loss.item():.4f}")

        if i == 500:  # ~2–4 minutes depending on GPU
            break


if __name__ == "__main__":
    main()
