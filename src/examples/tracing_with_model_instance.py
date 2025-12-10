import random
import torch
import torch.nn as nn
import torch.optim as optim
from traceml.decorators import trace_model_instance


# A CNN that tolerates variable input sizes via AdaptiveAvgPool2d
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(32, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    device = torch.device("cpu")

    epochs = 10
    steps_per_epoch = 12

    # Vary batch size and input resolution each step
    batch_choices = [8, 16, 32, 64, 96, 128]
    size_choices = [28, 32, 40, 48, 56, 64, 72, 80]

    model = SimpleCNN().to(device)

    # Attach activation hooks so your sampler sees events
    trace_model_instance(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            bs = random.choice(batch_choices)
            H = W = random.choice(size_choices)

            inputs = torch.randn(bs, 3, H, W, device=device)
            labels = torch.randint(0, 10, (bs,), device=device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 3, 64, 64, device=device)
        _ = model(test_input)


if __name__ == "__main__":
    main()
