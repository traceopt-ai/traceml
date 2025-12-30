import torch
import torch.nn as nn
import torch.optim as optim
from traceml.decorators import trace_model, trace_step


# ⚠️ Deep instrumentation (use only for detailed debugging / profiling)
# Attaches per-layer forward/backward memory + timing hooks.
# Can add significant overhead (up to ~20% in long training runs).
# Recommended for short runs or one-off investigations.
@trace_model(
    trace_layer_forward__memory=True,
    trace_layer_backward_memory=True,
    trace_layer_forward_time=True,
    trace_layer_backward_time=True,
)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 3
    batch_size = 64
    input_size = 32
    channels = 3

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dummy_input_shape = (batch_size, channels, input_size, input_size)
    dummy_target_shape = (batch_size,)

    for epoch in range(epochs):
        for i in range(100):
            with trace_step(model):
                inputs = torch.randn(dummy_input_shape, device=device)
                labels = torch.randint(0, 10, dummy_target_shape, device=device)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    with torch.no_grad():
        test_input = torch.randn(1, channels, input_size, input_size, device=device)
        _ = model(test_input)


if __name__ == "__main__":
    main()
