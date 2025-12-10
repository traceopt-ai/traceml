import torch
import torch.nn as nn
import torch.optim as optim
from traceml.decorators import trace_model


# Define a simple CNN model and decorate the class
@trace_model()
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output 16x16

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output 8x8

        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # 10 output classes

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)  # Flatten for FC layer
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 3
    batch_size = 64
    input_size = 32
    channels = 3

    # Instantiate model (decorator will trace it automatically)
    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Simulate dummy data loader
    dummy_input_shape = (batch_size, channels, input_size, input_size)
    dummy_target_shape = (batch_size,)

    for epoch in range(epochs):
        for i in range(100):
            inputs = torch.randn(dummy_input_shape).to(device)
            labels = torch.randint(0, 10, dummy_target_shape).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        test_input = torch.randn(1, channels, input_size, input_size).to(device)
        _ = model(test_input)


if __name__ == "__main__":
    main()
