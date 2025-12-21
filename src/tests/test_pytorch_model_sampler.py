import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from traceml.decorators import trace_model_instance

from traceml.samplers.process_sampler import ProcessSampler
from traceml.samplers.system_sampler import SystemSampler
from traceml.samplers.layer_memory_sampler import LayerMemorySampler
from traceml.samplers.activation_memory_sampler import ActivationMemorySampler
from traceml.samplers.gradient_memory_sampler import GradientMemorySampler

from traceml.manager.tracker_manager import TrackerManager

from traceml.renderers.system_process_renderer import SystemProcessRenderer
from traceml.renderers.layer_combined_memory_renderer import (
    LayerCombinedRenderer,
)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=256, output_dim=10):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def test_system_sampler_with_pytorch_model():
    """
    Runs a simple PyTorch model training workload to test LayerMemorySampler.
    """

    # Initialize samplers and loggers
    system_sampler = SystemSampler()
    process_sampler = ProcessSampler()
    layer_memory_sampler = LayerMemorySampler()
    activation_memory_sampler = ActivationMemorySampler()
    gradient_memory_sampler = GradientMemorySampler()

    system_process_stdout_logger = SystemProcessRenderer()
    combined_stdout_logger = LayerCombinedRenderer()

    tracker_components = [
        ([system_sampler, process_sampler], [system_process_stdout_logger]),
        (
            [layer_memory_sampler, activation_memory_sampler, gradient_memory_sampler],
            [combined_stdout_logger],
        ),
    ]
    tracker = TrackerManager(components=tracker_components, interval_sec=0.5)

    try:
        tracker.start()

        # Define model, optimizer, and loss
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleMLP(input_dim=100, hidden_dim=256, output_dim=10).to(device)
        trace_model_instance(model)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Training loop (simulated workload)
        test_duration = 10
        end_time = time.time() + test_duration
        iteration = 0

        while time.time() < end_time:
            batch_size = 512

            # Random synthetic data
            inputs = torch.randn(batch_size, 100).to(device)
            targets = torch.randn(batch_size, 10).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            iteration += 1
            time.sleep(0.01)  # Allow samplers to run without contention

        print(
            f"\n[TraceML Test] Training completed after {iteration} iterations.",
            file=sys.stderr,
        )

    except Exception as e:
        print(f"[TraceML Test] Error during test execution: {e}", file=sys.stderr)
        raise

    finally:
        tracker.stop()
        tracker.log_summaries()

    print(
        "\n[TraceML Test] PyTorch model memory tracking test passed successfully.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    test_system_sampler_with_pytorch_model()
