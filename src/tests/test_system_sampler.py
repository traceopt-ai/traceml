import time
import sys
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

from traceml.samplers.process_sampler import ProcessSampler
from traceml.samplers.system_sampler import SystemSampler

from traceml.manager.tracker_manager import TrackerManager

from traceml.loggers.stdout.system_logger import SystemStdoutLogger
from traceml.loggers.stdout.process_logger import ProcessStdoutLogger
from traceml.loggers.stdout.display_manager import StdoutDisplayManager


def test_system_sampler_with_heavy_task():
    """
    Tests SystemSampler (CPU and RAM) with a CPU-intensive linear regression workload.
    Validates that logs are created and contain valid system metrics.
    """
    # Initialize samplers and loggers
    system_sampler = SystemSampler()
    process_sampler = ProcessSampler()

    system_stdout_logger = SystemStdoutLogger()
    process_stdout_logger = ProcessStdoutLogger()

    # Setup TrackerManager components: (sampler, [list of loggers])
    tracker_components = [
        (system_sampler, [system_stdout_logger]),
        (process_sampler, [process_stdout_logger]),
    ]
    tracker = TrackerManager(components=tracker_components, interval_sec=0.5)

    try:
        tracker.start()

        # Run some heavy task for a short duration to produce CPU and some RAM load
        test_duration = 10
        end_time = time.time() + test_duration
        iteration = 0
        while time.time() < end_time:
            # Generates a large arrays, and increases RAM allocation/deallocation
            X, y = make_regression(n_samples=int(2e5), n_features=50, noise=0.1)
            model = LinearRegression()
            model.fit(X, y)
            iteration += 1
            # print(f"[TraceML Test] Iteration {iteration} completed (remaining: {round(end_time - time.time(), 1)}s)")
            # Add a small sleep to allow other threads (like sampler) to run without extreme contention
            time.sleep(0.01)

        print(
            f"\n[TraceML Test] Heavy task finished after {iteration} iterations.",
            file=sys.stderr,
        )

    except Exception as e:
        print(f"[TraceML Test] Error during test execution: {e}", file=sys.stderr)
        raise

    finally:
        tracker.stop()
        tracker.log_summaries()
        StdoutDisplayManager.stop_display()

    print(
        "\n[TraceML Test] SystemSampler (CPU & RAM) test passed successfully.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    # You might need to adjust PYTHONPATH if your traceml modules are not directly
    # importable from where you run this script. For example, if this test file
    # is in 'tests/' and 'traceml/' is in the project root, you'd run:
    # python -m pytest tests/test_system_sampler.py
    # or just:
    test_system_sampler_with_heavy_task()
