import timeit
from collections import deque

# Buffer mimicking Traceml's global state
_layer_forward_time_start_buffer_before = {}
_layer_forward_time_start_buffer_after = {}


class BeforePreHook:
    def __init__(self, model_id: int, layer_name: str):
        self.model_id = model_id
        self.layer_name = layer_name

    def __call__(self):
        # Dictionary lookups on critical path
        model_buf = _layer_forward_time_start_buffer_before.setdefault(
            self.model_id, {}
        )
        layer_q = model_buf.setdefault(self.layer_name, deque())
        layer_q.append(1)


class AfterPreHook:
    def __init__(self, model_id: int, layer_name: str):
        self.model_id = model_id
        self.layer_name = layer_name
        # Resolved once during initialization
        self.layer_q = _layer_forward_time_start_buffer_after.setdefault(
            self.model_id, {}
        ).setdefault(self.layer_name, deque())

    def __call__(self):
        # Direct reference on critical path
        self.layer_q.append(1)


def main():
    print("Benchmarking hook optimizations...")
    # Initialize both with identical states
    before_hook = BeforePreHook(model_id=42, layer_name="encoder.layer.0")
    after_hook = AfterPreHook(model_id=42, layer_name="encoder.layer.0")

    # Warmup
    before_hook()
    after_hook()

    # We use a large number of calls to simulate processing many batches
    # across many layers
    n_calls = 1_000_000

    before_time = timeit.timeit(before_hook, number=n_calls)
    after_time = timeit.timeit(after_hook, number=n_calls)

    print(f"Num calls: {n_calls}")
    print(f"Before (Dictionary lookups):     {before_time:.4f} seconds")
    print(f"After  (Cached deque reference): {after_time:.4f} seconds")
    print("-" * 50)
    print(f"Speedup: {before_time / after_time:.2f}x faster")


if __name__ == "__main__":
    main()
