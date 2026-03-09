import time
from typing import Any, Dict

import msgspec


def benchmark_encoding_overhead():
    # Simulate a realistic telemetry payload
    sample_payload = {
        "rank": 0,
        "sampler": "layer_backward_memory",
        "tables": {
            "memory": [
                {
                    "model_id": 12345,
                    "layer_name": f"layer_{i}",
                    "device": "cuda:0",
                    "bytes": i * 1024,
                    "step": 100,
                }
                for i in range(50)  # 50 records in one payload
            ]
        },
    }

    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder()

    # Pre-encode the raw payload as it would arrive over the wire
    raw_payload_bytes = encoder.encode(sample_payload)

    num_iterations = 100_000
    print(
        f"Benchmarking MsgPack pipeline with {num_iterations:,} incoming payloads..."
    )

    # Scenario 1: The Current "Full Decode & Re-encode" Pipeline

    start_time_old = time.perf_counter()
    for _ in range(num_iterations):
        # 1. Receiver fully decodes
        msg_dict = decoder.decode(raw_payload_bytes)

        # 2. SQLite Writer fully re-encodes
        encoded_again = encoder.encode(msg_dict)

    duration_old = time.perf_counter() - start_time_old

    # Scenario 2: The Optimized "Bypass Re-encoding" Pipeline
    start_time_new = time.perf_counter()
    for _ in range(num_iterations):
        # 1. Receiver fully decodes AND keeps the raw bytes
        msg_dict = decoder.decode(raw_payload_bytes)
        kept_raw_bytes = raw_payload_bytes

        # 2. SQLite Writer just uses the raw bytes! (No re-encoding)
        encoded_again = kept_raw_bytes

    duration_new = time.perf_counter() - start_time_new

    print("\n=== Results ===")
    print(f"Current overhead (Decode + Re-encode): {duration_old:.4f} seconds")
    print(f"Optimized approach (Decode only):     {duration_new:.4f} seconds")

    time_saved = duration_old - duration_new
    cpu_percent_saved = (time_saved / duration_old) * 100
    speedup = duration_old / duration_new

    print(
        f"\nTime saved: {time_saved:.4f} seconds ({cpu_percent_saved:.1f}% less CPU time)"
    )
    print(f"Speedup: {speedup:.2f}x faster in the aggregator data path")


if __name__ == "__main__":
    benchmark_encoding_overhead()
