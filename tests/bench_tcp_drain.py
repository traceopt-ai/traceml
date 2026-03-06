import struct
import time
from typing import Optional


def drain_frames_current(buffer: bytearray, expected: Optional[int]):
    frames = []
    while True:
        if expected is None:
            if len(buffer) < 4:
                break
            expected = struct.unpack("!I", buffer[:4])[0]
            buffer = buffer[4:]

        if len(buffer) < expected:
            break

        frames.append(buffer[:expected])
        buffer = buffer[expected:]
        expected = None

    return frames, buffer, expected


def benchmark_drain():
    # Simulate a massive payload of telemetry frames from worker ranks to the aggregator.
    # We use 100,000 messages of 128 bytes each, which mimics a fast burst of
    # telemetry records over TCP.
    num_frames = 100_000
    frame_size = 128

    print(
        f"Building mock payload with {num_frames} frames of {frame_size} bytes each..."
    )
    huge_payload = bytearray()
    frame_data = b"x" * frame_size
    for _ in range(num_frames):
        huge_payload.extend(struct.pack("!I", frame_size))
        huge_payload.extend(frame_data)

    print(
        f"Total simulated TCP buffer size: {len(huge_payload) / 1024 / 1024:.2f} MB"
    )

    buffer = bytearray(huge_payload)

    print("Benchmarking current O(N^2) implementation...")
    start = time.perf_counter()
    frames, remaining_buffer, expected = drain_frames_current(buffer, None)
    end = time.perf_counter()

    duration = end - start
    print(f"Extraction completed in {duration:.4f} seconds.")
    print(f"Total frames extracted: {len(frames)}")


if __name__ == "__main__":
    benchmark_drain()
