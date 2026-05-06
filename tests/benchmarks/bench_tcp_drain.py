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


def drain_frames_optimized(buffer: bytearray, expected: Optional[int]):
    frames = []
    offset = 0
    buf_len = len(buffer)

    while True:
        if expected is None:
            if buf_len - offset < 4:
                break
            expected = struct.unpack("!I", buffer[offset : offset + 4])[0]
            offset += 4

        if buf_len - offset < expected:
            break

        frames.append(buffer[offset : offset + expected])
        offset += expected
        expected = None

    if offset > 0:
        del buffer[:offset]

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

    buffer_copy1 = bytearray(huge_payload)
    print("Benchmarking current O(N^2) implementation...")
    start1 = time.perf_counter()
    frames1, remaining_buffer1, expected1 = drain_frames_current(
        buffer_copy1, None
    )
    end1 = time.perf_counter()

    duration1 = end1 - start1
    print(f"Extraction completed in {duration1:.4f} seconds.")
    print(f"Total frames extracted: {len(frames1)}\n")

    buffer_copy2 = bytearray(huge_payload)
    print("Benchmarking OPTIMIZED O(N) implementation...")
    start2 = time.perf_counter()
    frames2, remaining_buffer2, expected2 = drain_frames_optimized(
        buffer_copy2, None
    )
    end2 = time.perf_counter()

    duration2 = end2 - start2
    print(f"Extraction completed in {duration2:.4f} seconds.")
    print(f"Total frames extracted: {len(frames2)}")
    if duration2 > 0:
        print(f"Speedup: {duration1 / duration2:.2f}x\n")

    assert len(frames1) == len(frames2)
    for f1, f2 in zip(frames1, frames2):
        assert f1 == f2


if __name__ == "__main__":
    benchmark_drain()
