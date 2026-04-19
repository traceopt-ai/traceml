# Transport

TCP bidirectional telemetry channel plus DDP rank detection. Frames are length-prefixed msgpack blobs. The aggregator runs a threaded `TCPServer` accepting rank connections; each rank's `TCPClient` sends in a background thread and never blocks the training loop.

::: traceml.transport
