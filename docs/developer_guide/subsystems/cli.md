# CLI

The user-facing command launcher (`traceml watch|run|deep`). Parses arguments, validates the script path, sets up environment variables for rank processes, and spawns the aggregator + training processes. Handles signals (SIGINT, SIGTERM) to tear down cleanly.

::: traceml.cli
