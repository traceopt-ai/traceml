# Aggregator

Out-of-process telemetry server. Hosts a TCP server that accepts rank connections, maintains a rank-aware unified store, and drives the display driver (CLI or NiceGUI). Never shares memory with training — fully isolated so an aggregator crash leaves training intact.

::: traceml.aggregator
