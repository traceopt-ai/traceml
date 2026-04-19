# Decorators

User-facing instrumentation entry points: `trace_step` (step boundary context manager), `trace_model_instance` (hook attachment for layer telemetry), `trace_time` (generic function timer). Fail-open by design — decorator errors log to stderr and do not break user code.

::: traceml.decorators
