# Integrations

Drop-in framework adapters. `TraceMLTrainer` subclasses Hugging Face's `Trainer` and wires step-boundary + model hooks via `trace_step` / `trace_model_instance`. `TraceMLCallback` does the equivalent for PyTorch Lightning.

::: traceml.integrations
