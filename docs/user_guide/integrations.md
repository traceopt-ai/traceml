# Use TraceML With Your Stack

TraceML can be added to a normal PyTorch training loop or used through a
framework-specific entry point. Pick the path that matches how your training
job already runs.

| Stack | TraceML entry point | Start here |
|---|---|---|
| Plain PyTorch loop | `traceml.trace_step(...)` | [Quickstart](quickstart.md) |
| Hugging Face Trainer | `TraceMLTrainer` | [Hugging Face](integrations/huggingface.md) |
| PyTorch Lightning | `TraceMLCallback` | [PyTorch Lightning](integrations/lightning.md) |
| Ray Train | `TraceMLTorchTrainer` | [Ray Train](integrations/ray.md) |
| W&B / MLflow | `traceml.summary()` | [W&B / MLflow](integrations/wandb-mlflow.md) |
