# Notebooks

Runnable, Colab-ready notebooks that demonstrate TraceML on real workloads.
Each opens in Google Colab (a free T4 GPU is enough). Runtime varies by
workload; larger case studies can take 20 minutes or more.

TraceML does not overwrite an existing run directory. To rerun a notebook,
start a fresh Colab runtime or change the run names used in the notebook.

| Notebook | What it shows | Open |
|---|---|---|
| `data_loading_bottleneck.ipynb` | Diagnose and fix a data-loading (input-bound) bottleneck on a real ResNet-18 + Imagenette run: train twice, change only the DataLoader, and read the before/after wall-clock speedup and GPU utilization from TraceML | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/traceopt-ai/traceml/blob/main/notebooks/data_loading_bottleneck.ipynb) |
| `huggingface_dataloading_bottleneck.ipynb` | Launch an ordinary Hugging Face Trainer script with `traceml run`, diagnose a real-image input bottleneck, and compare wall time, Input Wait, and GPU utilization after changing only the `TrainingArguments` data-loader settings | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/traceopt-ai/traceml/blob/main/notebooks/huggingface_dataloading_bottleneck.ipynb) |
| `lightning_dataloading_bottleneck.ipynb` | Launch an ordinary Lightning `Trainer.fit()` script with `traceml run`, diagnose and fix a real-image input bottleneck, compare the runs, and inspect per-step Input Wait to reveal the cold first batch that the run average hides | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/traceopt-ai/traceml/blob/main/notebooks/lightning_dataloading_bottleneck.ipynb) |
| `monai_dataloading_bottleneck.ipynb` | Diagnose and improve a real MONAI training pipeline: train a 3D UNet on the Medical Segmentation Decathlon spleen task, evaluate worker, caching, and threaded-loading changes, then use mixed precision to see how the bottleneck moves | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/traceopt-ai/traceml/blob/main/notebooks/monai_dataloading_bottleneck.ipynb) |
| `huggingface_trl_lora_gradient_accumulation.ipynb` | Fine-tune Qwen3-1.7B with TRL and LoRA on a free T4; hold effective batch size constant while changing the physical microbatch, then compare optimizer-step time, phase timing, GPU utilization, and peak memory | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/traceopt-ai/traceml/blob/main/notebooks/huggingface_trl_lora_gradient_accumulation.ipynb) |
