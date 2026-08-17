# Notebooks

Runnable, Colab-ready notebooks that demonstrate TraceML on real workloads.
Each opens in Google Colab (a free T4 GPU is enough) and runs top to bottom in
a few minutes.

| Notebook | What it shows | Open |
|---|---|---|
| `data_loading_bottleneck.ipynb` | Diagnose and fix a data-loading (input-bound) bottleneck on a real ResNet-18 + Imagenette run: train twice, change only the DataLoader, and read the before/after wall-clock speedup and GPU utilization from TraceML | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/traceopt-ai/traceml/blob/main/notebooks/data_loading_bottleneck.ipynb) |
| `huggingface_dataloading_bottleneck.ipynb` | The Hugging Face Trainer path: diagnose and fix the same real-image input bottleneck with `TraceMLTrainerCallback`, ResNet-50, and Imagenette; compare the two TraceML summaries after changing only `TrainingArguments` data-loader settings | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/traceopt-ai/traceml/blob/main/notebooks/huggingface_dataloading_bottleneck.ipynb) |
| `huggingface_trl_lora_gradient_accumulation.ipynb` | Fine-tune Qwen3-1.7B with TRL and LoRA on a free T4; hold effective batch size constant while changing the physical microbatch, then compare optimizer-step time, phase timing, GPU utilization, and peak memory | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/traceopt-ai/traceml/blob/main/notebooks/huggingface_trl_lora_gradient_accumulation.ipynb) |
