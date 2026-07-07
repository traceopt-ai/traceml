# Notebooks

Runnable, Colab-ready notebooks that demonstrate TraceML on real workloads.
Each opens in Google Colab (a free T4 GPU is enough) and runs top to bottom in
a few minutes.

| Notebook | What it shows | Open |
|---|---|---|
| `data_loading_bottleneck.ipynb` | Diagnose and fix a data-loading (input-bound) bottleneck on a real ResNet-18 + Imagenette run: train twice, change only the DataLoader, and read the before/after wall-clock speedup and GPU utilization from TraceML | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/traceopt-ai/traceml/blob/main/notebooks/data_loading_bottleneck.ipynb) |
