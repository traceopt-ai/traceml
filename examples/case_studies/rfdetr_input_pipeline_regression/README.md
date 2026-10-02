# RF-DETR non-JPEG input-pipeline regression

This case study is a reproducible experiment for a released RF-DETR training
regression. It compares RF-DETR 1.10.1, 1.11.0 and 1.11.1 on the same bounded
training workload and asks whether TraceML attributes any change to input wait
rather than GPU computation.

The experiment is based on the public report in
[RF-DETR issue #1544](https://github.com/roboflow/rf-detr/issues/1544) and the
corresponding [fix in PR #1551](https://github.com/roboflow/rf-detr/pull/1551).
The reported regression affected large PNG and BMP images. JPEG input does not
exercise the affected path.

## Status

A measured result should be added only after the complete
experiment passes the publication checks below on a CUDA host.

## Question

Can a release-to-release TraceML comparison:

1. detect a training-step regression in RF-DETR 1.11.0 relative to 1.10.1;
2. localize the increase to exposed DataLoader input wait while the model's GPU
   phases remain stable; and
3. confirm that RF-DETR 1.11.1 restores the earlier behavior?

This is a reproduction of a known upstream regression. A successful result
supports a claim that TraceML independently detected and localized the released
regression; it would not support a claim that TraceML originally discovered it.

## Experimental controls

The runner holds these variables constant across versions:

- one NVIDIA GPU and one host;
- RF-DETR Nano in eager mode at 384-pixel model resolution;
- a deterministic COCO-format dataset of large non-JPEG source images;
- batch size four, synchronous loading (`num_workers=0`), seed, precision,
  augmentation and step window;
- pretrained model weights and all dependencies other than the RF-DETR wheel;
- 10 warm-up steps followed by 40 measured optimizer steps; and
- matched native and TraceML-instrumented runs for every version.

Three repeats use different version orders. Native and traced order is also
reversed in the second repeat. This does not eliminate host noise, but it makes
simple cache, thermal and execution-order effects visible.

The generated images are deliberately simple. The experiment measures the
released decode and copy path, not model quality or storage throughput on a
natural-image corpus. Synchronous loading makes per-batch decode latency visible
at the training-process boundary instead of allowing worker prefetch to hide it.

## Run the experiment

Requirements are Linux x86-64, Python 3.11, one NVIDIA GPU, a CUDA
12.8-compatible driver and network access for the initial environment and model
weight downloads.

From the TraceML repository root:

```bash
bash examples/case_studies/rfdetr_input_pipeline_regression/run_experiment.sh
```

The runner creates a reusable environment, downloads the three released wheels,
generates the dataset and writes a fresh result directory under:

```text
logs/rfdetr_input_pipeline_regression/experiments/<timestamp>/
```

For a BMP reproduction instead of the default PNG workload:

```bash
bash examples/case_studies/rfdetr_input_pipeline_regression/run_experiment.sh \
  --image-format bmp
```

BMP consumes substantially more disk space. Keep one image format per
experiment; do not combine PNG and BMP measurements in one comparison.

## Publication checks

`analyze.py` rejects incomplete or mismatched runs before producing a report.
The report is marked **supported** only when all of the following hold:

- all 18 runs completed: three versions, three repeats, native and traced;
- dataset, weights, hardware, workload settings and non-RF-DETR dependencies
  match across runs;
- native wall time regresses in 1.11.0 in every repeat and by at least 10% at
  the median;
- traced input wait increases in 1.11.0 in every repeat and by at least 10% at
  the median;
- the combined forward, backward and optimizer time changes by no more than 10%;
- 1.11.1 native wall time and input wait return to within 10% of 1.10.1; and
- median TraceML overhead for each version is no more than 5%.

Thresholds are declared before the measurement and printed in the report. A
failed check means the experiment is inconclusive on that host; it must not be
rewritten as a positive result.

## Evidence retained

Each run records the resolved RF-DETR version and source-tree hash, TraceML
revision, dataset manifest hash, annotation hashes, weights hash, selected
training configuration, installed-package fingerprint, GPU identity and the
exact measured window. Raw TraceML telemetry and terminal logs remain beside
the run records.

Before publishing an artifact bundle, review it for local paths and machine
identifiers. Do not commit datasets, model weights, virtual environments or raw
telemetry.

## Scope

The experiment evaluates eager RF-DETR Nano training on one GPU. It does not
measure model accuracy, JPEG loading, segmentation, keypoints, CUDA graphs,
`torch.compile`, multi-GPU scaling, multi-node execution or every possible
storage and worker configuration.
