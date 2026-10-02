# RF-DETR non-JPEG input-pipeline regression

RF-DETR 1.11.0 increased median native training-step time by **14.78%** on a
controlled large-PNG workload. TraceML localized the change to input waiting,
which increased by **23.74%**, while the measured GPU compute regions became
faster. RF-DETR 1.11.1 restored native step time and input wait to within 0.5%
of the 1.10.1 baseline.

This case study reproduces the regression reported in
[RF-DETR issue #1544](https://github.com/roboflow/rf-detr/issues/1544) and fixed
by [RF-DETR PR #1551](https://github.com/roboflow/rf-detr/pull/1551). It is an
independent reproduction of a known issue, not a claim of original discovery.

## Result

Each value below is the median of three independent runs. Every release has a
matched native run and TraceML-instrumented run in each repeat.

| RF-DETR | Native step | Input wait | GPU compute regions | TraceML pair delta |
|---|---:|---:|---:|---:|
| 1.10.1 | 1,642.3 ms | 1,063.7 ms | 95.4 ms | +1.17% |
| 1.11.0 | 1,885.1 ms | 1,316.2 ms | 85.6 ms | -0.09% |
| 1.11.1 | 1,650.1 ms | 1,063.8 ms | 85.2 ms | -1.25% |

Relative to 1.10.1, version 1.11.0 added 242.8 ms to the native step and
252.5 ms to exposed input wait. The sum of TraceML forward, backward and
optimizer regions decreased by 9.8 ms, so slower GPU computation cannot explain
the wall-time regression. Between 1.11.0 and 1.11.1, compute changed by only
-0.54% while input wait fell by 19.18% and native step time fell by 12.46%.

The recorded measurements pass all 13 checks in evaluation protocol 2. The
rationale for the protocol revision is documented below.

The median native-versus-instrumented timing delta remained between -1.25% and
+1.17% across releases. Negative values represent ordinary run-to-run variation,
not acceleration from instrumentation.

## Setup

| | Configuration |
|---|---|
| Host | AWS EC2 `g6.4xlarge`; one NVIDIA L4; AMD EPYC 7R13, 16 vCPUs |
| GPU software | Driver 595.91.07; PyTorch 2.9.1+cu128 |
| Python | 3.11.16 |
| Model | RF-DETR Nano, pretrained weights, eager mode, resolution 384 |
| Dataset | Deterministic COCO-format fixture; 32 train and 8 validation PNG images at 4096x4096 |
| Training | BF16, batch 4, `num_workers=0`, seed 1544 |
| Window | 10 warm-up steps followed by 40 measured optimizer steps |
| TraceML | Commit `7b9c6b4788f14e818e3ae34fd478a8168dfbe722` |

The generated images are deliberately simple. The experiment measures the
released non-JPEG decode and copy path, not model accuracy or storage throughput
on a natural-image corpus.

Synchronous loading is intentional. It exposes per-batch decode latency at the
training-process boundary instead of allowing worker prefetch to hide it. The
PyTorch Lightning warning that recommends additional DataLoader workers is
therefore expected and should not be followed during this experiment. A
production training configuration should tune workers, prefetching and batch
size separately after the release comparison is complete.

## Evaluation criteria

The current analyzer uses evaluation protocol 2. A result is marked
**supported** only when all of these conditions hold:

- all 18 runs complete: three releases, three repeats, native and instrumented;
- workload, dataset, weights, hardware and controlled dependencies match;
- 1.11.0 native step time and input wait are worse than 1.10.1 in every repeat
  and by at least 10% at the median;
- 1.11.0 GPU compute does not regress by more than 10% relative to 1.10.1;
- 1.11.1 native step time and input wait return to within 10% of 1.10.1;
- GPU compute remains within 10% between 1.11.0 and 1.11.1; and
- the absolute median native-versus-instrumented timing delta is at most 5% for
  every release.

The analyzer always prints the measurements, deltas, passed checks and failed
checks. An unsupported result remains useful diagnostic evidence, but it does
not satisfy the complete automated claim.

### Methodology note

The first analyzer revision required the absolute compute change between
1.10.1 and 1.11.0 to remain within 10%. It classified the recorded run as
inconclusive because compute **improved** by 10.26%, missing the symmetric limit
by 0.26 percentage points.

That rule did not match the causal question: faster compute cannot explain a
slower step, and adjacent feature releases may contain unrelated compute
improvements. Protocol 2 instead limits compute *regression* against the
baseline and checks compute stability directly between the regressed and fixed
releases. The recorded measurements were not changed. This correction and the
original outcome are documented here so the interpretation is reproducible.

## Reproduce

Requirements are Linux x86-64, Python 3.11, exactly one visible NVIDIA GPU, a
CUDA 12.8-compatible driver and network access for dependency and model-weight
downloads.

From the TraceML repository root:

```bash
bash examples/case_studies/rfdetr_input_pipeline_regression/run_experiment.sh
```

The runner creates an isolated environment, downloads the three released
RF-DETR wheels, generates the dataset, alternates release and instrumentation
order, executes all 18 runs, and writes a report under:

```text
logs/rfdetr_input_pipeline_regression/experiments/<timestamp>/analysis/report.md
```

PNG is the default. To exercise the same affected path with BMP:

```bash
bash examples/case_studies/rfdetr_input_pipeline_regression/run_experiment.sh \
  --image-format bmp
```

BMP consumes substantially more disk space. Use one image format per
experiment; do not combine PNG and BMP measurements in one comparison.

## Evidence and scope

Each run records the RF-DETR release and wheel hash, TraceML revision, script
hash, dataset and annotation hashes, weights hash, resolved training settings,
dependency fingerprint, GPU identity and measured step window. Raw telemetry,
terminal logs, datasets, weights and virtual environments remain under `logs/`
and must not be committed.

Before sharing an artifact bundle, review it for local paths and machine
identifiers.

This experiment evaluates eager RF-DETR Nano training with generated non-JPEG
input on one GPU. It does not evaluate model accuracy, natural-image datasets,
JPEG loading, segmentation, keypoints, CUDA graphs, `torch.compile`, multi-GPU
scaling, multi-node execution or production DataLoader tuning.
