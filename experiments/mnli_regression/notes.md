# MNLI experiment notes

Keep setup, run commands, results, and decisions here. Each training attempt
gets a fresh run name; its raw artifacts stay in `logs/<run-name>/`.

## Experiment 1 — BERT/MNLI on one machine with four T4 GPUs

Goal: check whether TraceML can expose a meaningful training difference between
the affected and fixed Transformers revisions described in the README. A setup
check alone is not evidence of a regression or a performance improvement.

### 2026-10-26 — Environment check

Source: terminal output supplied from the AWS machine. Reported machine time
was `17:00:25`; its timezone was not recorded.

| Item | Observed value |
| --- | --- |
| Branch | `dev/abhinav/mnli_regression`, up to date with origin |
| Git commit | Not yet recorded |
| AWS instance type / region | Not yet recorded |
| Topology | One machine, four GPUs |
| GPU | 4 × Tesla T4, 15,360 MiB reported per GPU |
| NVIDIA driver | `595.71.05` |
| CUDA version shown by `nvidia-smi` | `13.2` |
| Python | `3.13.14` |
| PyTorch | `2.12.1+cu130` |
| PyTorch CUDA version | `13.0` |
| GPUs detected by PyTorch | `4` |
| Initial GPU state | No running GPU processes; 0 MiB used on each GPU |
| Untracked files on AWS | `gpu-tests/` |

PyTorch imported and detected all four GPUs. It emitted a `FutureWarning`
about the deprecated `pynvml` package; this check did not fail. No training or
CUDA computation was demonstrated by this output.

The README specifies Python 3.11/3.12. The smoke run below subsequently
completed on this machine's Python 3.13 environment; this does not establish
compatibility for other environments.

### 2026-10-26 — One-GPU smoke run: `mnli-smoke-1`

Source: supplied launch output and final terminal summary. Raw AWS artifacts
have not been copied locally or independently inspected.

```bash
RUN=mnli-smoke-1
traceml run experiments/mnli_regression/train.py \
  --nproc-per-node=1 --mode=summary \
  --logs-dir=logs --run-name="$RUN" --history-retention=4h \
  --args --variant affected --output-dir "logs/$RUN/training" \
  --train-fraction 0.002 --epochs 1
```

AWS artifact directory: `/home/ec2-user/traceml/logs/mnli-smoke-1`.

| Result | Reported value |
| --- | --- |
| Completion | Exit code 0; aggregator stopped; telemetry complete |
| Instrumentation | Hugging Face Trainer callback added automatically |
| Training subset | 785 examples, one epoch |
| TraceML coverage | 1 rank, 50 common steps |
| Mean training loss | 1.114 |
| Matched validation loss / accuracy | 1.092 / 34.83% |
| Mismatched validation loss / accuracy | 1.088 / 36.56% |
| Trainer runtime (includes validation) | 27.31 s |
| TraceML displayed run duration | 49.0 s |
| Mean step / input wait / compute / residual | 84.5 / 0.8 / 69.7 / 13.9 ms |
| Mean per-step peak allocated / reserved memory | 2.0 / 2.4 GiB |

The summary observed all four physical GPUs, but training used only one rank
and one GPU. It reported residual-heavy timing (16%) and high GPU power on
N0/G0. These labels alone do not establish the cause of a training slowdown.
No fixed-revision comparison has run yet.

One logged gradient norm was `nan` at epoch 1, while losses and evaluation
metrics were finite. Numerical stability remains unresolved. FP16 gradient
scaling can skip updates with nonfinite gradients; the output does not show
how many updates were skipped or establish the cause. See the
[PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).
Check the pilot's gradient/loss history before accepting it as performance
evidence. Repeated nonfinite gradients require investigation before scaling up.

Other messages: unauthenticated Hub access, unused pretraining-head weights,
newly initialized classification-head weights, and a distributed API
deprecation warning. None prevented completion.

### 2026-10-26 — Four-GPU affected pilot: `mnli-affected-pilot-1`

Source: supplied AWS terminal output, ending at the saved-logs message. Training
finished, model shards were written, and the final summary was generated. The
pasted output does not include the launcher's final exit-code line.

```bash
python experiments/mnli_regression/train.py --variant affected --prepare-only
RUN=mnli-affected-pilot-1
traceml run experiments/mnli_regression/train.py \
  --nproc-per-node=4 --mode=summary \
  --logs-dir=logs --run-name="$RUN" --history-retention=4h \
  --args --variant affected --output-dir "logs/$RUN/training" \
  --train-fraction 0.1 --epochs 1
```

| Result | Reported value |
| --- | --- |
| Training subset | 39,270 examples, one epoch |
| Coverage | 4/4 ranks, 1/1 node, 614 common steps |
| Trainer runtime / TraceML run duration | 123.2 s / 2m 20s |
| Mean training loss | 3.035 |
| Matched validation loss / accuracy | 2.571 / 73.26% |
| Mismatched validation loss / accuracy | 2.460 / 74.42% |
| Displayed step / input wait / compute | 188.8 / 0.2 / 175.6 ms |
| Displayed backward time | 145.1 ms (77% of step) |
| GPU utilization / temperature / power averages | 97% / 45 C / 55 W |
| System GPU memory per device | 3.4 GiB (23%) |
| Median rank average per-step peak allocated / reserved | 2.4 / 2.8 GiB |

The timing panel identifies its representative rank as R0/N0. Its
`COMPUTE-BOUND` label includes backward timing; DDP gradient communication
occurs during backward, so this does not isolate arithmetic from communication.
Input wait below 1% suggests little exposed input delay in this configuration.
A fixed-revision comparison is still needed; this run alone does not establish
the proposed collator regression.

The separate `HIGH GPU PWR` warning cites 80.5% of the power limit on N0/G0.
That threshold crossing does not establish power throttling. High utilization
means kernels were active much of the sampled time, not that peak arithmetic
throughput was achieved.

The first logged gradient norm was `inf`; all subsequent displayed gradient
norms were finite, and validation accuracy reached 73–74%. This is encouraging
but does not quantify skipped FP16 updates. The DDP unused-parameter warning
also appeared; no settings have been changed in response. Loss values are
recorded as printed, without assuming comparability to the one-GPU smoke run.

AWS artifacts: `/home/ec2-user/traceml/logs/mnli-affected-pilot-1`.
Raw artifacts have not yet been copied locally.

### 2026-10-26 — Four-GPU fixed pilot: `mnli-fixed-pilot-1`

The fixed Transformers revision completed successfully with exit code 0 and
4/4-rank coverage over 614 common steps.

| Measurement | Affected | Fixed | Fixed minus affected |
| --- | ---: | ---: | ---: |
| Step Time | 188.8 ms | 189.4 ms | +0.6 ms (+0.3%) |
| Input Wait | 0.2 ms | 0.2 ms | 0.0 ms |
| Forward | 17.1 ms | 17.4 ms | +0.3 ms |
| Backward | 145.1 ms | 145.5 ms | +0.4 ms |
| Optimizer | 13.4 ms | 13.4 ms | 0.0 ms |
| Trainer runtime | 123.2 s | 123.7 s | +0.5 s |
| Training samples/s | 318.7 | 317.5 | -1.2 |
| Matched accuracy | 73.26% | 73.12% | -0.14 percentage points |
| Mismatched accuracy | 74.42% | 74.41% | -0.01 percentage points |

The fixed run also reported 96% GPU utilization, 3.4 GiB system GPU memory per
device, and a 2.8 GiB median-rank reserved-memory average. Its first gradient
norm was `inf`; subsequent displayed values were finite, matching the affected
run's pattern closely.

Conclusion for this pair: the proposed collator fix produced no material
training improvement in this BERT/MNLI configuration. Input Wait remained
0.2 ms, and the fixed run was 0.3% slower in measured Step Time. This single
pair is not a statistical result, but it provides no reason to spend a longer
run on this collator hypothesis. The large four-GPU backward interval remains
a separate observation to qualify with a matching one-GPU pilot.

AWS artifacts: `/home/ec2-user/traceml/logs/mnli-fixed-pilot-1`.
Raw artifacts and the generated compare report have not yet been copied locally.

### 2026-10-26 — Formal affected-versus-fixed comparison

`traceml compare` returned `EQUIVALENT`: the primary diagnosis remained
`COMPUTE-BOUND`, and primary training signals stayed stable. GPU Step Time was
188.8 ms for the affected run and 189.5 ms for the fixed run, a +0.7 ms
(+0.3%) candidate delta. Compute changed by +0.4%; peak reserved memory was
unchanged at 2.79 GiB; average GPU utilization changed from 96.6% to 96.5%.
The system label changed from `HIGH GPU PWR` to `NORMAL`, while the displayed
utilization and other primary signals remained stable. That label change does
not alter the equivalent performance conclusion.

AWS comparison artifacts:
`/home/ec2-user/traceml/compare/mnli-affected-vs-fixed.{json,txt}`.

### 2026-10-26 — One-GPU fixed control: `mnli-fixed-single-pilot-1`

The matching 10% fixed-revision workload completed successfully with one rank
and 2,455 common steps.

| Measurement | One GPU | Four GPUs | Four-GPU observation |
| --- | ---: | ---: | ---: |
| Step Time | 75.7 ms | 189.5 ms | 2.50× per step |
| Forward | 17.6 ms | 17.4 ms | essentially unchanged |
| Backward | 32.8 ms | 145.5 ms | 4.44× per step |
| Optimizer | 13.3 ms | 13.4 ms | essentially unchanged |
| Input Wait | 0.1 ms | 0.2 ms | negligible in both |
| Training samples/s | 186.8 | 317.5 | 1.70× throughput |
| GPU Step Time coverage | 2,455 steps | 614 steps | expected from global batch |

With the same per-device batch size, four GPUs provided 1.70× the one-GPU
training throughput, or about 42.5% of ideal four-GPU scaling. Forward and
optimizer timing stayed flat while the measured backward interval increased by
112.7 ms. DDP gradient reduction occurs inside backward, so synchronization or
communication is the leading hypothesis. TraceML does not currently separate
local backward compute from DDP communication, so this run does not prove how
the 112.7 ms is divided.

The single-GPU run reported `MEMORY RISING` from an 8.1 MiB (0.4%) change. That
small single-run change is not evidence of a leak. Its system GPU utilization
was 24% because the sampler observed all four physical GPUs while only one was
used for training.

Loss and validation results are not directly comparable as a scaling measure:
the per-device batch stayed at 16, so the four-GPU run used four times the
global batch and performed roughly one quarter as many optimizer updates. The
four-GPU logged loss also appears scaled differently across ranks. Throughput
and phase timing are the relevant observations here.

AWS artifacts:
`/home/ec2-user/traceml/logs/mnli-fixed-single-pilot-1`.

### 2026-10-26 — GPU topology

`nvidia-smi topo -m` reported `PHB` for every GPU pair. All four T4s share CPU
affinity `0-47` and NUMA affinity 0. No direct NVLink relationship was shown.
`PHB` means the path traverses PCIe and a PCIe host bridge, typically the CPU.
This topology is consistent with communication being exposed during DDP
backward, but topology alone does not measure bandwidth or prove the cause of
the observed 112.7 ms increase.

AWS artifact: `/home/ec2-user/traceml/compare/gpu-topology.txt`.

### Next run

1. Preserve the smoke run's AWS artifacts; record instance type and region.
   The AWS checkout commit is still unknown. A local commit must not be
   substituted without confirming the AWS code matches it.
2. Record the machine's GPU interconnect topology, then test the existing DDP
   warning's explicit recommendation: disable unused-parameter discovery for
   BERT, where the run reported that no unused parameters were found.
3. Repeat the four-GPU 10% fixed pilot with only that DDP setting changed.

For each run, append its exact command, run name/artifact path, outcome,
training loss and validation accuracy, timing/coverage findings, and any errors
or environment changes. Record conclusions only after examining the results.

Status: both four-GPU variants, their formal comparison, and the matching
one-GPU control completed. The collator pair is equivalent. Four-GPU scaling
is 1.70× with a much larger backward interval. DDP attribution/optimization,
raw-artifact inspection, and multi-node testing remain undone.
