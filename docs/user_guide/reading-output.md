# How to Read TraceML Output

TraceML is built to answer one question quickly:

**Why is this training job slow or unstable?**

This guide explains how to read the output shown in:

- the default end-of-run summary
- the live CLI view
- the local UI

The concepts are the same in both.

---

## Start with the diagnosis

TraceML output has two layers:

1. **Primary Diagnosis**
   - the first answer in the end-of-run summary
   - focused on why training was slow
   - example: `INPUT-BOUND`, `COMPUTE STRAGGLER`, `RESIDUAL-HEAVY`

2. **Section Diagnoses**
   - detailed findings for System, Process, Step Time, and Step Memory
   - includes performance findings and health/resource findings
   - example: `HIGH GPU TEMP`, `MEMORY CREEP`, `HIGH PROCESS CPU`

3. **Evidence**
   - the numbers and trends that support the diagnosis
   - example: step breakdown, skew, residual time, memory peaks

The top-level primary diagnosis is the best place to start when asking why a
run was slow. Section diagnoses explain the details and keep health warnings
visible.

The tables and charts are there to explain **why** that diagnosis was chosen.

The primary diagnosis is intentionally performance-focused. A high GPU
temperature, memory creep, or high RSS finding can be important, but it stays
in its section unless TraceML has step-time evidence that it explains slow
training. GPU utilization is treated as supporting context or as an
unexplained-utilization fallback, not as root-cause proof by itself.

---

## What the summary, CLI, and local UI show

### End-of-run summary

By default, `traceml run train.py` prints a compact final summary and writes
`final_summary.json` plus `final_summary.txt`.

The text summary is intentionally verdict-first:

- `TraceML Verdict`: the promoted performance diagnosis and severity
- `Why`: the short evidence-backed reason
- `Next`: the first action to try or inspect
- `Section Status`: compact health/status across System, Process, Step Time,
  and Step Memory
- `System Evidence` and `Step Time Evidence`: the core numbers behind the
  verdict

Detailed section prose remains in the `system.card`, `process.card`,
`step_time.card`, and `step_memory.card` fields inside `final_summary.json`.

### Shareable HTML report

Add `--html-report` to `traceml run` (or `traceml watch`) to also write
`final_summary.html` next to the JSON/TXT. It is a single self-contained file
(inline styling and charts, no JavaScript, no network requests) that opens in
any browser and is easy to drop into Slack, an email, or an issue. It shows a
run header, a top-level verdict from `primary_diagnosis` in schema 1.5 reports,
and per-domain diagnosis cards, metric tables, and bars over the same data as
the JSON. Older saved reports without `primary_diagnosis` fall back to the
strongest section diagnosis for the top banner.

You can also render it from a saved run after the fact:

```bash
traceml view logs/<run_name>/final_summary.json --html        # -> <...>.html
traceml view logs/<run_name>/final_summary.json --html out.html
```

The HTML report is optional and additive: the JSON and TXT artifacts are
unchanged whether or not you pass `--html-report`.

### Live CLI

When launched with `--mode=cli`, the terminal shows live:

- system metrics
- process metrics
- step-time diagnosis and summary
- step-memory diagnosis and summary

Live CLI mode is intended for single-node runs, including single-node
multi-GPU.

### Local UI

The local UI shows the same ideas in a more compact review format:

- system card
- process card
- step-time analysis
- step-memory analysis
- diagnostics rail

The local UI is also intended for single-node runs. Multi-node runs should use
the default final summary path.

The CLI is best for live diagnosis while the job is running.

The local UI is best for:

- richer review
- local comparison
- browser-based inspection

---

## Step-time diagnoses

The step-time diagnosis explains where training time is going.

It is based on:

- input wait
- H2D transfer time
- forward time
- backward time
- optimizer time
- step time
- residual / overhead
- culprit/victim visible rank skew in distributed runs

### `BALANCED`

Meaning:

- no single bottleneck is clearly dominating the current window

This usually means:

- no strong input bottleneck
- no strong compute bottleneck
- no clear straggler
- no large residual-heavy pattern

What to do next:

- only optimize further if overall throughput is still too low
- compare runs if you expected better performance

---

### `INPUT-BOUND`

Meaning:

- input wait is taking a large share of iteration time
- TraceML uses the median per-rank input share, so one unusually slow rank
  does not hide a broad input bottleneck

Common causes:

- too few dataloader workers
- slow preprocessing
- slow storage
- slow host-to-device copies

What to look at:

- `Input Wait`
- its share of the step
- whether the issue is broad or rank-specific

What to do next:

- increase dataloader workers
- reduce preprocessing cost
- improve storage throughput
- inspect batch construction

---

### `COMPUTE-BOUND`

Meaning:

- model compute dominates the typical step
- this is informational when no material input or residual overhead is visible

In practice this means most step time is going into:

- forward
- backward
- optimizer

Common causes:

- large model compute cost
- large batch or sequence length
- expensive backward pass
- expensive optimizer step

What to look at:

- `Forward`
- `Backward`
- `Optimizer Step`
- which compute phase is largest

What to do next:

- optimize model compute
- check batch size / precision / kernels
- use an operator-level profiler only after TraceML shows the hot path

---

### `INPUT STRAGGLER`

Meaning:

- one rank has meaningfully more input burden than a typical rank

TraceML uses this idea:

- detect visible wait cost from backward in DDP/default, or forward + backward
  in FSDP
- identify the likely culprit as the rank that waited least in the visible
  phase
- blame input wait when the culprit has material input-wait excess compared
  with the victim rank

In simpler words:

- one rank is slower in the input path, enough to matter to the overall run

Common causes:

- uneven data loading
- rank-local preprocessing jitter
- slow input pipeline on one rank
- storage or host-side imbalance

What to look at:

- `Input Wait`
- culprit rank
- victim/reference rank
- skew (%)
- diagnosis evidence

What to do next:

- inspect input loading on the culprit rank
- compare batch preparation across ranks
- check for host-side interference or noisy neighbors

---

### `COMPUTE STRAGGLER`

Meaning:

- in DDP/default strategy, the likely culprit rank has materially more forward
  time than the victim rank

FSDP does not emit `COMPUTE STRAGGLER` from the rank-skew rule for now because
forward and backward can include sharding communication.

TraceML uses this idea:

- detect visible wait cost from backward in DDP/default strategy
- compare the culprit rank's forward time with the victim rank
- blame compute when the forward excess is material

In simpler words:

- one rank is spending more time in forward work than the victim rank

Common causes:

- uneven shapes or data
- rank-local branching or extra work
- compute imbalance in forward, backward, or optimizer

What to look at:

- `Forward`
- culprit rank
- victim/reference rank
- skew (%)
- diagnosis note

What to do next:

- inspect the called-out forward phase on the culprit rank
- compare input shapes and rank-local logic

---

### `H2D STRAGGLER`

Meaning:

- one rank spends meaningfully more time in host-to-device transfer than a
  typical rank

TraceML reports this when H2D is the largest material excess on the culprit
rank.

Common causes:

- uneven CPU tensor sizes
- rank-local transfer path differences
- pinned-memory or device-transfer jitter on one rank

What to do next:

- inspect batch shapes and transfer placement on the culprit rank
- compare CPU-to-GPU copy timing across ranks

---

### `STRAGGLER`

Meaning:

- visible rank skew exists, but input wait, H2D, and DDP forward do not explain
  the likely culprit

In the current policy, this is used when:

- the rank difference is large enough to matter
- the culprit's input wait, H2D, and DDP forward excesses are not material

This is sync-bound or unattributed rank skew.

Common causes:

- one bad rank with multiple problems
- one phase uneven in input and another uneven in compute, H2D, or residual
- more than one imbalance pattern at the same time

What to do next:

- inspect input wait, H2D, compute, and residual signals
- inspect sync, collective, and unattributed work around the culprit rank
- reduce complexity by isolating one issue at a time

---

### `RESIDUAL-HEAVY`

Meaning:

- a meaningful part of the traced step is not attributed to H2D, forward,
  backward, or optimizer work

In TraceML:

- `compute = forward + backward + optimizer`
- `residual = step_time - h2d - compute`
- `total_step = input_wait + step_time`

TraceML evaluates residual as the median per-rank share of selected-clock
iteration time (`input_wait + step_time`). Input and residual findings warn at
10% and are critical at 20%; rank skew is supporting evidence rather than a
gate that hides a typical bottleneck.

This is residual unattributed time inside the traced step, not direct
collective, NCCL, or all-reduce timing.

Common causes:

- validation or evaluation inside the measured loop
- checkpointing or logging work
- framework orchestration outside the traced phases
- CPU stalls
- unobserved transfer or orchestration overhead

What to look at:

- `Residual`
- whether the run is also showing straggler behavior

What to do next:

- inspect work happening around the traced training step
- inspect rank imbalance
- inspect CPU-side delays, logging, checkpointing, validation, and unobserved transfer paths

---

### `NO DATA`

Meaning:

- TraceML does not yet have enough complete step data to make a diagnosis

This is common:

- early in the run
- when steps are still being aligned across ranks

What to do next:

- wait for more steps
- make sure the training loop is actually running

---

## How to read the step-time table

In the CLI step summary, the important columns are:

- `IW` / `Input Wait`
- `H2D`
- `Forward`
- `Backward`
- `Optimizer`
- `STEP` / `Traced Step`
- `Residual`

Important rows:

### `Median`

- the typical rank in the current window

### `Worst`

- the slowest or heaviest rank in the current window

### `Worst Rank`

- which rank produced the worst value

### `Skew (%)`

- how much larger the worst value is than the median

### `Residual`

- how much of the traced step is unattributed to H2D, forward, backward, or
  optimizer work

A good reading pattern is:

1. read the diagnosis
2. look at the median row
3. compare worst vs median
4. inspect `Worst Rank`
5. inspect `Skew (%)`
6. inspect `Residual`

---

## Step-memory diagnoses

The step-memory diagnosis explains memory pressure, imbalance, and drift over time.

It is based on:

- memory peaks over the aligned step window
- worst-rank vs median-rank differences
- head-vs-tail growth over the visible window

### `BALANCED`

Meaning:

- no clear memory pressure
- no clear cross-rank imbalance
- no strong memory creep signal

What to do next:

- keep monitoring if throughput is good
- investigate only if you expected lower memory usage

---

### `HIGH PRESSURE`

Meaning:

- memory is close to device capacity

Common causes:

- batch size too large
- activation or optimizer state too large
- fragmented or crowded memory state

What to look at:

- peak allocated / peak reserved
- how close worst peak is to device capacity

What to do next:

- reduce memory load
- lower batch size
- inspect activation / optimizer footprint

---

### `IMBALANCE`

Meaning:

- memory usage is uneven across ranks

Common causes:

- uneven data shapes
- rank-local work differences
- one rank carrying extra state

What to look at:

- `Worst Peak`
- `Worst Rank`
- `Skew (%)`

What to do next:

- inspect per-rank workload
- compare shapes and per-rank behavior

---

### `MEMORY RISING`

Meaning:

- memory is trending upward across the visible window
- this is an early warning, not a final conclusion

In the current policy, this is based on:

- early, middle, and recent memory bands increasing
- both worst and median memory rising
- growth that has not yet crossed the stronger confirmed-creep threshold

Common causes:

- retained tensors
- caches that keep growing
- delayed cleanup
- fragmentation-like growth

What to do next:

- watch the next window
- inspect retained tensors and caches
- look for per-step state that stays alive

---

### `MEMORY CREEP`

Meaning:

- memory growth is stronger and more consistent across the visible window

This is a stronger signal than `MEMORY RISING`.

Common causes:

- persistent retention of tensors
- graph-backed tensors kept alive across steps
- expanding caches
- repeated accumulation of step-local state

Example cause:

- appending tensors like `loss`, `logits`, or hidden states to a list every step without detaching them

What to do next:

- inspect caches and retained references
- detach tensors before storing them
- inspect whether graph-backed tensors are being kept alive

---

### `NO DATA`

Meaning:

- TraceML does not yet have enough aligned memory data to diagnose the run

What to do next:

- wait for more completed steps

---

## How to read the step-memory table

In the CLI memory summary, the important rows are:

### `Median Peak (max/K)`

- the typical rank’s peak memory over the window

### `Worst Peak (max/K)`

- the largest rank peak over the window

### `Worst Rank`

- which rank had the largest peak

### `Skew (%)`

- how much larger the worst peak is than the median peak

### `Head/Tail Delta (worst)` or window delta row

- a compact trend hint showing whether worst memory is moving up or down

Use the diagnosis as the main interpretation.
The delta row is a helpful clue, not the full diagnosis logic.

---

## System metrics

The system panel reports machine-level pressure and GPU-utilization symptoms.
It is still context for the training diagnosis: low or moderate GPU
utilization says the GPU was not fully busy, but it does not prove why.

It helps answer:

- is the machine saturated?
- is CPU high?
- is RAM high?
- are GPUs hot or close to full memory?
- are GPUs idle, partly utilized, or uneven?

Common fields:

- CPU
- RAM
- GPU utilization
- GPU memory
- GPU temperature
- GPU headroom

Use this panel to understand machine-level pressure around the training run.

For average GPU utilization, System diagnosis uses these bands:

- below 30%: `LOW_GPU_UTILIZATION`
- 30% through 70%: `MODERATE_GPU_UTILIZATION`
- above 70%: no GPU-utilization issue; System can stay `NORMAL` if no pressure
  rule fires

Use Step Time to explain the likely cause. For example, a System diagnosis of
`MODERATE_GPU_UTILIZATION` plus a Step Time diagnosis of `INPUT-BOUND` means the
GPU was only partly utilized and the step breakdown points to input loading as
the likely reason.

---

## Process metrics

The process panel shows what the training processes themselves are consuming.

It helps answer:

- how much CPU the worst rank is using
- how much GPU memory the processes are using
- whether process-level GPU memory is imbalanced

Common fields:

- worst-rank CPU
- GPU memory used / reserved / total
- GPU used imbalance

Use this panel when:

- the step diagnosis looks odd
- you want rank-level process context
- you suspect a specific rank is heavier than the others

---

## In the local UI

The local UI shows the same ideas in a more compact form.

### Diagnostics rail

This is the best place to start in the local UI.

It gives:

- overall severity
- compact step-time diagnosis
- compact step-memory diagnosis
- short evidence strings

### Step Time Analysis

This card shows:

- median/reference vs rank breakdown
- visible rank skew
- culprit rank when a rank straggler is diagnosed
- residual time
- dominant split

Use it to validate the step-time diagnosis.

### Step Memory Analysis

This card shows:

- worst vs median memory trend
- compact KPIs
- skew
- worst rank

Use it to validate the memory diagnosis.

### System and Process cards

Use these as context cards:

- system tells you about host and GPU pressure
- process tells you about training-process consumption

---

## Common next actions

| Diagnosis | Good next step |
|---|---|
| `INPUT-BOUND` | inspect input loading, preprocessing, and storage |
| `COMPUTE-BOUND` | inspect forward/backward/optimizer cost |
| `INPUT STRAGGLER` | inspect input path on the culprit rank |
| `COMPUTE STRAGGLER` | inspect DDP forward work on the culprit rank |
| `H2D STRAGGLER` | inspect host-to-device transfer on the culprit rank |
| `STRAGGLER` | inspect sync, collective, or unattributed work around the culprit rank |
| `RESIDUAL-HEAVY` | inspect logging, checkpointing, validation, CPU stalls, and unobserved transfer paths |
| `MEMORY RISING` | inspect retained state and watch the next window |
| `MEMORY CREEP` | inspect retained tensors and growing caches |
| `HIGH PRESSURE` | reduce memory load |
| `IMBALANCE` | inspect per-rank memory workload |

---

## Common pitfalls

### High residual skew alone does not automatically mean a material bottleneck

Look at:

- `Residual`
- the diagnosis
- the rest of the step breakdown

A tiny residual value with large percentage skew can still be minor in practice.

### A high compute share does not mean every compute phase is equally important

Look at:

- which compute phase is largest
- whether forward, backward, or optimizer is dominating

### A memory delta hint is not the whole memory diagnosis

Use:

- the diagnosis label
- the diagnosis note
- worst vs median trend

not just a single raw delta

### System metrics are context, not the final explanation

Low GPU utilization by itself does not prove an input bottleneck.
Moderate GPU utilization is the same kind of symptom: it means the GPU was not
fully busy, not that the GPU was slow.

Always read:

- diagnosis first
- evidence second

---

## A simple reading workflow

If you are in a hurry:

1. read the diagnosis
2. identify the culprit rank if shown
3. compare culprit vs victim/reference rank
4. look at residual time or memory trend
5. take the suggested next action

That is usually enough to decide where to investigate next.
