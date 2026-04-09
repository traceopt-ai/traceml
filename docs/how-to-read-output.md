# How to Read TraceML Output

TraceML is built to answer one question quickly:

**Why is this training job slow or unstable?**

This guide explains how to read the output shown in:

- the CLI
- the local UI

The concepts are the same in both.

---

## Start with the diagnosis

TraceML output has two layers:

1. **Diagnosis**
   - the short answer
   - example: `INPUT-BOUND`, `COMPUTE STRAGGLER`, `MEMORY CREEP`

2. **Evidence**
   - the numbers and trends that support the diagnosis
   - example: step breakdown, skew, wait share, memory peaks

The diagnosis is the best place to start.

The tables and charts are there to explain **why** that diagnosis was chosen.

---

## What the CLI and local UI show

### CLI

During a run, the CLI shows:

- system metrics
- process metrics
- step-time diagnosis and summary
- step-memory diagnosis and summary

In `deep` mode, it can also show:

- layer timing
- layer memory

### Local UI

The local UI shows the same ideas in a more compact review format:

- system card
- process card
- step-time analysis
- step-memory analysis
- model diagnostics rail

The CLI is best for live diagnosis while the job is running.

The local UI is best for:

- richer review
- local comparison
- browser-based inspection

---

## Step-time diagnoses

The step-time diagnosis explains where training time is going.

It is based on:

- dataloader time
- forward time
- backward time
- optimizer time
- step time
- wait / overhead
- worst-rank vs median-rank differences in distributed runs

### `BALANCED`

Meaning:

- no single bottleneck is clearly dominating the current window

This usually means:

- no strong input bottleneck
- no strong compute bottleneck
- no clear straggler
- no large wait-heavy pattern

What to do next:

- only optimize further if overall throughput is still too low
- compare runs if you expected better performance

---

### `INPUT-BOUND`

Meaning:

- dataloader or input work is taking a large share of the typical step

Common causes:

- too few dataloader workers
- slow preprocessing
- slow storage
- slow host-to-device copies

What to look at:

- `Dataloader Fetch`
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
- use deeper profiling only after TraceML shows the hot path

---

### `INPUT STRAGGLER`

Meaning:

- one rank has meaningfully more input burden than a typical rank

TraceML uses this idea:

- compare the worst rank to the median rank
- measure how much extra dataloader work the worst rank is carrying
- normalize that by a typical local step burden

In simpler words:

- one rank is slower in the input path, enough to matter to the overall run

Common causes:

- uneven data loading
- rank-local preprocessing jitter
- slow input pipeline on one rank
- storage or host-side imbalance

What to look at:

- `Dataloader Fetch`
- worst rank
- skew (%)
- diagnosis note

What to do next:

- inspect input loading on the worst rank
- compare batch preparation across ranks
- check for host-side interference or noisy neighbors

---

### `COMPUTE STRAGGLER`

Meaning:

- one rank has meaningfully more compute burden than a typical rank

TraceML uses this idea:

- compare worst compute vs median compute
- normalize the excess by a typical local step burden

In simpler words:

- one rank is doing more model-side work than the others

Common causes:

- uneven shapes or data
- rank-local branching or extra work
- compute imbalance in forward, backward, or optimizer

What to look at:

- `Forward`
- `Backward`
- `Optimizer Step`
- worst rank
- skew (%)
- diagnosis note

What to do next:

- inspect the called-out compute phase on the worst rank
- compare input shapes and rank-local logic
- inspect imbalance in backward or optimizer time

---

### `STRAGGLER`

Meaning:

- both input and compute are materially uneven in the same window

In the current policy, this is used when:

- input straggler score is high
- compute straggler score is high

This is a mixed unevenness case.

Common causes:

- one bad rank with multiple problems
- one phase uneven in input and another uneven in compute
- more than one imbalance pattern at the same time

What to do next:

- inspect both dataloader and compute signals
- inspect the worst rank and the largest uneven phases
- reduce complexity by isolating one issue at a time

---

### `WAIT-HEAVY`

Meaning:

- a meaningful part of the typical step is going into wait / overhead instead of useful model work

In TraceML:

- `WAIT* = step_time - (forward + backward + optimizer_step)`

This is a proxy, not a direct collective-wait measurement.

Common causes:

- synchronization delays
- uneven progress across ranks
- CPU stalls
- host-side delays
- transfer / orchestration overhead

What to look at:

- `WAIT Share (%)`
- `Wait*`
- whether the run is also showing straggler behavior

What to do next:

- inspect sync points
- inspect rank imbalance
- inspect CPU-side delays and transfer paths

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

- `Dataloader Fetch`
- `Forward`
- `Backward`
- `Optimizer Step`
- `Step Time`
- `Wait*`

Important rows:

### `Median`

- the typical rank in the current window

### `Worst`

- the slowest or heaviest rank in the current window

### `Worst Rank`

- which rank produced the worst value

### `Skew (%)`

- how much larger the worst value is than the median

### `WAIT Share (%)`

- how much of the typical step is going into wait / overhead

A good reading pattern is:

1. read the diagnosis
2. look at the median row
3. compare worst vs median
4. inspect `Worst Rank`
5. inspect `Skew (%)`
6. inspect `WAIT Share (%)`

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

### `MEMORY CREEP (EARLY)`

Meaning:

- memory is trending upward across the visible window
- this is an early warning, not a final conclusion

In the current policy, this is based on:

- multiple head-vs-tail slice comparisons
- both worst and median memory rising
- a meaningful absolute increase

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

This is a stronger signal than `MEMORY CREEP (EARLY)`.

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

The system panel is context, not the main bottleneck diagnosis.

It helps answer:

- is the machine saturated?
- is CPU high?
- is RAM high?
- are GPUs hot or close to full memory?
- is GPU utilization uneven?

Common fields:

- CPU
- RAM
- GPU utilization
- GPU memory
- GPU temperature
- GPU headroom

Use this panel to understand machine-level pressure around the training run.

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

### Model Diagnostics rail

This is the best place to start in the local UI.

It gives:

- overall severity
- compact step-time diagnosis
- compact step-memory diagnosis
- short evidence strings

### Step Time Analysis

This card shows:

- median vs worst step breakdown
- gap
- worst rank
- wait share
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
| `INPUT-BOUND` | inspect dataloader workers, preprocessing, storage |
| `COMPUTE-BOUND` | inspect forward/backward/optimizer cost |
| `INPUT STRAGGLER` | inspect input path on the worst rank |
| `COMPUTE STRAGGLER` | inspect compute path on the worst rank |
| `STRAGGLER` | inspect both input and compute unevenness |
| `WAIT-HEAVY` | inspect sync points and uneven progress |
| `MEMORY CREEP (EARLY)` | inspect retained state and watch the next window |
| `MEMORY CREEP` | inspect retained tensors and growing caches |
| `HIGH PRESSURE` | reduce memory load |
| `IMBALANCE` | inspect per-rank memory workload |

---

## Common pitfalls

### High `WAIT*` skew alone does not automatically mean a real wait bottleneck

Look at:

- `WAIT Share (%)`
- the diagnosis
- the rest of the step breakdown

A tiny wait share with large percentage skew can still be minor in practice.

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

Always read:

- diagnosis first
- evidence second

---

## A simple reading workflow

If you are in a hurry:

1. read the diagnosis
2. identify the worst rank if shown
3. compare worst vs median
4. look at wait share or memory trend
5. take the suggested next action

That is usually enough to decide where to investigate next.
