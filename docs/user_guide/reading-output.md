# How to Read TraceML Output

Start with the diagnosis, check the evidence, then follow the suggested next
step. This guide explains the default final report first; live views and
measurement details follow below.

## Start with the diagnosis

By default, `traceml run train.py` prints a final report and saves
`final_summary.json` and `final_summary.txt`. Here is an excerpt from the
illustrative report in the README:

```text
TraceML Run Summary
bert_finetune · 1 rank · 1 GPU observed · 256 common steps · 52.4s

Verdict: INPUT-BOUND (CRITICAL)
Why: Input Wait took 64% of Step Time.
Next: Increase workers, prefetch, or storage throughput.

STEP TIMING (Window Average), GPU Clock
Step Time           200.4 ms  100%
├─ Input Wait       128.0 ms   64%
├─ Compute           68.0 ms   34%
│  ├─ Forward        24.0 ms   12%
│  ├─ Backward       38.0 ms   19%
│  └─ Optimizer       6.0 ms    3%
├─ H2D                0.4 ms   <1%
└─ Residual           3.6 ms    2%
DataLoader fetch: 120.0 ms (CPU, supplemental)
```

| Report field | How to use it |
| --- | --- |
| **Verdict** | The likely performance bottleneck in the analyzed window. |
| **Why** | The measurements supporting that verdict. |
| **Next** | The first change or investigation to try. |
| **Run context** | Check the device, observed ranks, analyzed steps, and duration before comparing runs. |

In this example, input waiting takes most of the step. Inspect the input
pipeline before optimizing model compute. The numbers illustrate how to read
the report; they are not a performance benchmark.

## Understand the timing breakdown

| Measurement | Meaning |
| --- | --- |
| **Step Time** | Input Wait plus Traced Step Time, using one selected clock. |
| **Input Wait** | Time waiting for training input on that clock; it is not worker-side preprocessing time or a direct GPU-idle measurement. |
| **Compute** | The measured Forward, Backward, and Optimizer phases combined. |
| **Forward** | Observed model computation, according to the integration's boundaries. |
| **Backward** | Observed gradient computation; it can include distributed synchronization. |
| **Optimizer** | Observed update work; the exact boundary depends on the integration. |
| **H2D** | Observed host-to-device transfers. |
| **Residual** | Traced training time not attributed to the measured phases. It is not automatically wasted time. |
| **DataLoader Fetch (CPU)** | Supplemental CPU fetch timing. It is not added to Step Time again. |

The report uses GPU timing only when the required GPU signals are complete;
otherwise it selects CPU timing. Read the clock label before comparing values.
Unmeasured values are omitted or shown as `n/a`, rather than replaced with
zero. A displayed `0.0 ms` can be a measured value rounded to display precision.
H2D events occur only when transfers are observed; absent H2D alone does not
mean timing coverage is incomplete.

Step boundaries differ between trainers and custom loops. See your
[integration guide](integrations.md) for accumulation, transfers, and optimizer
coverage.

## Read distributed results

For multiple ranks, the final timing breakdown comes from one representative
rank: the rank whose window-average Step Time is closest to the cross-rank
median. Its phases stay together; the report does not combine unrelated
per-phase medians.

| Label | Meaning |
| --- | --- |
| **N / R / G** | Node, global rank, and GPU index. |
| **Median** | The middle value across the observed ranks or nodes for the displayed measurement. |
| **Worst** | The largest or most pressured value for that measurement, not necessarily the same rank in every table. |
| **Worst Rank / Node** | Where that value was observed. |
| **Skew (%)** | How far the worst value is above the median. |

If the verdict identifies a straggler, inspect the named culprit and its
supporting phase evidence. Large percentage skew on a tiny timing value can
still be a minor issue.

## Check memory and resource context

**Step Memory** shows CUDA allocation peaks within measured steps and findings
such as pressure, imbalance, or growth. Its reported averages of per-step
peaks differ from the sampled process-memory averages.

**System** shows host and GPU resource measurements. **Process** shows the
training processes' CPU and memory use. Their available measurements and
section diagnoses help explain the run, but low GPU utilization alone does
not prove an input bottleneck.

Read the diagnosis together with its evidence. A memory trend hint or a
single utilization percentage is not the complete diagnosis.

## Find the saved report

```text
logs/<run-name>/final_summary.json
logs/<run-name>/final_summary.txt
```

The JSON contains structured diagnoses and measurements for
[Compare Runs](compare.md) and [Regression Guard](regression-guard.md).
Read a saved report with:

```bash
traceml view logs/<run-name>/final_summary.json
```

Training stdout and stderr are saved by default under
`logs/<run-name>/nodes/node_<node-rank>/`. If training fails, start with
[Training Crashes](training-crashes.md) for log locations and incomplete
telemetry.

### Shareable HTML report

Add `--html-report` to `traceml run` (or `traceml watch`) to also write
`final_summary.html` next to the JSON/TXT. It is a single self-contained file
(inline styling and charts, no JavaScript, no network requests) that opens in
any browser and is easy to drop into Slack, an email, or an issue. It shows a
run header, a top-level verdict from `primary_diagnosis` in schema 1.5 and later
reports,
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

`traceml view` prints the card that was stored with the run, so an artifact
always reads back exactly as it was written. To read an older artifact in the
current card layout, rebuild it from the same JSON:

```bash
traceml view logs/<run_name>/final_summary.json --re-render
```

This only changes what is printed. The stored artifact is not modified.
Because earlier schemas use different Step Time meanings, payloads older than
schema 1.7 keep their stored card instead of using the current renderer.

## What the summary, CLI, and local UI show

The final summary is the default. For live feedback, choose one of these
single-node views, including for single-node multi-GPU runs.

### Live CLI

```bash
traceml run train.py --mode=cli
```

The terminal updates step timing, step memory, system, and process findings
while training runs.

### Local UI

```bash
traceml run train.py --mode=dashboard
```

The browser dashboard provides timing breakdowns, memory trends, resource
cards, and a diagnostics rail. Start with the diagnosis, then inspect the
corresponding chart or table. Multi-node runs use summary mode.

`traceml watch` provides system and process visibility without training-step
measurements or a performance verdict. To diagnose step timing, use `run`
with a supported trainer or explicit loop setup.

## Common next actions

| Diagnosis | Good next step |
|---|---|
| `INPUT-BOUND` | inspect input loading, preprocessing, and storage |
| `H2D-BOUND` | inspect pinned memory, batch transfer, and host-to-device copies |
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

## Measurement and diagnosis reference

Use the sections below when you need a field definition or want to interpret
a specific diagnosis.

## Step Time glossary

TraceML uses one selected clock for an aligned analysis window. It selects GPU
only when every required timing signal is complete on GPU for every included
rank and step; otherwise it uses CPU. The selected clock is shown as
`diagnosis_clock` in JSON and in the live-output footer.

```text
Step Time = Input Wait + Traced Step Time
Residual = Traced Step Time − H2D − Forward − Backward − Optimizer
```

- **Step Time** is the complete selected-clock duration used for diagnosis,
  displayed shares, and run comparison.
- **Traced Step Time** is the inner duration instrumented by
  `traceml.trace_step(...)`; it is a subtotal, not an end-to-end replacement
  for Step Time.
- **Input Wait** is selected-clock time spent waiting for the next batch.
- **DataLoader Fetch (CPU)** is supplemental CPU evidence. When GPU is
  selected it can differ from Input Wait; it is never added into Step Time or
  counted twice.
- `null` / `n/a` means a signal was not measured in the window. A measured
  `0.0 ms` remains a real zero.

The Run terminal card keeps Traced Step Time in the structured summary but
does not print that redundant subtotal. It shows Input Wait, Compute, H2D,
and Residual directly beneath Step Time instead.

Examples:

| Window | Selected values | Interpretation |
|---|---|---|
| GPU complete | GPU Step Time and GPU Traced Step Time | GPU timing is used consistently for phases and shares. |
| GPU incomplete | CPU Step Time and CPU Traced Step Time | CPU is selected; unavailable GPU aggregates remain `null`. |
| Partial GPU evidence | CPU selected values plus optional GPU aggregates | Do not replace missing GPU values with zero or mix clocks. |

The raw `_traceml_internal:step_time` event is an internal persisted telemetry
key. After SQLite normalization, user-facing output calls that inner metric
Traced Step Time.

## Step-time diagnoses

<details markdown="1">
<summary>Timing diagnoses, evidence, and suggested actions</summary>

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

A signal that was never measured in the analyzed window is reported as
missing (`null` in `final_summary.json`, `n/a` in text), never as a fake
`0.0`. A phase measured at zero milliseconds stays `0.0`.

### `INCOMPLETE DATA`

Meaning:

- timing samples exist, but one or more phase signals were never measured,
  and no reliable conclusion is possible from the signals that remain

This usually means:

- an integration or manual-mode setup did not instrument every phase (for
  example calling `model.forward(...)` directly, an unwrapped DataLoader,
  or a custom optimizer without `wrap_optimizer`)

What to do next:

- check the missing signal names listed in the diagnosis evidence
- use the recommended automatic initialization for your training framework,
  or use the matching `wrap_*` helpers in manual/selective mode

---

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

- Input Wait is taking a large share of selected-clock Step Time
- TraceML uses the median per-rank input share, so one unusually slow rank
  does not hide a broad input bottleneck

Common causes:

- too few dataloader workers
- slow preprocessing
- slow storage
- slow host-to-device copies

What to look at:

- `Input Wait`
- its share of selected-clock Step Time
- whether the issue is broad or rank-specific

What to do next:

- increase dataloader workers
- reduce preprocessing cost
- improve storage throughput
- inspect batch construction

---

### `H2D-BOUND`

Meaning:

- host-to-device transfer is taking a large share of typical selected-clock
  Step Time
- TraceML uses the median per-rank H2D share, so one unusually slow rank does
  not hide a broad transfer bottleneck
- this diagnosis requires GPU-selected timing; CPU host-call duration is not
  treated as transfer cost

TraceML reports a warning at 10% of Step Time and critical severity at
20%.

What to do next:

- use pinned host memory where appropriate
- inspect batch transfer placement and non-blocking copies
- reduce transferred batch size or unnecessary host-to-device copies

`H2D-BOUND` describes broad transfer cost. `H2D STRAGGLER` instead describes
one rank with excess transfer time.

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

- a meaningful part of Traced Step Time is not attributed to H2D, forward,
  backward, or optimizer work

In TraceML:

- `compute = forward + backward + optimizer`
- `residual = traced_step_time - h2d - compute`
- `step_time = input_wait + traced_step_time`

TraceML evaluates residual as the median per-rank share of selected-clock Step
Time. Input and residual findings warn at
10% and are critical at 20%; rank skew is supporting evidence rather than a
gate that hides a typical bottleneck.

This is residual unattributed time inside Traced Step Time, not direct
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

</details>

## How to read the step-time table

<details markdown="1">
<summary>Live table columns and rank statistics</summary>

In the CLI step summary, the important columns are:

- `IW` / `Input Wait`
- `H2D`
- `Forward`
- `Backward`
- `Optimizer`
- `Step Time`
- `Traced Step Time`
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

- how much of Traced Step Time is unattributed to H2D, forward, backward, or
  optimizer work

</details>

## Step-memory diagnoses

<details markdown="1">
<summary>Memory pressure, imbalance, and growth diagnoses</summary>

The step-memory diagnosis explains memory pressure, imbalance, and drift over time.

The end-of-run summary always includes Step Memory status, compact Evidence,
and available allocated/reserved average-per-step-peak rows. On a distributed
run, the stored median and worst reserved-memory points select grouped rank
rows, and both Allocated and Reserved come from each selected row. If a
reserved-memory selector is unavailable, its allocated-memory selector is
used instead.

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

</details>

## How to read the step-memory table

<details markdown="1">
<summary>Memory peaks and trend fields</summary>

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

</details>

## System metrics

<details markdown="1">
<summary>Host and GPU measurements</summary>

The system panel reports machine-level pressure and GPU-utilization symptoms.
It is still context for the training diagnosis: low or moderate GPU
utilization says the GPU was not fully busy, but it does not prove why.

The end-of-run summary always includes the System status and available core
CPU, RAM, GPU utilization, GPU memory, temperature, and power measurements.
These are averages over the shared final-report timestamp interval. One
observed node uses an `avg` column. Multiple
observed nodes use `median node avg` and `worst node avg` columns, with the
stored worst node shown per metric. This is a comparison of node averages, not
peak values over time.

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
- GPU power
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

</details>

## Process metrics

<details markdown="1">
<summary>Training-process measurements</summary>

The process panel shows what the training processes themselves are consuming.

The end-of-run summary includes the Process status and available CPU capacity,
RSS, CUDA allocated, and CUDA reserved measurements. One observed rank uses an
`avg` column. Multiple observed ranks use `median rank avg` and
`worst rank avg` columns, with stored worst-rank and node identities shown
when available. These values cover the shared final-report timestamp interval;
they may have a different sample count from Step Time and are not temporal
peaks.

For a non-normal status, `Evidence` shows the stored diagnostic trigger—for
example, peak RSS pressure, CUDA memory pressure, allocator overhang, or
cross-rank CUDA-memory imbalance. A normal Process status shows
its measurements without a redundant evidence line.

It helps answer:

- how much CPU capacity the processes used on average
- how much RSS and CUDA memory the processes used on average
- whether process-level GPU memory is imbalanced

Common fields:

- CPU capacity
- RSS used
- CUDA allocated
- CUDA reserved, including device-capacity percentage when available

Use this panel when:

- the step diagnosis looks odd
- you want rank-level process context
- you suspect a specific rank is heavier than the others

</details>

## Advanced report details

<details markdown="1">
<summary>Aligned windows, selected ranks, and rendering</summary>

The final report resolves one interval from the latest
optimizer step completed across all observed ranks. System and Process query
that same timestamp interval, while Step Memory uses the same step bounds.
Their sample counts can differ because their sampling rates differ.
Run and Watch share the same 156-column System/Process layout. Run also shows
Step Timing, Step Memory, and a diagnostic `Verdict`/`Why`/`Next`. Watch omits
those performance sections and instead points to `trace_step(model)` and
`traceml run` for step-time measurement. Watch gets rank coverage from Process
telemetry and node coverage from System telemetry; missing Process data stays
at zero observed ranks rather than borrowing the expected world size. Its
scope legend appears only when the card uses an `N`, `R`, or `G` identity.
In the multi-node System table, each metric uses the median and worst
node-average points already stored in the summary. “Worst” is not a temporal
maximum, and different rows can identify different nodes. RAM and GPU-memory
bytes/percent stay paired from the same selected node row.
The Process table follows the same rule across observed ranks: every value is
an observation-window average, each metric may name a different worst rank,
and RSS and CUDA-reserved byte/percentage pairs come from one selected rank
row. For non-normal System and Process statuses, `Evidence` uses a compact
presentation of the stored structured trigger and scope, falling back to the
stored diagnosis summary when needed. Normal statuses leave the evidence row
blank. The table itself never labels its averages as peaks.
Evidence and long values wrap within their pane and retain the fixed divider
position.
Unavailable measurements are omitted, while a measured zero remains visible.
For older or partial artifacts that lack the stored fields needed by a
structured scope, the renderer preserves the stored diagnosis summary instead
of inferring new evidence.

Detailed section prose remains in the `system.card`, `process.card`,
`step_time.card`, and `step_memory.card` fields inside `final_summary.json`.
If terminal-card rendering itself fails during shutdown, TraceML prints a
minimal failure card and directs the user to the structured JSON evidence.

</details>

### TraceML internal error logs

TraceML implementation errors are separate from training stderr:

```text
logs/<run-name>/rank_<global_rank>/traceml_errors.log
logs/<run-name>/aggregator/traceml_errors.log
logs/<run-name>/nodes/node_<node_rank>/launcher_errors.log
```

User exceptions remain in the saved training stderr. See
[Training Crashes](training-crashes.md) and the [CLI reference](public-api.md#cli)
for output capture and telemetry-failure details.

## Next Steps

- [Compare Runs](compare.md)
- [Catch Regressions in CI](regression-guard.md)
- [Training Integrations](integrations.md)
- [Training Crashes](training-crashes.md)
