# Rank Straggler Policy

This page documents how TraceML decides that one rank in a distributed job is a
straggler, which rank it blames, and how confident it is. It is the developer
reference behind the Step Time rank-straggler diagnoses. For the user-facing
walkthrough of a slow rank, see
[Diagnosing a slow rank in DDP](../guides/ddp-slow-training-rank-straggler.md).

The implementation lives in `src/traceml_ai/diagnostics/` (rule evaluation and
culprit/victim context) and `src/traceml_ai/reporting/primary_diagnosis.py`
(evidence assembly). The canonical policy summary is
`src/traceml_ai/diagnostics/DIAGNOSIS.md` (Step Time section); this page expands
on it.

## Diagnosis kinds

Rank-straggler diagnoses describe skew *across ranks* in a single analyzed
window, as opposed to the single-rank bottleneck kinds (`INPUT_BOUND`,
`COMPUTE_BOUND`, `RESIDUAL_HEAVY`).

| Kind | Meaning |
|---|---|
| `STRAGGLER` | Visible rank skew exists, but input wait, H2D, and forward do not explain the likely culprit. |
| `INPUT_STRAGGLER` | The culprit rank has materially higher selected-clock input wait than the victim. |
| `COMPUTE_STRAGGLER` | DDP / default strategy only: the culprit rank has materially higher forward time than the victim. |
| `H2D_STRAGGLER` | The culprit rank has materially higher host-to-device transfer time than the victim. |

## Selected clock and eligibility

Step Time diagnosis analyzes one *selected clock* for the window. It uses GPU
event timing when every rank and step has GPU timing for the step envelope,
input wait, and traced phase events; otherwise it falls back to explicit
`cpu_ms` timing. All rank-straggler comparisons below are in that selected
clock.

TraceML first picks one visible synchronization phase per rank:

```text
visible_r = backward_r              # DDP / default
visible_r = forward_r + backward_r  # FSDP
```

Only ranks with a measured visible-phase anchor and a measured step envelope are
eligible:

```text
DDP / default: backward_r > 0 and step_time_r > 0
FSDP:          forward_r > 0 and backward_r > 0 and step_time_r > 0
```

If fewer than two ranks are eligible, TraceML does not report a rank straggler.
Missing visible instrumentation makes the rule abstain rather than treat a
zero as a fast rank. Component values of `input_wait == 0` and `h2d == 0` are
still valid measurements and are kept.

## Culprit and victim selection

- The **culprit** is the rank with the minimum visible value. It is the rank
  that most likely arrived late at the synchronization phase and therefore
  waited least in it.
- The **victim** is the upper actual median rank by visible value. Using the
  upper median rather than the maximum keeps the reference rank representative
  of the group instead of a single outlier.

## Straggler score

The score normalizes the visible gap between victim and culprit by the victim's
own selected-clock iteration cost:

```text
denom = input_wait_victim + step_time_victim
score = (visible_victim - visible_culprit) / denom
```

If `score < 0.10`, TraceML does not report a rank straggler. The gap is treated
as normal run-to-run variation rather than a real straggler.

## Attribution

Once the score clears the threshold, TraceML compares the culprit directly with
the victim to explain the visible wait:

```text
input_excess   = input_wait_culprit - input_wait_victim
h2d_excess     = h2d_culprit - h2d_victim
forward_excess = forward_culprit - forward_victim   # DDP / default only
```

The largest material positive excess selects the kind: `INPUT_STRAGGLER`,
`H2D_STRAGGLER`, or (DDP / default only) `COMPUTE_STRAGGLER`. DDP / default
compute attribution requires measured forward time on both the culprit and the
victim. If no excess explains the visible wait cost, the diagnosis stays
`STRAGGLER` with a sync-or-unattributed component. That residual can cover
validation, checkpointing, logging, framework orchestration, CPU stalls, or
unobserved transfer stalls inside the timed step. It is not direct NCCL,
all-reduce, or synchronization timing.

## Training strategy context

When runtime environment metadata is available, Step Time diagnosis receives an
advisory training strategy such as `ddp` or `fsdp`. This context only chooses
attribution behavior; it is not a public Step Time metric. Missing or
unrecognized strategy metadata defaults to `ddp` to preserve the DDP / default
visible-backward straggler behavior.

- DDP / default stays eligible for critical rank-straggler diagnoses once
  thresholds and confidence gates are met.
- FSDP diagnoses are capped at warning, because collective masking can make
  cross-rank attribution under-confident.

## Confidence gates

Shared Step Time diagnosis needs at least 2 steps to emit warning-only
diagnoses, and at least 20 steps before a critical diagnosis is allowed. The
live CLI table, dashboard, and final summary use the same gates and the same
global-rank window loader; they differ only by the selected timing window size.

## See also

- [Diagnosing a slow rank in DDP](../guides/ddp-slow-training-rank-straggler.md)
- [Reading Output](../user_guide/reading-output.md)
- `src/traceml_ai/diagnostics/DIAGNOSIS.md` (canonical Step Time policy)
