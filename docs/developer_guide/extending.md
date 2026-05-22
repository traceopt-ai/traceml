# Extending TraceML

This guide is for contributors adding a new metric, diagnosis, summary section,
or compare field. It follows the current code layout and avoids older internal
paths.

## Mental Model

```text
training process
  -> samplers collect telemetry
  -> runtime sender publishes batches
  -> aggregator stores SQLite history
  -> live display reads renderer/computer payloads
  -> final report builds reporting sections
  -> compare reads final summary JSON
```

Live UI and final summaries are separate paths. They can share diagnostics, but
they should pass explicit policies such as `LIVE_STEP_TIME_POLICY` or
`SUMMARY_STEP_TIME_POLICY` when thresholds differ.

## TraceML Lifecycle

TraceML has two runtime pieces:

- one aggregator, which receives TCP telemetry, writes SQLite history, and
  creates the final summary
- one runtime per training process, which samples local telemetry and sends
  batches to the aggregator

CLI launchers may run these pieces as subprocesses. Framework integrations may
run them inside Ray actors or worker processes. Both paths should use
`traceml.runtime.lifecycle` so startup and shutdown stay consistent.

The owner that starts a component must stop it. Use `try/finally` around
training work, and make stop paths safe to call more than once.


## Add a Diagnostic Rule

Diagnostics live under `src/traceml/diagnostics/<domain>/`.

Current domains include:

- `system`
- `process`
- `step_time`
- `step_memory`

Typical files:

```text
context.py   normalized input signals
policy.py    thresholds and named policies
rules.py     one rule class per issue
api.py       public builder that runs rules and selects primary diagnosis
```

Add one rule class in `rules.py`, add it to the domain's default rule tuple,
then update priority sorting if the new issue should beat existing issues.

Tests should live in `tests/diagnostics/` and cover:

- the rule triggers
- the rule does not trigger for normal input
- priority when multiple issues trigger together

## Add a Summary Section

Final-report sections live under `src/traceml/reporting/sections/`.

Current sections:

- `system`
- `process`
- `step_time`
- `step_memory`

Each section follows this shape:

```text
loader.py     read SQLite / section inputs
builder.py    build JSON payload and card text
formatter.py  render section text
model.py      section-local data helpers
```

Register sections through `src/traceml/reporting/final.py`. Keep the aggregator
as a caller only; report assembly belongs in `reporting`.

Tests should live in `tests/reporting/summary/`. Prefer small SQLite fixtures
over large golden snapshots. Assert stable schema keys and a few important text
lines.

## Add a Sampler

Runtime sampler selection is in `src/traceml/runtime/sampler_registry.py`.

To add a sampler:

1. Implement a `BaseSampler` subclass under `src/traceml/samplers/`.
2. Add a `SamplerSpec` to `DEFAULT_SAMPLER_REGISTRY`.
3. Restrict it by `profiles` and `modes` so it only runs where needed.
4. Add SQLite projection, renderer, or summary code only if the data is
   user-facing.

Deep/layer profiling has been removed from the public CLI for now. Keep normal
sampler changes scoped to `run` and `watch`, and do not document layer-level
profiling as a public path unless that surface is reintroduced deliberately.

Tests should live in `tests/runtime/` for selection behavior and in a more
specific folder if the sampler has domain logic.

## Add a Compare Metric

Compare code lives under `src/traceml/reporting/compare/`.

Important files:

```text
sections/<section>.py  extract comparable values from final summary JSON
model.py              typed compare objects
verdict.py            rule-based verdict selection
formatters.py         terminal text output
core.py               payload assembly
```

Add metric extraction to the relevant section comparer first. Only add a verdict
rule if the metric should affect the top-level outcome. Only show a row in the
text formatter if it helps users compare runs quickly.

Tests should live in `tests/reporting/compare/` and cover missing data, changed
values, and verdict priority when multiple signals disagree.

## Add Live Display

Live display code is renderer-driven. CLI and dashboard renderers may differ.

Relevant paths:

- `src/traceml/renderers/`
- `src/traceml/aggregator/display_drivers/`

Keep renderer methods focused on presentation. Put data shaping in a compute
object or formatter when the logic is reusable or non-trivial.

## Fail Open

TraceML should not break user training because optional telemetry, rendering,
or reporting failed. Existing code logs advisory failures through
`traceml.loggers.error_log.get_error_logger`.

Use that pattern for non-critical paths:

```python
logger = get_error_logger("MyComponent")
try:
    ...
except Exception as exc:
    logger.exception("[TraceML] MyComponent failed: %s", exc)
```

Prefer returning an empty payload, `NO DATA` diagnosis, or fallback text over
raising from live display, compare rendering, or final-report generation.

## Test Layout

Tests are grouped by area:

```text
tests/core/
tests/diagnostics/
tests/reporting/summary/
tests/reporting/compare/
tests/runtime/
tests/sdk/
tests/telemetry/
tests/display/
tests/integrations/
```

Keep tests close to the behavior they protect. The most valuable tests are
small and direct: rule behavior, priority, schema shape, and fail-open behavior.
