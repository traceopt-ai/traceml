## What changed

<!-- Briefly describe the user-visible change. -->

## Why

<!-- What problem does this solve for TraceML users or contributors? -->

## How I tested

<!-- Include exact commands, examples, or environments used. -->

- [ ] Unit tests
- [ ] Integration or smoke test
- [ ] Example training script
- [ ] Docs-only change
- [ ] Not run; reason:

## Runtime impact

<!-- Required for tracing, sampler, transport, aggregator, or reporting changes. -->

- [ ] No training-path runtime impact
- [ ] May affect training-path overhead
- [ ] May affect distributed launch, telemetry, or aggregator behavior
- [ ] Not applicable

Notes:

## Documentation

- [ ] README updated
- [ ] User docs updated
- [ ] Examples updated
- [ ] API docs updated
- [ ] Not needed

## Risk checklist

- [ ] Does not add unnecessary CUDA synchronizations
- [ ] Does not add blocking I/O on the training path
- [ ] Fails safely without crashing user training
- [ ] Keeps new dependencies optional unless discussed
- [ ] Redacts or avoids sensitive training-environment data in examples

## Screenshots or output

<!-- Paste relevant CLI output, final_summary text, viewer screenshot, or compare output. -->
