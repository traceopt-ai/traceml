# What Happens When My Training Crashes

TraceML preserves training failures and saves stdout and stderr from the
supervised training command. Start with those logs to find the failing rank
and the original error.

Training status and telemetry health are separate: training can fail while
TraceML successfully saves the measurements collected before the failure.

## Where to look first

1. Read the final training message and the command's exit code.
2. Open `logs/<run-name>/nodes/node_<node-rank>/training.stderr.log`.
3. Find your exception, a `Fatal Python error` frame dump, or torchrun's
   `Root Cause` block identifying the failing worker.
4. Check `status` and `telemetry_status` in `logs/<run-name>/manifest.json`.
5. If telemetry failed, inspect `aggregator/process.stderr.log` on node 0.

A traceback for torchrun's `ChildFailedError` describes a worker failure. It
is not necessarily the original exception from your training code.

## What is saved

Paths below are relative to `logs/<run-name>/`:

| Artifact | What to look for |
| --- | --- |
| `nodes/node_<n>/training.stderr.log` | Python traceback, fatal-signal frame dump when available, and torchrun's failing-rank details. |
| `nodes/node_<n>/training.stdout.log` | Training output before the failure. |
| `manifest.json` | Training status, telemetry health, and saved artifact paths. |
| `aggregator/process.stderr.log` | Aggregator diagnostics; owned by node 0 in multi-node runs. |
| `final_summary.json`, `final_summary.txt` | Measurements saved if history was enabled and summary finalization succeeded. |
| `aggregator/finalization_warning.json` | Missing ranks or other finalization warnings, when present. |
| `aggregator/finalization_error.json` | Finalization failure details, when present. |

Training stream saving is enabled by default. `--no-save-training-output`
lets training inherit the terminal and creates no training stdout/stderr
files. It does not disable aggregator diagnostics.

Direct `python` or `torchrun` launches with `traceml serve` leave training
output capture to your terminal, scheduler, or container runtime.

## Python exceptions

An uncaught exception propagates through the worker. TraceML stops its runtime
in cleanup without suppressing the training error.

On the torchrun path, TraceML uses PyTorch's error recorder. The saved stderr
includes the traceback and torchrun's `Root Cause` block. PyTorch may also
write a temporary `error.json` and report its path as `error_file`; TraceML
does not copy that file into the run folder.

Normal cleanup sends remaining telemetry and a rank-finished marker. If the
aggregator also finalizes successfully, the report is saved and telemetry
can be `complete` even though training failed. In a distributed failure,
other workers may be terminated before completing their cleanup.

## Native crashes and signal deaths

A segmentation fault or a killed process can bypass Python cleanup. Telemetry
still queued in that worker is lost.

On the torchrun error-recorder path, Python's fault handler can print a frame
dump for fatal signals such as `SIGSEGV`:

```text
Fatal Python error: Segmentation fault

Current thread ...:
  File "/path/to/train.py", line 23 in main
```

This shows Python frames, not the native C, C++, or CUDA stack. Some deaths,
such as `SIGKILL`, provide no fault-handler dump. In those cases, inspect
scheduler or system logs as well as torchrun's worker details.

A torchrun `Root Cause` block can show the worker's signal:

```text
rank      : 0 (local_rank: 0)
exitcode  : -11 (pid: ...) (SIGSEGV)
error_file: <N/A>
traceback : Signal 11 (SIGSEGV) received by PID ...
```

The CLI returns the supervised process's exit code. On the torchrun path,
this is torchrun's nonzero exit code; the worker's signal is in its
`Root Cause` block. If the supervised process itself dies from a signal,
TraceML returns `128 + signal number`.

## What the terminal prints

Summary and dashboard modes show training output live. CLI mode keeps it out
of the live display and, after a failure, prints a saved stderr excerpt of
up to 40 lines and 8 KiB.

With stream saving enabled, the final output includes the log paths:

```text
[TraceML] Stderr: /path/to/logs/<run-name>/nodes/node_0/training.stderr.log
[TraceML] Stdout: /path/to/logs/<run-name>/nodes/node_0/training.stdout.log
```

The aggregator-owning launcher also reports telemetry health before the final
training outcome. Other nodes do not report authoritative aggregator health.
A telemetry failure does not replace the training exit code after training
has started.

## Telemetry and the final summary after a crash

A report after a crash contains only telemetry that reached the aggregator.
An early failure can leave too few completed steps for a diagnosis. A final
summary is not guaranteed if the aggregator fails, history is disabled, or
summary generation cannot finish.

When a dead worker leaves a missing rank-finished marker, the aggregator can
finalize after all rank connections close and incoming telemetry settles.
Successful finalization with missing ranks records a warning and normally
reports `telemetry_status: degraded` with
`telemetry_reason: finalization_warning`.

If connections remain open, finalization waits within its configured budget:
`--finalize-timeout-sec`, 300 seconds by default. This is a telemetry deadline;
it does not terminate a hung training job.

See [Reading the Output](reading-output.md) before interpreting an incomplete
run as evidence of a performance problem.

## Manifest fields

| Field | Meaning |
| --- | --- |
| `status` | `completed` for exit code 0, `failed` for a nonzero training exit, or `interrupted` when the launcher receives an interruption. |
| `lifecycle.training_ended_at` | When the supervised training process exited, if recorded. |
| `artifacts` | Paths to saved training streams, aggregator diagnostics, and reports that exist. |
| `telemetry_status`, `telemetry_reason` | Whether telemetry completed, degraded, failed, or was unavailable, and why. |
| `aggregator_exit_code` | Aggregator process outcome, when known to its owner. |

Use the command's exit code and torchrun's worker details for the training
exit code or signal; the manifest does not have a general training-exit-code
field.

## Related

- [FAQ: Training output files](faq.md#where-are-training-stdout-and-stderr-saved)
- [Distributed Training](distributed-training.md)
- [Missing-aggregator behavior](public-api.md#missing-aggregator-behavior)
