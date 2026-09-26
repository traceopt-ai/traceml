# What Happens When My Training Crashes

This page explains what `traceml run` keeps when your training process dies.
It covers two kinds of failure.

- A **Python exception** that nobody catches, such as a `RuntimeError` or a CUDA out-of-memory error.
- A **native crash or signal death**, such as a segmentation fault inside a C or CUDA extension.

TraceML does not write a crash report of its own.
It keeps the evidence that Python and torchrun already produce, and it saves the raw output streams of your training process.

---

## Where to look first

1. Read the last `[TraceML]` line in the terminal and the command's exit code.
2. Open the saved training stderr log:
   `logs/<run-name>/nodes/node_<node-rank>/training.stderr.log`.
3. Search it for `Fatal Python error` (a native crash) or for your exception's name, such as `RuntimeError:` (a Python exception).
4. Read torchrun's `Root Cause` block near the end of the same file.
   It names the failing rank, its exit code, and the signal if there was one.
5. Check `status` and `telemetry_status` in `logs/<run-name>/manifest.json`.

torchrun prints a traceback of its own `ChildFailedError` whenever a worker fails.
A `Traceback` line alone therefore does not mean that your code raised an exception.

The terminal prints the exact paths of the saved logs after every run:

```text
[TraceML] Stderr: logs/<run-name>/nodes/node_0/training.stderr.log
[TraceML] Stdout: logs/<run-name>/nodes/node_0/training.stdout.log
```

---

## Python exceptions

TraceML runs your script unchanged.
An uncaught exception propagates to Python and torchrun exactly as it would without TraceML.

The full traceback ends up in `training.stderr.log`.
torchrun also prints a `Root Cause` block with the rank, an exit code of 1, and the same traceback.
torchrun writes its own `error.json` in a temporary directory and prints that path as `error_file`.
TraceML does not copy that file into the run directory.

TraceML still stops its per-rank runtime while the exception unwinds.
The rank sends its last telemetry and reports that it finished.
Telemetry therefore normally finishes as `complete`, and the final summary is written.

---

## Native crashes and signal deaths

A native crash kills the process without raising a Python exception.
There is **no exception traceback** from your code, because no Python exception exists.
The only traceback in the log is torchrun's own `ChildFailedError`.

Under `traceml run`, your script runs inside torchrun's error recorder.
That recorder turns on Python's `faulthandler` in the worker process.
When the process receives a fatal signal, `training.stderr.log` therefore contains a frame dump like this:

```text
Fatal Python error: Segmentation fault

Current thread 0x... (most recent call first):
  File "/path/to/train.py", line 23 in main
  ...
```

The dump lists the Python frames of every thread at the moment of the crash.
It shows which line of your script was running.
It does not show the native C, C++, or CUDA frame that faulted.

torchrun then reports the dead worker in its `Root Cause` block:

```text
  exitcode  : -11 (pid: 12345)  (SIGSEGV)
  error_file: <N/A>
  traceback : Signal 11 (SIGSEGV) received by PID 12345
```

The training process that TraceML supervises is torchrun, not the worker.
When a worker dies, torchrun exits with its own nonzero code, and `traceml run` exits with that same code.
The last terminal line reads `Training failed — torchrun exited with code <code>.`
The worker's signal is visible in torchrun's `Root Cause` block, not in the exit code.

If the supervised process itself is killed by a signal, the exit code follows the shell convention of 128 plus the signal number.
One example is torchrun being killed by the system.
In that case the last line reads `Training terminated by <SIGNAL> (exit code <code>).`

---

## What the terminal prints

In `--mode=cli`, the live display keeps training output off the screen.
After a failure, TraceML prints a bounded excerpt of the training stderr.
The excerpt covers at most the last 40 lines and the last 8 KiB.

```text
[TraceML] Training stderr excerpt:
<last lines of training stderr>
[TraceML] Stderr: logs/<run-name>/nodes/node_0/training.stderr.log
[TraceML] Stdout: logs/<run-name>/nodes/node_0/training.stdout.log
[TraceML] Telemetry <status line>
[TraceML] Training failed — torchrun exited with code <code>.
```

In `--mode=summary` and `--mode=dashboard`, training output is shown live as it happens.
These modes print no excerpt, but they print the same log paths.

A clean run prints no excerpt.

---

## What is saved

| File under `logs/<run-name>/` | Python exception | Native crash |
| --- | --- | --- |
| `nodes/node_<n>/training.stderr.log` | traceback and torchrun `Root Cause` | `faulthandler` frame dump and torchrun `Root Cause` |
| `nodes/node_<n>/training.stdout.log` | everything printed to stdout | everything printed to stdout |
| `aggregator/process.stderr.log` | aggregator stderr | aggregator stderr |
| `manifest.json` | `status: failed` | `status: failed` |
| `final_summary.json`, `final_summary.txt` | written | written, from telemetry that arrived before the crash |
| `aggregator/finalization_warning.json` | not written | lists the rank that never finished |

The stderr and stdout logs are only saved while training-output saving is on, which is the default.
With `--no-save-training-output`, training inherits the launcher's own stdout and stderr, and TraceML keeps no copy.

---

## Manifest fields

After a failed run, `manifest.json` records these facts:

- `status` is `failed` for any nonzero exit.
  It is `completed` for exit code 0.
  It is `interrupted` when the launcher itself receives Ctrl+C or `SIGTERM`.
- `lifecycle.training_ended_at` records when the training process exited.
- `artifacts.training_stderr_log`, `artifacts.training_stdout_log`, and `artifacts.aggregator_stderr_log` give the absolute paths of the saved streams.
- `telemetry_status`, `telemetry_reason`, and `aggregator_exit_code` describe telemetry health.

The manifest does not record the training exit code or the signal.
Use the command's exit code and torchrun's `Root Cause` block for those.

---

## Telemetry and the final summary after a crash

After a Python exception, the rank reports that it finished.
The aggregator finalizes without waiting out its deadline, and `telemetry_status` is `complete`.

After a native crash, the rank never reports that it finished.
Telemetry still queued inside the crashed process is lost.
The crashed process's connection to the aggregator closes when it dies.
Once no rank connection is still open, the aggregator waits one short quiet window for data already in flight, then finalizes.
It writes `aggregator/finalization_warning.json` with the missing rank in `missing_ranks`.
The manifest records `telemetry_status: degraded` with `telemetry_reason: finalization_warning`.
The terminal prints this line:

```text
[TraceML] Telemetry degraded: finalization completed with warnings.
```

The final summary is still written.
It covers only the telemetry that reached the aggregator before the crash.
A crash early in training can leave too few steps for a diagnosis.

A rank that hangs instead of dying keeps its connection open.
In that case the aggregator waits until its finalize deadline, `--finalize-timeout-sec` (300 seconds by default), before it finalizes.

---

## Related

- [FAQ: Where are training stdout and stderr saved?](faq.md#where-are-training-stdout-and-stderr-saved)
- [Distributed Training](distributed-training.md)
