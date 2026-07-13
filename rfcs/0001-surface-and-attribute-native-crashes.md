# RFC-0001: Surface and Correctly Attribute Native Training Crashes

| | |
|---|---|
| **RFC** | 0001 |
| **Title** | Surface and correctly attribute native training crashes |
| **Status** | Draft |
| **Created** | 2026-07-12 |
| **Tracking issue** | TBD (epic) |
| **Target release** | 0.4.x |

> Status legend: `Draft` -> `Discussion` -> `Accepted` -> `Implemented` (see `README.md`).

---

## 1. Summary

When a training process dies from a native fault (a segmentation fault, a host or
container OOM-kill, or any signal death), TraceML does not currently capture the
crash.
The run waits out its finalization timeout (up to ~5 minutes), then writes a summary
that attributes the stall to a TraceML rank-timeout rather than to the crash, which
points the user at TraceML instead of at the fault they need to debug.
We reproduced this end-to-end (a forced native crash under `traceml run --mode
summary`) on both CPU and a single-GPU CUDA run.

This RFC proposes: install `faulthandler` in the training process; write a
structured `crash.json` on signal death; make finalization *crash-aware* so a dead
worker no longer hangs the run or is mislabeled a TraceML timeout; and optionally
capture the child's stderr tail. Attribution is by cause, verified against an authoritative liveness
oracle, and never claims more than the evidence supports.

## 2. Motivation

### Background (verified against `main`, e7813c1)

Walk a `traceml run` in summary mode where the training process dies on `SIGSEGV`:

- The user script runs inside `runtime/executor.py` (`runpy.run_path`,
  `executor.py:346`). Its crash logger (`write_user_error_log` ->
  `torchrun_error.log`, `executor.py:113-130`) is only reached from the Python
  handlers in `main()` (`executor.py:413-445`). A native fault raises no Python
  exception, so none of these run and no `torchrun_error.log` is written.
- No `faulthandler` is installed anywhere in the package. Python's `faulthandler`
  is the standard-library tool that dumps a C-level traceback on `SIGSEGV`/
  `SIGABRT`/`SIGBUS`/`SIGFPE`/`SIGILL`; without it, a native fault leaves no trace.
- The child is spawned with no output capture
  (`subprocess.Popen(train_cmd, env=env, cwd=cwd, start_new_session=True)`,
  `launcher/process.py:148-153`), so it inherits the console. The torchrun "Signal
  11 (SIGSEGV) received" block prints to the terminal and is never persisted. In
  summary mode the stdout/stderr sampler is off (`runtime/sampler_registry.py:74`,
  `modes=("cli",)`), so console output is not captured to disk either.
- The graceful `rank_finished` marker is sent only from `TraceMLRuntime.stop()`
  (`runtime/runtime.py:207,230-231`), which never runs on a signal death. So the
  aggregator's settle loop, which only short-circuits when
  `len(finished_ranks) >= expected_world_size` (`aggregator/trace_aggregator.py:
  439-479`), waits the entire finalize timeout for a marker that never arrives. The
  launcher terminates the aggregator with a `finalize_timeout + shutdown` budget
  (`launcher/commands.py:417-420`, default 300s), so the run hangs for minutes at
  `[TraceML] Training finished; stopping aggregator...`.
- The failure is then mis-framed on disk. With partial telemetry the aggregator
  produces a degraded all-`n/a` `final_summary.json` plus a `finalization_warning.json`
  that reads "Timed out waiting for all ranks to report finished... missing_ranks:[0]";
  with none it raises `TraceMLFinalizationError` and writes `finalization_error.json`
  (`trace_aggregator.py:287-307`).
- The exit code is propagated (`raise SystemExit(train_rc)`, `commands.py:453`;
  manifest `failed`, `commands.py:428`), but from the artifacts a user cannot tell a
  segfault from an OOM-kill from a TraceML bug.

**Net: two problems compound.**
(1) The run waits out the full finalize timeout for a shutdown marker a dead process
can never send, so it appears to hang for minutes.
(2) The only failure artifact on disk attributes the cause to a TraceML rank-timeout,
so it points the user at TraceML rather than at the crash they need to debug.
This is the gap the RFC closes: on a crash, TraceML should record the gap and its
cause, in line with the standing contract that a report is never silently incomplete.

**Reproduced.** A forced `SIGSEGV` under `traceml run --mode summary` (CPU, and a
single-GPU CUDA run) produced exactly this: no `torchrun_error.log`; the
`traceml_errors.log` files 0 bytes; the SIGSEGV visible only in torchrun's transient
stderr (`exitcode: -11`, `error_file: <N/A>`); a ~300s hang; a degraded all-`n/a`
summary; and a `finalization_warning.json` attributing the stall to a rank timeout. The only correct
signal was `manifest.json -> "status":"failed"`, which users do not read.

**Contrast cases that already work** (so we do not over-fix): an ordinary Python
exception, a CUDA-OOM (`torch.cuda.OutOfMemoryError` is a Python error), and a crash
inside a DataLoader worker (torch re-raises a `RuntimeError` naming the killed
worker) are all caught by `executor.py:436` and land in `torchrun_error.log` with
`final_summary.json` still produced. The gap is specifically the main process dying
on a signal.

### Goals

- **G1.** A native/signal death leaves a real trace on disk: a `faulthandler`
  C-level dump where possible, plus a structured `crash.json`, in every display mode.
- **G2.** Finalization is crash-aware: a dead worker never hangs the run, and the
  on-disk artifact attributes the failure to the crash, not to TraceML.
- **G3.** In summary mode, the child's stderr tail is optionally captured so a
  post-mortem is possible without re-running.
- **G4.** Attribution is by cause (`signal_death` / `likely_oom` /
  `scheduler_terminated` / `unknown`), and never over-claims: a cause is asserted
  only with evidence, otherwise the record hedges. A confident wrong cause is worse
  than an honest "unknown".
- **G5.** Preserve the core contracts: local-first (nothing leaves the box),
  best-effort instrumentation that never blocks or crashes training, and low
  overhead. All new wire frames are additive and back-compatible (an older peer that
  does not send them is tolerated, exactly as `rank_finished` absence is today).

### Non-Goals

- **N1.** Catching or preventing native crashes. TraceML records and attributes them.
- **N2.** A general log-aggregation subsystem. G3 is a bounded stderr tail.
- **N3.** Changing the diagnosis engine, wire format semantics, or the summary schema
  (crash artifacts are additive).
- **N4.** Deadlock/hang detection and elastic-restart handling. Both are real but
  separable; see §4.6.

## 3. Proposal

### User stories

- **US1.** A run segfaults after hours. Instead of a multi-minute hang and a
  TraceML-blaming artifact, the user sees `[TraceML] Training process was killed by
  SIGSEGV ...` and finds, in the run directory, a `faulthandler` dump (C traceback),
  `crash.json` (cause + signal + pid + host), and (if enabled) the stderr tail. They
  know it was their crash and roughly where.
- **US2.** A container run is OOM-killed (SIGKILL, no traceback possible). `crash.json`
  reads `likely_oom` with the last memory snapshot ("rank 3 at 61/64 GB, 98% of the
  cgroup limit"), not a generic "process died".

### UX / API surface

- G1/G2 require no new user API: `faulthandler` is installed by the executor; the
  crash report and crash-aware finalization are launcher/aggregator behavior.
- Optional stderr-tail capture (G3): opt-in `--capture-stderr` /
  `TRACEML_CAPTURE_STDERR=1` (default off; preserves today's inherited-console
  behavior and zero overhead when unused).

## 4. Design Details

### 4.1 Capture native crashes (G1)

**faulthandler.** Early in `runtime/executor.py` (before `runpy.run_path`,
`executor.py:346`, and before importing torch so a torch/NCCL handler does not
displace it) open a per-rank dump file and call
`faulthandler.enable(file=..., all_threads=True)`, guarded on
`faulthandler.is_enabled()`. On `SIGSEGV`/`SIGABRT`/`SIGBUS`/`SIGFPE`/`SIGILL`,
Python writes a C-level traceback to that file as the process dies. `all_threads=True`
because the fault is often in a background CUDA/dataloader thread. Best-effort
(guard the setup; never raise into training). Note that `faulthandler` does not chain
to a previously-installed custom handler; if a lower library replaces it, the
crash-aware finalization path (§4.2) still surfaces the death.

**crash.json.** In the launcher poll loop, when the child dies on a signal
(`train_rc < 0`), before `raise SystemExit(train_rc)` (`commands.py:453`) write a
structured `crash.json` to the session root, print one `[TraceML]` line pointing at
it, and add it to the manifest artifacts. Reuse the executor's best-effort write
discipline (`executor.py:90-110`; try/except, never raise). The record is a tagged
document with a machine `cause` enum plus a human `label`, so finalization can branch
deterministically and the artifact can render one honest sentence:

```
{ "cause": "signal_death" | "likely_oom" | "scheduler_terminated" | "unknown_native",
  "label": "<human sentence>",
  "global_rank": ..., "node_rank": ..., "hostname": ..., "pid": ...,
  "signal": 11, "signal_name": "SIGSEGV", "exit_code": -11,
  "timestamp": "...",
  "evidence": { ... },          // e.g. the memory snapshot for likely_oom
  "hints": [ "..." ] }          // e.g. "run `dmesg | grep -i oom` on <host>"
```

The launcher has rank/pid/host, which are what make a crash attributable in
DDP/multi-node rather than "something died". **Cause is derived from the signal, not
the raw exit code alone:** `SIGSEGV`/`SIGBUS`/`SIGABRT`/`SIGFPE` -> `signal_death`;
`SIGKILL` (-9) -> `likely_oom` only with memory evidence (§4.4), else `unknown_native`;
`SIGTERM` (-15) -> `scheduler_terminated` (an operator kill, Slurm `scancel`, or spot
preemption, not a crash). When the cause is ambiguous (bare SIGKILL with no
evidence), the record enumerates candidate causes with verification hints rather than
guessing one.

**SIGTERM handler.** A `SIGTERM` is catchable, leaves no `faulthandler` trace, and
today hangs the aggregator exactly like a SIGSEGV. The executor installs a best-effort
`SIGTERM` handler that flushes a `rank_finished` marker with `reason="terminated"`
before exiting, turning an orchestrator kill into a clean, attributed shutdown.

### 4.2 Crash-aware finalization (G2)

Today the aggregator waits ~300s for a `rank_finished` marker a dead process can
never send. The core principle: **a closed socket is never the death verdict, only a
trigger to consult an out-of-band liveness oracle.** TraceML already owns the perfect
oracle: the launcher holds a `subprocess.Popen` per rank, so `Popen.poll()` /
`os.waitpid` is an authoritative "is this PID alive, and with what exit code" source
that the noisy TCP stream can never be.

1. **Classify how the stream ended, not merely that it ended.** An explicit
   end-of-run goodbye frame (a new control frame the executor sends on normal finish
   or intentional restart) followed by EOF is a clean drain -> finalize. A raw EOF or
   `ConnectionReset` without a goodbye is a *candidate death* -> verify. A mid-frame
   decode error is a separate path (corrupt, not a close).
2. **EOF marks the rank `SUSPECT`, never `DEAD`.** The aggregator then consults the
   oracle (the launcher reports whether rank N's PID exited, and with what code). PID
   exited -> `DEAD` with the real returncode/signal -> `crash.json` and crash-aware
   finalize. PID alive -> either a transient blip (the `TCPClient` lazily reconnects
   after a send error, `transport/tcp_transport.py:246-258`, so a close is not proof
   of death) or `HUNG` ("rank silent but process alive", a distinct state, not a
   crash). This resolves the reconnect-vs-death ambiguity a bare socket-close cannot.
3. **Positive liveness from data freshness; silence only escalates.** Every telemetry
   frame is proof-of-life until the next expected tick. "No data for N ticks" is not
   death by itself (a rank can be in a long compute or H2D phase); staleness only
   starts a per-rank strikes counter (reset to full on any frame) and, past a grace,
   triggers the oracle check. An initial grace after each (re)connect avoids striking
   a rank that is still registering.
4. **Short bounded grace, then verdict.** On a candidate death, arm a few-second
   timer, not the 300s blind wait. A reconnect within it cancels; expiry plus an
   oracle-confirmed exit finalizes that rank.
5. **Idempotent, single-fire finalization.** A rank can be finalized by racing
   triggers (goodbye frame, EOF, grace-timeout, oracle exit event). Guard each rank
   with a state latch `LIVE -> DRAINING -> FINALIZED` (compare-and-set) so exactly one
   path writes its final rows and closes its stream, and a late reconnect cannot
   resurrect a finalized rank.
6. **Flush-telemetry-first, close-connections-last.** Finalization order: drain the
   read buffer to EOF, flush pending rows to SQLite (forced), commit, close the DB,
   write `final_summary`, close sockets last. Never close the DB (the reader) while a
   flush (the writer) is pending. Give it a bounded graceful budget with a force
   fallback: if the drain hangs, still close the DB and emit `final_summary` from what
   committed, then exit.
7. **Attribution.** When finalizing a crashed run, write a crash-attributed artifact
   (`run_crashed.json`, or set `cause:"training_crash"`) instead of the
   `finalization_warning.json` / `finalization_error.json` that today reads as a rank
   timeout (`trace_aggregator.py:287-307`).

Net: no multi-minute hang, no misattributed timeout, and no false death from a
transient reconnect. The launcher is the authoritative parent; the aggregator is the remote
observer that verifies through it rather than guessing from the socket.

### 4.3 stderr-tail capture (G3)

Under `--capture-stderr`, the launcher spawns the child with `stderr=subprocess.PIPE`
and a small reader thread that tees to the real console (live output unchanged) while
keeping a bounded ring-buffer tail; on child exit it flushes the tail to a local
`crash_stderr.log`. Because the child inherits the pipe as its fd 2, this captures the
native fd-2 output that `faulthandler` misses (the segfault backtrace, a CUDA/NCCL
abort, the OOM-killer preamble). The reader lives in the launcher, where a bug cannot
take down training, and must drain continuously to avoid a full-pipe deadlock.
Bounded memory, no behavior change when off, and the file stays local (nothing leaves
the box). Under torchrun the tail is the merged multi-rank stderr; the file is labeled
as such and a frame is not attributed to a specific rank. Per-rank Python context is
still preserved by the per-rank `faulthandler` dumps (§4.1).

### 4.4 OOM attribution (G4)

A host or container OOM kills the process with `SIGKILL`, which is uncatchable, so the
only usable signal is the memory sample TraceML already collects each tick. On a
`SIGKILL` death whose last sample was near the memory limit, `crash.json` records
`cause: "likely_oom"` with the memory snapshot (used / limit / fraction, plus the
culprit rank in a multi-rank run).

**Correctness requirement: measure against the cgroup limit, not host RAM.** TraceML
today reads `psutil.virtual_memory().total` as the memory denominator everywhere
(`samplers/process_sampler.py:81`, `samplers/system_sampler.py:102`) and reads no
cgroup limit. Most training runs (k8s, Slurm, Docker) are cgroup-capped well below
host RAM, so comparing against host total misses the common container-OOM case
entirely, which is exactly the death users most need explained. So OOM attribution
reads the cgroup limit and compares against it:

- Compare cgroup current usage (`memory.current` on cgroup v2, `memory.usage_in_bytes`
  on v1) against the limit (`memory.max` / `memory.limit_in_bytes`). Cgroup accounting
  includes page cache and sibling processes and is the truer container-OOM signal;
  this-process RSS-vs-limit is the fallback.
- Fail-open ladder: cgroup v2 -> v1 -> `/proc/meminfo`. On a literal `"max"`, an
  unparsable value, or an unlimited cgroup, fall back to host total and mark the
  attribution low-confidence. Never emit a confidently-wrong "OOM" label.
- Never claim OOM on `-9` alone (a kernel OOM-kill, a manual `kill -9`, and spot
  preemption are identical at the signal level); with no memory evidence the cause
  stays `unknown_native` and the record hints at `dmesg | grep -i oom`.

Scope: the cgroup read is added on the OOM-attribution path only. The broader fact
that TraceML's standing RAM% uses a host denominator (same two lines above) is a
related inaccuracy that touches diagnosis thresholds and wire meaning, so it is a
separate follow-up, not part of this RFC.

### 4.5 Distributed / multi-rank

The design extends to distributed runs with no new mechanism. `faulthandler` is
per-process, so each rank writes its own dump and a multi-rank crash yields the native
stack of the rank that actually faulted. Crash-aware finalization (§4.2) triggers when
any expected rank dies, not only rank 0. Under Ray, the framework already surfaces a
native worker death to the driver as a Python `ActorDiedError`, so it is partly
covered by the existing Python-exception path; TraceML's value there is the per-rank
`faulthandler` dump (the native fault location Ray's driver error lacks) and not
hanging the aggregator, and the Ray integration catches `ActorDiedError` to write
`crash.json`.

### 4.6 Deferred (out of scope for v1, recorded here)

Two capabilities are real but separable from crash surfacing, and are follow-ups:

- **Active liveness probe (deadlock / hung-training detection).** The launcher's
  `Popen.poll()` already tells hung-alive from dead. Distinguishing a *deadlocked*
  rank from a *legitimately slow* one needs an app-level PING/PONG with a deadline on
  the existing socket, the one new mechanism TraceML would add (telemetry is
  one-way push today) and is its own diagnosis ("training hang"), distinct from crash
  surfacing.
- **Restart / incarnation fencing.** torchrun-elastic can restart a failed rank in
  place with a new pid (`TORCHELASTIC_RESTART_COUNT`). Full handling stamps each frame
  `(rank, incarnation)`, keys telemetry on the stable rank, fences the series at the
  incarnation boundary (no averaging pre/post-restart monotonic state; drop the dead
  incarnation's slot from any "all N ranks reported" barrier), and caps tolerated
  incarnations (a flapping rank is itself a finding). Needed only once elastic-restart
  support is claimed. v1 carries a one-line guard: ignore a same-rank reconnect whose
  incarnation changed rather than double-count.

## 5. Test Plan

Per the precision-first discipline, each new attribution ships with a demo run where
it fires correctly and a control run where it stays quiet.

- **T1 (G1).** A fixture that faults on a signal (`os.kill(os.getpid(),
  signal.SIGSEGV)`) under `traceml run` asserts a nonzero exit, manifest `failed`, a
  `crash.json` with `cause:"signal_death"` and `signal_name:"SIGSEGV"`, and a non-empty
  `faulthandler` dump. Control: a clean run writes neither.
- **T2 (G1).** A Python-exception script still produces `torchrun_error.log` and no
  `crash.json` (no regression of the working path).
- **T3 (G2).** A signal-death run finalizes fast (well under the finalize timeout) and
  writes a crash-attributed artifact, not `finalization_warning.json` /
  `finalization_error.json`. Assert on elapsed time and artifact contents. Control: a
  clean run finalizes normally with a real summary.
- **T4 (G2).** A transient reconnect (socket close then reconnect with the same PID
  alive) does not finalize the rank as dead (the oracle sees the PID alive).
- **T5 (G4).** A `SIGTERM`ed run is labeled `scheduler_terminated`, not a crash. A
  `SIGKILL` after a near-cgroup-limit sample is `likely_oom` with the snapshot; a
  `SIGKILL` with no memory evidence stays `unknown_native` (control: no false OOM).
  The cgroup read uses the fail-open ladder; a `"max"`/unlimited value yields a
  low-confidence label.
- **T6 (empirical, manual).** Force a real 2+-rank native abort under torchrun and
  inspect whether the crashing rank's backtrace is readable in the merged stderr tail.
  This gates whether per-rank stderr capture stays deferred (§4.6) or moves into v1.

## 6. Rollout / Graduation

- **Alpha (0.4.0):** `faulthandler` + `crash.json` + crash-aware finalization + cgroup
  OOM attribution (all default-safe and additive). Docs page "What happens when my
  training crashes".
- **Beta:** `--capture-stderr` opt-in, dogfooded on real multi-hour runs.
- **Stable:** crash capture on by default. Graduation: T1-T5 green in CI on Linux plus
  at least one real GPU run exhibiting a forced native crash that finalizes fast with a
  correct crash artifact, and T6 resolved.

## 7. Drawbacks

- More launcher/aggregator surface in the crash path, which must itself never raise.
  Mitigated by best-effort guards, a raw fallback (a formatting bug must never suppress
  the crash attribution), and T1-T5.
- The liveness/oracle machinery (strikes, grace, idempotent latch) adds state to the
  aggregator; kept minimal (a per-rank dict plus a periodic sweep).
- Cgroup reading is Linux-specific; it is a no-op with a low-confidence fallback
  elsewhere.

## 8. Alternatives (rejected options recorded)

- **Native-crash capture.** Chosen: `faulthandler` + launcher `crash.json`. Rejected:
  a custom `SIGSEGV` handler in the executor (a faulting process cannot be trusted to
  run arbitrary Python, and it cannot see its own OOM-kill). Rejected: core dumps (they
  need debug symbols, are multi-GB for a GPU process, and `RLIMIT_CORE` is usually 0 on
  cloud/Slurm); `faulthandler` + stderr-tail is the right altitude.
- **Death detection.** Chosen: EOF as a trigger to consult the launcher `poll()`
  oracle, with a strikes/grace window. Rejected: treating a socket close as death (our
  client reconnects, so this false-positives on a transient blip). Rejected: shortening
  the 300s timeout (hurts legitimate slow-drain multi-node runs and still
  mis-attributes).
- **stderr capture.** Chosen: launcher-level `PIPE` + tee. Rejected for v1: executor-
  level `os.dup2` to per-rank files (in-process fd surgery has a deadlock-on-full-pipe
  failure mode worst exactly at crash time, and a reader thread inside a crashing
  process dies before flushing); deferred to a follow-up for per-rank separation.
- **OOM denominator.** Chosen: cgroup limit. Rejected: host RAM (misses the common
  container-OOM case, a confident false negative).

## 9. Prior art

- CPython `faulthandler` is the standard-library answer to native-fault tracebacks;
  many training frameworks enable it in their workers.
- PyTorch Elastic writes per-rank error files on worker failure; this RFC adapts the
  idea to native/signal deaths in TraceML's own session directory.
- Ray's design informed the finalization model: its raylet treats a lost connection as
  a trigger to consult an authoritative liveness oracle rather than as a death verdict,
  classifies deaths into a small exit-type taxonomy, and reads the cgroup limit for OOM
  attribution. TraceML adapts these to a single-launcher, per-rank, pure-Python
  observer.

## 10. Unresolved questions

- **U1.** Exact `crash.json` / `run_crashed.json` field set and whether to also append
  a human line to a single crash-log location.
- **U2.** The launcher <-> aggregator liveness channel: the two are separate processes,
  so the oracle result (`Popen.poll()`) must reach the aggregator. Options: the
  launcher pushes per-rank exit events, a shared control file the aggregator watches,
  or the launcher owns the crash-finalization step. This shapes the implementation.
- **U3.** Whether to detect host-OOM specifically (signal 9 plus a `dmesg`/`/proc`
  probe) or stay signal-plus-memory-snapshot in v1.
- **U4.** Multi-rank: the owner-node vs per-node responsibility for writing `crash.json`
  when a non-owner rank faults.
- **U5.** Empirically verify (do not reason): does merged multi-rank stderr under
  torchrun shred the crashing rank's NCCL/CUDA backtrace badly enough that per-rank
  capture must move into v1 (§4.6)? Settle with a real 2+-rank native abort.

## 11. Child-issue breakdown

| # | Title | Labels |
|---|---|---|
| 1 | Surface and correctly attribute native crashes (faulthandler + crash.json + crash-aware finalization + cgroup OOM) | `enhancement`, `bug` |
| 2 | Persist the child's stderr tail in summary mode (`--capture-stderr`) | `help wanted`, `enhancement` |
| 3 | Tests + docs: "what happens when my training crashes" | `good first issue`, `documentation` |

Issues tagged `help wanted` or `good first issue` are open for contributors to claim;
the rest are maintainer-owned.

Related existing issues: #24 (verify runtime failures do not stop training),
#141 (harden instrumentation-state).
