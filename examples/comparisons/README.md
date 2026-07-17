# Comparing TraceML with torch.profiler and cProfile

A small, reproducible harness that runs the **same** PyTorch training workload
under three tools plus a bare baseline, on one input-bound run, and produces an
apples-to-apples table and figures. Use it to see what each tool costs and what
each tool tells you on the identical run.

## What it runs

One workload (`workload_core.py`: ResNet-18 on Imagenette with `num_workers=0`,
so the run is input-bound and the GPU is starved), measured five ways:

1. **bare** - no instrumentation; the wall-clock reference every overhead % is
   computed against.
2. **traceml** - `traceml run` (always-on), through the real launcher, produces
   `final_summary.json`.
3. **torch.profiler** - a standard windowed profile (small active window, chrome
   trace + `key_averages`).
4. **cProfile** - whole-run Python function profiling.
5. **combo** - torch.profiler run under cProfile (the combined workflow experts
   commonly use).

An independent `nvidia-smi` GPU-util sample runs at 1 Hz during every config, so
no tool grades its own homework.

## Files

| File | Role |
|---|---|
| `workload_core.py` | the shared training core (no tool-specific code) |
| `run_bare.py` | config 1 (bare) and the cProfile entry point |
| `train_traceml.py` | config 2, launched via `traceml run` |
| `run_torch_profiler.py` | configs 3 + 5 (windowed profile) |
| `study_driver.sh` | runs all 5 configs, times each, records artifact bytes |
| `parse_results.py` | raw artifacts -> `results.json` + a markdown table |
| `make_figures.py` | `results.json` -> `figures/*.png,*.svg` |

## Run it

Prereqs: a single-GPU box with PyTorch and `traceml-ai` installed, and Imagenette
(full-res) at `$DATA_DIR` (a folder with `train/` and `val/`).

```bash
export DATA_DIR=/path/to/imagenette2
export OUT=$PWD/study_out
bash study_driver.sh
python3 parse_results.py --in "$OUT" --out-json results.json --out-md results.md
python3 make_figures.py            # reads ./results.json, writes ./figures/
```

`study_driver.sh` is nohup-safe: it writes `study_out/STUDY_DONE` at the end, so
launch it with `nohup ... &` and poll that sentinel rather than holding an
interactive shell for the whole run.

## Shared fixture

To keep results comparable across people, pin the workload to the shared fixture
named in `FIXTURE.md` instead of each person choosing their own model, batch
size, or topology.

## Notes that will save you time

- torch.profiler `with_stack=True` inflates the trace with Python source stacks.
  This harness uses `with_stack=False` for a fair "find the bottleneck" setup and
  notes the extra cost separately.
- Do not compute a GPU-busy fraction from the trace and trust it: summing
  per-kernel CUDA self-time can exceed wall time (kernels overlap across streams).
  Use `nvidia-smi` for utilization ground truth.
- cProfile only sees single-thread work: with `num_workers=0` it pinpoints the
  main-process decode cost; with workers on, that work moves to child processes
  and cProfile goes blind. State this in any comparison.
- Separate TraceML's fixed launcher startup (a one-time aggregator spin-up) from
  per-step overhead: the fair per-step number comes from `final_summary.json`'s
  `duration_s` versus the bare in-process loop time, not the driver wall.
