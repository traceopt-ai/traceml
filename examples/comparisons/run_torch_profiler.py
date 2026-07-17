"""Config 3 (torch.profiler) and Config 5 combo entry point.

Standard, documented torch.profiler setup (steelmanned, not crippled):
CPU+CUDA activities, a schedule with a small active window (the way the
PyTorch docs recommend, because full-run profiling is impractically heavy),
record_shapes + profile_memory + with_stack, chrome-trace export, and the
key_averages() op table saved to text.

Captures the numbers the study needs: collection wall time, per-step time
inside the active window (for overhead vs bare), chrome-trace size, export
time, and a derived GPU-busy fraction (sum of CUDA kernel self-time / active
wall) to demonstrate what a user must COMPUTE to reach the same "the GPU is
starved" conclusion TraceML prints in one line.

For Config 5 (combo) this same script is run under cProfile by the driver.
"""

import argparse
import json
import os
import time

import torch
from torch.profiler import ProfilerActivity, profile, schedule

import workload_core as wc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--profile", choices=["baseline", "optimized"], default="baseline"
    )
    ap.add_argument("--data-dir", default="imagenette2")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--wait", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--active", type=int, default=15)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--metrics-out", default=None)
    ap.add_argument("--tag", default="torch_profiler")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    total_steps = args.wait + args.warmup + args.active
    trace_path = os.path.join(args.outdir, "trace.json")
    ka_path = os.path.join(args.outdir, "key_averages.txt")

    c = wc.build(args.data_dir, args.profile, args.batch_size)
    print(
        f"[{args.tag}] profile={args.profile} device={c['device']} "
        f"num_workers={c['num_workers']} steps={total_steps} "
        f"(wait={args.wait} warmup={args.warmup} active={args.active})",
        flush=True,
    )

    export_times = []
    cuda_self_us = {"v": 0.0}

    def handler(p):
        t0 = time.perf_counter()
        p.export_chrome_trace(trace_path)
        export_times.append(time.perf_counter() - t0)
        ka = p.key_averages()
        with open(ka_path, "w") as f:
            f.write(ka.table(sort_by="self_cuda_time_total", row_limit=30))
        # Sum of self CUDA kernel time (microseconds) across the active window.
        cuda_self_us["v"] = sum(
            getattr(
                e,
                "self_device_time_total",
                getattr(e, "self_cuda_time_total", 0.0),
            )
            for e in ka
        )

    step_times = []

    sched = schedule(
        wait=args.wait, warmup=args.warmup, active=args.active, repeat=1
    )
    prof_t0 = time.perf_counter()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        on_trace_ready=handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,  # source-attribution stacks multiply trace size;
        # off = the standard "see my timeline / find the gap" setup, fair to
        # torch.profiler. (with_stack is noted in the report as extra cost.)
    ) as prof:

        def on_step(step):
            step_times.append(time.perf_counter())
            prof.step()

        wc.train_loop(c, total_steps, on_step=on_step)
    collect_wall = time.perf_counter() - prof_t0

    # Per-step wall inside the active window (steps wait+warmup .. end).
    active_start = args.wait + args.warmup
    active_ts = step_times[active_start - 1 :]
    active_per_step_ms = None
    if len(active_ts) >= 2:
        active_per_step_ms = (
            (active_ts[-1] - active_ts[0]) / (len(active_ts) - 1) * 1000.0
        )
    active_wall_s = (
        (active_ts[-1] - active_ts[0]) if len(active_ts) >= 2 else None
    )
    cuda_busy_s = cuda_self_us["v"] / 1e6
    gpu_busy_frac = (
        cuda_busy_s / active_wall_s
        if active_wall_s and active_wall_s > 0
        else None
    )

    trace_bytes = (
        os.path.getsize(trace_path) if os.path.exists(trace_path) else 0
    )
    metrics = {
        "config": args.tag,
        "total_steps": total_steps,
        "active_steps": args.active,
        "collect_wall_s": collect_wall,
        "export_s": export_times[0] if export_times else None,
        "active_per_step_ms": active_per_step_ms,
        "trace_bytes": trace_bytes,
        "cuda_self_busy_s": cuda_busy_s,
        "active_wall_s": active_wall_s,
        "derived_gpu_busy_frac": gpu_busy_frac,
    }
    print(f"[{args.tag}] {json.dumps(metrics)}", flush=True)
    if args.metrics_out:
        with open(args.metrics_out, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
