"""Config 1 (BARE) and cProfile entry point.

Runs the workload with zero instrumentation. This is the wall-clock
reference every overhead number is computed against. It is ALSO the script
cProfile wraps (Config 4): `python -m cProfile -o out.prof run_bare.py ...`,
because cProfile instruments whatever entry point it is given.
"""

import argparse
import json
import time

import torch

import workload_core as wc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--profile", choices=["baseline", "optimized"], default="baseline"
    )
    ap.add_argument("--data-dir", default="imagenette2")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--metrics-out", default=None)
    args = ap.parse_args()

    c = wc.build(args.data_dir, args.profile, args.batch_size)
    print(
        f"[bare] profile={args.profile} device={c['device']} "
        f"num_workers={c['num_workers']} batch={args.batch_size} "
        f"max_steps={args.max_steps}",
        flush=True,
    )
    t0 = time.perf_counter()
    steps = wc.train_loop(c, args.max_steps)
    wall = time.perf_counter() - t0

    per_step_ms = wall / steps * 1000.0
    print(
        f"[bare] steps={steps} wall_s={wall:.3f} "
        f"per_step_ms={per_step_ms:.2f}",
        flush=True,
    )
    if args.metrics_out:
        with open(args.metrics_out, "w") as f:
            json.dump(
                {
                    "config": "bare",
                    "steps": steps,
                    "wall_s": wall,
                    "per_step_ms": per_step_ms,
                    "cuda": torch.cuda.is_available(),
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
