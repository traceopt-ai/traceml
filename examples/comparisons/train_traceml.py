"""Config 2 (TraceML). Launched via `traceml run --mode summary ...`.

Same workload_core as every other config; the only additions are the two
TraceML lines from the notebook: init() once, and trace_step(model) wrapped
around each step. Run through the launcher so the aggregator + end-of-run
final_summary are produced exactly as a real user gets them.
"""

import argparse

import traceml_ai as traceml

import workload_core as wc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--profile", choices=["baseline", "optimized"], default="baseline"
    )
    ap.add_argument("--data-dir", default="imagenette2")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=300)
    args = ap.parse_args()

    c = wc.build(args.data_dir, args.profile, args.batch_size)
    print(
        f"[traceml] profile={args.profile} device={c['device']} "
        f"num_workers={c['num_workers']} batch={args.batch_size} "
        f"max_steps={args.max_steps}",
        flush=True,
    )

    traceml.init(mode="auto")  # TraceML line 1
    model = c["model"]
    wc.train_loop(
        c,
        args.max_steps,
        step_ctx=lambda: traceml.trace_step(model),  # TraceML line 2
    )


if __name__ == "__main__":
    main()
