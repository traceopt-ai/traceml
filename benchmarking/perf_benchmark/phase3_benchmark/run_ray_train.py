"""Dependency-check stub for a future Ray Train launcher.

It does not submit a Ray job. The torchrun launcher is the default publishable
path until this route is implemented for a confirmed target cluster.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from common.io_utils import read_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config.resolve())

    try:
        import ray  # noqa: F401
        import ray.train  # noqa: F401
        from traceml_ai.integrations.ray import (
            TraceMLTorchTrainer,
        )  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Ray Train is not available. Install `traceml-ai[ray]` and run "
            "this on a configured Ray cluster, or use run_benchmark.py."
        ) from exc

    if args.dry_run:
        print("[phase3-ray] Ray imports succeeded.")
        print(
            f"[phase3-ray] config workloads: {len(config.get('workloads', []))}"
        )
        return 0

    raise SystemExit(
        "Ray Phase 3 execution is intentionally gated until the target "
        "Ray/K8s cluster shape is known. Use torchrun for publishable runs now; "
        "use --dry-run to validate Ray dependencies."
    )


if __name__ == "__main__":
    raise SystemExit(main())
