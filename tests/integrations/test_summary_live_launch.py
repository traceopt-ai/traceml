"""In-training ``final_summary()`` calls through the real launcher."""

import json
import sys

import pytest

from tests.integrations.launcher_utils import _run

pytest.importorskip("torch")

_STEPS = 40
_RUN_NAME = "summary-live"

_WORKLOAD = """
import json
import os
import time

import torch
import torch.distributed as dist
from torch import nn

import traceml_ai as traceml

STEPS = NUM_STEPS
dist.init_process_group(backend="gloo")
rank = dist.get_rank()
traceml.init(mode="auto")
torch.manual_seed(0)
model = nn.parallel.DistributedDataParallel(
    nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(8, 8)
y = torch.randint(0, 4, (8,))


def run(steps):
    for _ in range(steps):
        with traceml.trace_step(model.module):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()


run(STEPS)
dist.barrier()
if rank == 0:
    first = traceml.final_summary()
run(STEPS)
dist.barrier()
# Rank 1 ships its last steps on its next sampler tick; rank 0 flushes its
# own when it asks.
time.sleep(1.5 * float(os.environ["TRACEML_INTERVAL"]))
if rank == 0:
    second = traceml.final_summary()
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        json.dump({"first": first, "second": second}, handle)
dist.barrier()
dist.destroy_process_group()
"""


def _steps_analyzed(summary):
    window = (
        ((summary or {}).get("step_time") or {}).get("global") or {}
    ).get("window") or {}
    return window.get("steps_analyzed") or 0


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="The two-rank torchrun smoke path is not verified on Windows.",
)
def test_in_training_summary_is_regenerated_until_ranks_finish(tmp_path):
    out_path = tmp_path / "in_training_summaries.json"
    script = tmp_path / "train.py"
    script.write_text(
        _WORKLOAD.replace("NUM_STEPS", str(_STEPS)).replace(
            "OUT_PATH", repr(str(out_path))
        )
    )

    result = _run(tmp_path, script, run_name=_RUN_NAME, nproc_per_node=2)
    assert result.returncode == 0, result.stdout + result.stderr

    calls = json.loads(out_path.read_text())
    first = _steps_analyzed(calls["first"])
    second = _steps_analyzed(calls["second"])
    session_root = tmp_path / "logs" / _RUN_NAME
    shutdown = _steps_analyzed(
        json.loads((session_root / "final_summary.json").read_text())
    )

    assert second > first
    assert second == shutdown == 2 * _STEPS
