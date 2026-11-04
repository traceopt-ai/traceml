"""Live JSON Lines export through the real launcher with two CPU ranks."""

import json
import sys

import pytest

from tests.integrations.launcher_utils import _run

pytest.importorskip("torch")

_STEPS = 4
_RUN_NAME = "export-launch"

_WORKLOAD = """
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import traceml_ai as traceml

STEPS = NUM_STEPS
torch.manual_seed(0)
traceml.init(mode="auto")
model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loader = DataLoader(
    TensorDataset(torch.randn(STEPS * 4, 8), torch.randn(STEPS * 4, 1)),
    batch_size=4,
)
for x, y in loader:
    with traceml.trace_step(model):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
"""


def _script(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(_WORKLOAD.replace("NUM_STEPS", str(_STEPS)))
    return script


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="The two-rank torchrun smoke path is not verified on Windows.",
)
def test_two_rank_run_exports_step_time_jsonl(tmp_path):
    result = _run(
        tmp_path,
        _script(tmp_path),
        run_name=_RUN_NAME,
        nproc_per_node=2,
        extra_env={"TRACEML_EXPORTER": "jsonl"},
    )
    assert result.returncode == 0, result.stdout + result.stderr

    session_root = tmp_path / "logs" / _RUN_NAME
    path = session_root / "aggregator" / "export" / "step_time.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    *data, closing = lines

    assert data
    for line in data:
        assert line["stream"] == "step_time"
        assert line["schema_version"] == 1
        assert "closing" not in line
    assert closing == {
        "stream": "step_time",
        "closing": True,
        "schema_version": 1,
        "written_records": len(data),
        "dropped_batches": 0,
    }

    keys = [(line["global_rank"], line["step"]) for line in data]
    assert len(keys) == len(set(keys))
    steps_by_rank = {}
    for rank, step in keys:
        steps_by_rank.setdefault(rank, set()).add(step)
    assert set(steps_by_rank) == {0, 1}
    first = min(steps_by_rank[0])
    for steps in steps_by_rank.values():
        assert steps == set(range(first, first + _STEPS))

    for line in data:
        assert "hostname" not in line
        assert line["rank"] == line["global_rank"]
        assert line["world_size"] == 2
        assert line["local_world_size"] == 2
        assert line["node_rank"] == 0
        assert isinstance(line["observed_timestamp_unix_ns"], int)
        assert isinstance(line["timestamp_unix_ns"], int)
        events = line["events"]
        for devices in events.values():
            for value in devices.values():
                assert value["gpu_ms"] is None
                assert isinstance(value["cpu_ms"], float)
                assert value["cpu_ms"] >= 0
        for name in ("forward_time", "backward_time"):
            devices = events[f"_traceml_internal:{name}"]
            assert sum(value["cpu_ms"] for value in devices.values()) > 0

    summary = json.loads((session_root / "final_summary.json").read_text())
    metadata = summary["step_time"]["metadata"]
    assert metadata["global_ranks_seen"] == 2
    assert metadata["global_ranks_used"] == 2
    assert "export" not in summary
    assert not (session_root / "finalization_warning.json").exists()
