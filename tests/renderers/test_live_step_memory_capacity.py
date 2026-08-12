"""Live step-memory diagnosis must see device capacity (issue #354).

`HIGH_PRESSURE` and `IMBALANCE` are both capacity-gated: without
`gpu_total_bytes` the engine has no pressure fraction, so neither rule can
fire and a near-capacity run stays silent until the final summary. These
tests drive both live surfaces end to end from SQLite, so dropping capacity
on either one fails here instead of shipping.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from rich.console import Console
from tests.sqlite_fixtures import (
    init_summary_schema,
    insert_process_sample,
    insert_step_memory_sample,
    sqlite_database,
)
from tests.step_time.factories import live_result_from_window

from traceml_ai.renderers.model_diagnostics.renderer import (
    ModelDiagnosticsRenderer,
)
from traceml_ai.renderers.step_memory.renderer import StepMemoryRenderer
from traceml_ai.step_time.model import StepTimeWindow
from traceml_ai.step_time.pipeline import LiveStepTimeResult

GIB = 1024.0**3

# LIVE_STEP_MEMORY_POLICY.thresholds.min_steps_for_diag is 50.
STEPS = 60


def _write_run(
    path: str,
    *,
    capacity_bytes: Optional[float],
    reserved_by_rank: Mapping[int, float],
) -> None:
    """Write one flat-memory run: N ranks, `STEPS` aligned completed steps."""
    world_size = len(reserved_by_rank)
    with sqlite_database(path, init_summary_schema) as conn:
        for rank in sorted(reserved_by_rank):
            insert_process_sample(
                conn,
                row_id=1000 + rank,
                rank=rank,
                ts=float(rank),
                gpu_available=True,
                gpu_count=world_size,
                world_size=world_size,
                local_world_size=world_size,
                gpu_mem_total_bytes=capacity_bytes,
            )
        row_id = 0
        for rank, reserved in sorted(reserved_by_rank.items()):
            for step in range(STEPS):
                row_id += 1
                insert_step_memory_sample(
                    conn,
                    row_id=row_id,
                    rank=rank,
                    step=step,
                    alloc=reserved * 0.9,
                    reserved=reserved,
                    world_size=world_size,
                    local_world_size=world_size,
                )


def _cold_step_time() -> LiveStepTimeResult:
    """Step Time is not under test here; supply an empty live window."""
    return live_result_from_window(StepTimeWindow(), freshness="cold")


def _dashboard_step_memory_item(db_path: str) -> Dict[str, Any]:
    """Return the Step Memory item from the live diagnostics rail."""
    payload = ModelDiagnosticsRenderer(db_path).get_dashboard_renderable(
        _cold_step_time()
    )
    return next(
        item for item in payload["items"] if item["source"] == "step_memory"
    )


def _terminal_panel_text(db_path: str) -> str:
    """Render the live terminal Step Memory panel to plain text."""
    console = Console(
        force_terminal=True,
        color_system=None,
        width=140,
        record=True,
    )
    console.print(StepMemoryRenderer(db_path).get_panel_renderable())
    return console.export_text()


def test_dashboard_rail_reports_high_pressure_live(tmp_path) -> None:
    # Reachability, dashboard path. 15 GiB reserved of a 16 GiB device is
    # 93.8% -- past pressure_warn_fraction (0.92). This assertion is what
    # the hardcoded `gpu_total_bytes=None` made unreachable.
    db_path = str(tmp_path / "pressure.db")
    _write_run(
        db_path,
        capacity_bytes=16.0 * GIB,
        reserved_by_rank={0: 15.0 * GIB},
    )

    item = _dashboard_step_memory_item(db_path)

    assert item["kind"] == "HIGH_PRESSURE"
    assert item["status"] == "HIGH PRESSURE"
    assert item["severity"] == "warn"
    assert item["evidence"]["pressure"] == "93.8%"


def test_terminal_panel_reports_high_pressure_live(tmp_path) -> None:
    # Reachability, terminal path. The CLI panel is a separate surface with
    # its own diagnosis call; both must claim pressure or the two live
    # surfaces disagree about the same run.
    db_path = str(tmp_path / "pressure.db")
    _write_run(
        db_path,
        capacity_bytes=16.0 * GIB,
        reserved_by_rank={0: 15.0 * GIB},
    )

    text = _terminal_panel_text(db_path)

    assert "HIGH PRESSURE" in text
    assert "near device capacity" in text


def test_dashboard_rail_reports_imbalance_live(tmp_path) -> None:
    # IMBALANCE is capacity-gated too (`classify_imbalance_severity` returns
    # None without a pressure fraction), so it needs the same coverage.
    # Peaks 5/8 GiB of 16 GiB: skew 23% past imbalance_skew_warn (0.20) at
    # 50% pressure, and well below the 92% that would fire HIGH_PRESSURE.
    db_path = str(tmp_path / "imbalance.db")
    _write_run(
        db_path,
        capacity_bytes=16.0 * GIB,
        reserved_by_rank={0: 5.0 * GIB, 1: 8.0 * GIB},
    )

    item = _dashboard_step_memory_item(db_path)

    assert item["kind"] == "IMBALANCE"
    assert item["worst_rank"] == 1

    # Same run, terminal surface.
    assert "IMBALANCE" in _terminal_panel_text(db_path)


def test_absent_capacity_makes_no_pressure_claim(tmp_path) -> None:
    # Negative case: capacity is best-effort. A run whose process telemetry
    # never reported a device total must degrade to today's behavior on both
    # surfaces -- no crash, and no fabricated pressure claim.
    db_path = str(tmp_path / "no-capacity.db")
    _write_run(
        db_path,
        capacity_bytes=None,
        reserved_by_rank={0: 15.0 * GIB},
    )

    item = _dashboard_step_memory_item(db_path)

    assert item["kind"] == "BALANCED"
    assert item["evidence"]["pressure"] == "n/a"

    text = _terminal_panel_text(db_path)
    assert "HIGH PRESSURE" not in text
    assert "BALANCED" in text


def test_well_under_capacity_stays_quiet(tmp_path) -> None:
    # Negative case: capacity is present and the rule must stay quiet. One
    # rank at 2 GiB of 16 GiB is 12.5% pressure with no cross-rank skew.
    db_path = str(tmp_path / "roomy.db")
    _write_run(
        db_path,
        capacity_bytes=16.0 * GIB,
        reserved_by_rank={0: 2.0 * GIB},
    )

    item = _dashboard_step_memory_item(db_path)

    assert item["kind"] == "BALANCED"
    assert item["evidence"]["pressure"] == "12.5%"
    assert "HIGH PRESSURE" not in _terminal_panel_text(db_path)


def test_live_and_summary_share_one_capacity_reader() -> None:
    # A second, independently written capacity query is how the two paths
    # would drift apart. Pin them to one function object.
    from traceml_ai.renderers.step_memory import common as live_common
    from traceml_ai.reporting.sections.step_memory import loader as summary

    assert summary.load_gpu_total_bytes is live_common.load_gpu_total_bytes
