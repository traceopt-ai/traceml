import re

from traceml_ai.reporting.html.svg import memory_bars, phase_bar


def _widths(html: str):
    return [float(w) for w in re.findall(r"width:([\d.]+)%", html)]


def test_phase_bar_emits_svg_rects_for_present_phases(make_section) -> None:
    section = make_section(
        metric_names=[
            "total_step_ms",
            "dataloader_ms",
            "input_wait_ms",
            "forward_ms",
            "backward_ms",
        ],
        average={
            "dataloader_ms": 20.0,
            "input_wait_ms": 20.0,
            "forward_ms": 30.0,
            "backward_ms": 50.0,
            "total_step_ms": 100.0,
        },
    )
    out = phase_bar(section)
    assert "<svg" in out
    assert out.count("<rect") == 3  # input wait + forward + backward


def test_phase_bar_falls_back_to_dataloader_ms_for_pre_1_6(
    make_section,
) -> None:
    # Schema < 1.6 reports carry dataloader_ms but no input_wait_ms; the
    # input phase must still render rather than silently dropping out.
    section = make_section(
        metric_names=["total_step_ms", "dataloader_ms", "forward_ms"],
        average={
            "dataloader_ms": 60.0,
            "forward_ms": 40.0,
            "total_step_ms": 100.0,
        },
    )
    out = phase_bar(section)
    assert "input wait" in out
    assert out.count("<rect") == 2  # input (via dataloader_ms) + forward


def test_phase_bar_empty_when_no_timing(make_section) -> None:
    section = make_section(
        metric_names=["total_step_ms"], average={"total_step_ms": 0.0}
    )
    assert phase_bar(section) == ""


def test_memory_bars_scale_to_process_capacity(make_section) -> None:
    step_memory = make_section(
        metric_names=["peak_reserved_bytes"],
        average={"peak_reserved_bytes": 2.3e9},
        rows={
            "0": {
                "identity": {"global_rank": 0, "hostname": "h0"},
                "metrics": {"peak_reserved_bytes": 2.3e9},
            }
        },
    )
    process = make_section(
        metric_names=["gpu_mem_reserved_bytes", "gpu_mem_headroom_bytes"],
        average={},
        rows={
            "0": {
                "identity": {"global_rank": 0, "hostname": "h0"},
                "metrics": {
                    "gpu_mem_reserved_bytes": 2.3e9,
                    "gpu_mem_headroom_bytes": 21.7e9,
                },
            }
        },
    )
    out = memory_bars(step_memory, process)
    assert "GPU capacity" in out
    # 2.3 of 24 GB total -> well under half width, not a saturated-looking bar.
    assert max(_widths(out)) < 50


def test_memory_bars_fall_back_to_worst_rank_with_caption(
    make_section,
) -> None:
    step_memory = make_section(
        metric_names=["peak_reserved_bytes"],
        average={"peak_reserved_bytes": 2.3e9},
        rows={
            "0": {
                "identity": {"global_rank": 0, "hostname": "h0"},
                "metrics": {"peak_reserved_bytes": 2.3e9},
            }
        },
    )
    out = memory_bars(step_memory, process_section={})
    assert "relative to worst rank" in out
    assert max(_widths(out)) == 100.0


def test_memory_bars_empty_when_no_rows(make_section) -> None:
    empty = make_section(
        metric_names=["peak_reserved_bytes"], average={}, rows={}
    )
    assert memory_bars(empty, {}) == ""
