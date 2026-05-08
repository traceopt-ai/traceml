from __future__ import annotations

import pytest

from traceml.diagnostics.process.api import build_process_diagnosis_result
from traceml.diagnostics.process.context import build_process_summary_signals
from traceml.diagnostics.process.rules import (
    GPUMemoryReservedOverhangRule,
    HighProcessCPURule,
    HighProcessGPUMemoryRule,
    HighProcessRSSRule,
    RankGPUMemoryImbalanceRule,
    VeryHighProcessGPUMemoryRule,
)


def _per_rank(
    *,
    rss_peak: float = 200.0,
    used_peak: float = 200.0,
    reserved_peak: float = 240.0,
    total: float = 1000.0,
) -> dict[int, dict[str, float]]:
    return {
        0: {
            "ram_peak_bytes": rss_peak,
            "gpu_mem_used_peak_bytes": used_peak,
            "gpu_mem_reserved_peak_bytes": reserved_peak,
            "gpu_mem_total_bytes": total,
        }
    }


def _rank(
    *,
    rss_peak: float = 200.0,
    used_peak: float = 200.0,
    reserved_peak: float = 240.0,
    total: float = 1000.0,
    overhang: float | None = None,
) -> dict[str, float]:
    out = {
        "ram_peak_bytes": rss_peak,
        "gpu_mem_used_peak_bytes": used_peak,
        "gpu_mem_reserved_peak_bytes": reserved_peak,
        "gpu_mem_total_bytes": total,
    }
    if overhang is not None:
        out["gpu_mem_reserved_overhang_ratio"] = overhang
    return out


def _signals(**overrides):
    values = dict(
        duration_s=10.0,
        samples=10,
        distinct_ranks=1,
        distinct_pids=1,
        cpu_avg_percent=120.0,
        cpu_peak_percent=200.0,
        cpu_logical_core_count=8,
        ram_avg_bytes=100.0,
        ram_peak_bytes=200.0,
        ram_total_bytes=1000.0,
        gpu_available=True,
        gpu_count=1,
        gpu_device_index=0,
        gpu_mem_used_avg_bytes=100.0,
        gpu_mem_used_peak_bytes=200.0,
        gpu_mem_reserved_avg_bytes=120.0,
        gpu_mem_reserved_peak_bytes=240.0,
        gpu_mem_total_bytes=1000.0,
        per_rank=_per_rank(),
    )
    values.update(overrides)
    return build_process_summary_signals(**values)


@pytest.mark.parametrize(
    ("rule", "overrides", "expected_kind"),
    [
        (
            VeryHighProcessGPUMemoryRule(),
            {"gpu_mem_reserved_peak_bytes": 930.0},
            "VERY_HIGH_PROCESS_GPU_MEMORY",
        ),
        (
            HighProcessGPUMemoryRule(),
            {"gpu_mem_reserved_peak_bytes": 850.0},
            "HIGH_PROCESS_GPU_MEMORY",
        ),
        (
            GPUMemoryReservedOverhangRule(),
            {
                "gpu_mem_used_peak_bytes": 400.0,
                "gpu_mem_reserved_peak_bytes": 700.0,
                "per_rank": _per_rank(used_peak=400.0, reserved_peak=700.0),
            },
            "GPU_MEMORY_RESERVED_OVERHANG",
        ),
        (
            RankGPUMemoryImbalanceRule(),
            {
                "per_rank": {
                    0: {
                        "gpu_mem_used_peak_bytes": 900.0,
                        "gpu_mem_reserved_peak_bytes": 900.0,
                        "gpu_mem_total_bytes": 1000.0,
                    },
                    1: {
                        "gpu_mem_used_peak_bytes": 400.0,
                        "gpu_mem_reserved_peak_bytes": 400.0,
                        "gpu_mem_total_bytes": 1000.0,
                    },
                }
            },
            "RANK_GPU_MEMORY_IMBALANCE",
        ),
        (
            HighProcessRSSRule(),
            {"ram_peak_bytes": 850.0},
            "HIGH_PROCESS_RSS",
        ),
        (
            HighProcessCPURule(),
            {"cpu_avg_percent": 700.0},
            "HIGH_PROCESS_CPU",
        ),
    ],
)
def test_process_rules_trigger_one_condition(
    rule,
    overrides,
    expected_kind,
) -> None:
    issue = rule.evaluate(_signals(**overrides))

    assert issue is not None
    assert issue.kind == expected_kind


@pytest.mark.parametrize(
    ("rule", "overrides"),
    [
        (
            VeryHighProcessGPUMemoryRule(),
            {"gpu_mem_reserved_peak_bytes": 850.0},
        ),
        (
            HighProcessGPUMemoryRule(),
            {"gpu_mem_reserved_peak_bytes": 500.0},
        ),
        (
            GPUMemoryReservedOverhangRule(),
            {
                "gpu_mem_used_peak_bytes": 400.0,
                "gpu_mem_reserved_peak_bytes": 450.0,
            },
        ),
        (RankGPUMemoryImbalanceRule(), {}),
        (HighProcessRSSRule(), {"ram_peak_bytes": 500.0}),
        (HighProcessCPURule(), {"cpu_avg_percent": 120.0}),
    ],
)
def test_process_rules_do_not_trigger_normal_condition(
    rule,
    overrides,
) -> None:
    assert rule.evaluate(_signals(**overrides)) is None


@pytest.mark.parametrize(
    ("overrides", "expected_primary"),
    [
        (
            {"gpu_mem_reserved_peak_bytes": 930.0},
            "VERY_HIGH_PROCESS_GPU_MEMORY",
        ),
        (
            {"gpu_mem_reserved_peak_bytes": 850.0},
            "HIGH_PROCESS_GPU_MEMORY",
        ),
        (
            {
                "gpu_mem_used_peak_bytes": 400.0,
                "gpu_mem_reserved_peak_bytes": 700.0,
                "per_rank": _per_rank(used_peak=400.0, reserved_peak=700.0),
            },
            "GPU_MEMORY_RESERVED_OVERHANG",
        ),
        (
            {
                "per_rank": {
                    0: {
                        "gpu_mem_used_peak_bytes": 900.0,
                        "gpu_mem_reserved_peak_bytes": 900.0,
                        "gpu_mem_total_bytes": 1000.0,
                    },
                    1: {
                        "gpu_mem_used_peak_bytes": 400.0,
                        "gpu_mem_reserved_peak_bytes": 400.0,
                        "gpu_mem_total_bytes": 1000.0,
                    },
                }
            },
            "RANK_GPU_MEMORY_IMBALANCE",
        ),
        ({"ram_peak_bytes": 850.0}, "HIGH_PROCESS_RSS"),
        ({"cpu_avg_percent": 700.0}, "HIGH_PROCESS_CPU"),
    ],
)
def test_process_primary_diagnosis_for_each_issue(
    overrides,
    expected_primary,
) -> None:
    result = _diagnose(**overrides)

    assert result.primary.kind == expected_primary


def test_process_primary_priority_when_everything_triggers() -> None:
    result = _diagnose(
        cpu_avg_percent=700.0,
        ram_peak_bytes=850.0,
        gpu_mem_used_peak_bytes=400.0,
        gpu_mem_reserved_peak_bytes=1000.0,
        per_rank={
            0: {
                "gpu_mem_used_peak_bytes": 900.0,
                "gpu_mem_reserved_peak_bytes": 1000.0,
                "gpu_mem_total_bytes": 1000.0,
                "gpu_mem_reserved_overhang_ratio": 1000.0 / 900.0,
            },
            1: {
                "gpu_mem_used_peak_bytes": 400.0,
                "gpu_mem_reserved_peak_bytes": 700.0,
                "gpu_mem_total_bytes": 1000.0,
                "gpu_mem_reserved_overhang_ratio": 700.0 / 400.0,
            },
        },
    )

    assert result.primary.kind == "VERY_HIGH_PROCESS_GPU_MEMORY"
    assert [issue.kind for issue in result.issues] == [
        "VERY_HIGH_PROCESS_GPU_MEMORY",
        "GPU_MEMORY_RESERVED_OVERHANG",
        "RANK_GPU_MEMORY_IMBALANCE",
        "HIGH_PROCESS_RSS",
        "HIGH_PROCESS_CPU",
    ]


def test_reserved_overhang_uses_rank_local_peak_ratio() -> None:
    result = _diagnose(
        gpu_mem_used_peak_bytes=1000.0,
        gpu_mem_reserved_peak_bytes=1200.0,
        gpu_mem_total_bytes=2000.0,
        per_rank={
            0: _rank(used_peak=1000.0, reserved_peak=1200.0, overhang=1.2),
            1: _rank(used_peak=100.0, reserved_peak=180.0, overhang=1.8),
        },
    )

    issue = result.issues[0]
    assert issue.kind == "GPU_MEMORY_RESERVED_OVERHANG"
    assert issue.ranks == (1,)
    assert issue.evidence["gpu_mem_reserved_overhang_ratio"] == 1.8


def test_process_cpu_only_normal_does_not_emit_gpu_context_status() -> None:
    result = _diagnose(
        gpu_available=False,
        gpu_count=0,
        gpu_device_index=None,
        gpu_mem_used_avg_bytes=None,
        gpu_mem_used_peak_bytes=None,
        gpu_mem_reserved_avg_bytes=None,
        gpu_mem_reserved_peak_bytes=None,
        gpu_mem_total_bytes=None,
        per_rank={0: {"ram_peak_bytes": 200.0}},
    )

    assert result.primary.kind == "NORMAL"
    assert result.primary.status == "NORMAL"
    assert "GPU" not in result.primary.reason
    assert result.issues == ()


def test_process_no_data_is_primary_without_running_rules() -> None:
    result = _diagnose(process_samples=0, per_rank={})

    assert result.primary.kind == "NO_DATA"
    assert result.issues == ()


def _diagnose(**overrides):
    values = dict(
        duration_s=10.0,
        process_samples=10,
        distinct_ranks=1,
        distinct_pids=1,
        cpu_avg_percent=120.0,
        cpu_peak_percent=200.0,
        cpu_logical_core_count=8,
        ram_avg_bytes=100.0,
        ram_peak_bytes=200.0,
        ram_total_bytes=1000.0,
        gpu_available=True,
        gpu_count=1,
        gpu_device_index=0,
        gpu_mem_used_avg_bytes=100.0,
        gpu_mem_used_peak_bytes=200.0,
        gpu_mem_reserved_avg_bytes=120.0,
        gpu_mem_reserved_peak_bytes=240.0,
        gpu_mem_total_bytes=1000.0,
        per_rank=_per_rank(),
    )
    values.update(overrides)
    return build_process_diagnosis_result(**values)
