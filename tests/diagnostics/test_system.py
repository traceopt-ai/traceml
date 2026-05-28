# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from traceml_ai.diagnostics.system.api import diagnose_system
from traceml_ai.diagnostics.system.context import (
    SystemDiagnosisInput,
    SystemGpuDiagnosisInput,
    SystemNodeDiagnosisInput,
    build_system_summary_signals,
)
from traceml_ai.diagnostics.system.rules import (
    HighCPURule,
    HighGPUMemoryRule,
    HighGPUPowerRule,
    HighGPUTemperatureRule,
    HighHostMemoryRule,
    LowGPUUtilizationRule,
    VeryHighGPUMemoryRule,
)


def _per_gpu(
    *,
    util: float = 70.0,
    mem_peak: float = 200.0,
    mem_total: float = 1000.0,
    temp_peak: float | None = None,
    power_avg: float | None = None,
    power_limit: float | None = None,
) -> dict[int, SystemGpuDiagnosisInput]:
    return {
        0: SystemGpuDiagnosisInput(
            util_avg_percent=util,
            mem_peak_bytes=mem_peak,
            mem_total_bytes=mem_total,
            temp_peak_c=temp_peak,
            power_avg_w=power_avg,
            power_limit_w=power_limit,
        )
    }


def _node_input(**overrides) -> SystemNodeDiagnosisInput:
    values = dict(
        node_label="0",
        node_rank=0,
        duration_s=10.0,
        samples=10,
        cpu_avg_percent=20.0,
        cpu_peak_percent=40.0,
        ram_avg_bytes=100.0,
        ram_peak_bytes=200.0,
        ram_total_bytes=1000.0,
        gpu_available=True,
        gpu_count=1,
        gpu_util_avg_percent=70.0,
        gpu_util_peak_percent=90.0,
        gpu_mem_avg_bytes=100.0,
        gpu_mem_peak_bytes=200.0,
        gpu_temp_avg_c=None,
        gpu_temp_peak_c=None,
        gpu_power_avg_w=None,
        gpu_power_peak_w=None,
        per_gpu=_per_gpu(),
    )
    values.update(overrides)
    return SystemNodeDiagnosisInput(**values)


def _input(**overrides) -> SystemDiagnosisInput:
    node = _node_input(**overrides)
    return SystemDiagnosisInput(
        duration_s=node.duration_s,
        samples=node.samples,
        nodes_seen=1,
        cpu_avg_percent=node.cpu_avg_percent,
        cpu_peak_percent=node.cpu_peak_percent,
        ram_avg_bytes=node.ram_avg_bytes,
        ram_peak_bytes=node.ram_peak_bytes,
        ram_total_bytes=node.ram_total_bytes,
        gpu_available=node.gpu_available,
        gpu_count=node.gpu_count,
        gpu_util_avg_percent=node.gpu_util_avg_percent,
        gpu_util_peak_percent=node.gpu_util_peak_percent,
        gpu_mem_avg_bytes=node.gpu_mem_avg_bytes,
        gpu_mem_peak_bytes=node.gpu_mem_peak_bytes,
        gpu_temp_avg_c=node.gpu_temp_avg_c,
        gpu_temp_peak_c=node.gpu_temp_peak_c,
        gpu_power_avg_w=node.gpu_power_avg_w,
        gpu_power_peak_w=node.gpu_power_peak_w,
        per_node={"0": node},
    )


def _signals(**overrides):
    return build_system_summary_signals(_node_input(**overrides))


@pytest.mark.parametrize(
    ("rule", "overrides", "expected_kind"),
    [
        (
            LowGPUUtilizationRule(),
            {
                "gpu_util_avg_percent": 10.0,
                "per_gpu": _per_gpu(util=10.0),
            },
            "LOW_GPU_UTILIZATION",
        ),
        (
            HighCPURule(),
            {"cpu_avg_percent": 95.0},
            "HIGH_CPU",
        ),
        (
            HighHostMemoryRule(),
            {"ram_peak_bytes": 930.0},
            "HIGH_HOST_MEMORY",
        ),
        (
            HighGPUMemoryRule(),
            {"per_gpu": _per_gpu(mem_peak=850.0)},
            "HIGH_GPU_MEMORY",
        ),
        (
            VeryHighGPUMemoryRule(),
            {"per_gpu": _per_gpu(mem_peak=930.0)},
            "VERY_HIGH_GPU_MEMORY",
        ),
        (
            HighGPUTemperatureRule(),
            {
                "gpu_temp_peak_c": 90.0,
                "per_gpu": _per_gpu(temp_peak=90.0),
            },
            "HIGH_GPU_TEMPERATURE",
        ),
        (
            HighGPUPowerRule(),
            {"per_gpu": _per_gpu(power_avg=90.0, power_limit=100.0)},
            "HIGH_GPU_POWER",
        ),
    ],
)
def test_system_rules_trigger_one_condition(
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
        (LowGPUUtilizationRule(), {"gpu_util_avg_percent": 60.0}),
        (HighCPURule(), {"cpu_avg_percent": 50.0}),
        (HighHostMemoryRule(), {"ram_peak_bytes": 500.0}),
        (HighGPUMemoryRule(), {"per_gpu": _per_gpu(mem_peak=500.0)}),
        (VeryHighGPUMemoryRule(), {"per_gpu": _per_gpu(mem_peak=850.0)}),
        (HighGPUTemperatureRule(), {"gpu_temp_peak_c": 70.0}),
        (
            HighGPUPowerRule(),
            {"per_gpu": _per_gpu(power_avg=50.0, power_limit=100.0)},
        ),
    ],
)
def test_system_rules_do_not_trigger_normal_condition(rule, overrides) -> None:
    assert rule.evaluate(_signals(**overrides)) is None


@pytest.mark.parametrize(
    ("overrides", "expected_primary"),
    [
        ({"per_gpu": _per_gpu(mem_peak=950.0)}, "VERY_HIGH_GPU_MEMORY"),
        (
            {
                "gpu_temp_peak_c": 90.0,
                "per_gpu": _per_gpu(mem_peak=850.0, temp_peak=90.0),
            },
            "HIGH_GPU_TEMPERATURE",
        ),
        ({"per_gpu": _per_gpu(mem_peak=850.0)}, "HIGH_GPU_MEMORY"),
        (
            {"per_gpu": _per_gpu(power_avg=90.0, power_limit=100.0)},
            "HIGH_GPU_POWER",
        ),
        ({"ram_peak_bytes": 930.0}, "HIGH_HOST_MEMORY"),
        ({"cpu_avg_percent": 95.0}, "HIGH_CPU"),
        (
            {
                "gpu_util_avg_percent": 10.0,
                "per_gpu": _per_gpu(util=10.0),
            },
            "LOW_GPU_UTILIZATION",
        ),
    ],
)
def test_system_primary_diagnosis_for_each_issue(
    overrides,
    expected_primary,
) -> None:
    result = _diagnose(**overrides)

    assert result.primary.kind == expected_primary


def test_system_primary_priority_when_everything_triggers() -> None:
    result = _diagnose(
        cpu_avg_percent=95.0,
        ram_peak_bytes=930.0,
        gpu_util_avg_percent=10.0,
        gpu_temp_peak_c=90.0,
        per_gpu=_per_gpu(
            util=10.0,
            mem_peak=950.0,
            temp_peak=90.0,
            power_avg=90.0,
            power_limit=100.0,
        ),
    )

    assert result.primary.kind == "VERY_HIGH_GPU_MEMORY"
    assert [issue.kind for issue in result.issues] == [
        "VERY_HIGH_GPU_MEMORY",
        "HIGH_GPU_TEMPERATURE",
        "HIGH_GPU_POWER",
        "HIGH_HOST_MEMORY",
        "HIGH_CPU",
        "LOW_GPU_UTILIZATION",
    ]


def test_system_cpu_only_normal_does_not_emit_no_gpu() -> None:
    result = _diagnose(
        gpu_available=False,
        gpu_count=0,
        gpu_util_avg_percent=None,
        gpu_util_peak_percent=None,
        gpu_mem_avg_bytes=None,
        gpu_mem_peak_bytes=None,
        per_gpu={},
    )

    assert result.primary.kind == "NORMAL"
    assert result.primary.status == "NORMAL"
    assert "GPU" not in result.primary.reason
    assert result.issues == ()


def test_system_no_data_is_primary_without_running_rules() -> None:
    result = _diagnose(system_samples=0, per_gpu={})

    assert result.primary.kind == "NO_DATA"
    assert result.issues == ()


def _diagnose(**overrides):
    if "system_samples" in overrides:
        overrides["samples"] = overrides.pop("system_samples")
    return diagnose_system(_input(**overrides))
