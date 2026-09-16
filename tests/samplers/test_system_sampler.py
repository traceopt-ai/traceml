# Copyright 2026 OptAI UG (haftungsbeschraenkt)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import traceml_ai.samplers.system_sampler as system_module
from traceml_ai.samplers.schema.system import GPUMetrics, SystemSample
from traceml_ai.samplers.system_sampler import SystemSampler


class _Logger:
    def error(self, *args, **kwargs) -> None:
        return None


def _raise_unavailable(*args, **kwargs):
    raise RuntimeError("unavailable")


def test_collection_failures_are_unavailable_not_zero(monkeypatch) -> None:
    sampler = SystemSampler.__new__(SystemSampler)
    sampler.logger = _Logger()
    sampler.gpu_available = True
    sampler.gpu_count = 1

    monkeypatch.setattr(
        system_module.psutil,
        "cpu_percent",
        _raise_unavailable,
    )
    monkeypatch.setattr(
        system_module.psutil,
        "virtual_memory",
        _raise_unavailable,
    )
    monkeypatch.setattr(
        system_module,
        "nvmlDeviceGetHandleByIndex",
        _raise_unavailable,
    )

    sampler._init_ram()

    assert sampler.ram_total_memory is None
    assert sampler._sample_cpu() is None
    assert sampler._sample_ram() is None
    assert sampler._sample_gpus()[0].to_wire() == [None] * 6


def test_nullable_system_sample_round_trips_through_wire_schema() -> None:
    sample = SystemSample(
        sample_idx=1,
        timestamp=2.0,
        cpu_percent=None,
        ram_used=None,
        ram_total=None,
        gpu_available=True,
        gpu_count=1,
        gpus=[GPUMetrics(None, None, None, None, None, None)],
    )

    assert SystemSample.from_wire(sample.to_wire()) == sample
