# Copyright 2026 OptAI UG (haftungsbeschraenkt)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import traceml_ai.samplers.system_sampler as system_module
from traceml_ai.samplers.process_sampler import ProcessSampler
from traceml_ai.samplers.step_memory_sampler import StepMemorySampler
from traceml_ai.samplers.system_sampler import SystemSampler


class _Logger:
    def error(self, *args, **kwargs) -> None:
        return None


class _FailingProcess:
    def cpu_percent(self, interval=None):
        raise RuntimeError("cpu unavailable")

    def memory_info(self):
        raise RuntimeError("memory unavailable")


def test_step_memory_keeps_step_end_timestamp_from_event() -> None:
    sampler = StepMemorySampler.__new__(StepMemorySampler)
    sampler.sample_idx = 3
    sample = sampler._event_to_sample(
        SimpleNamespace(
            timestamp=12.25,
            model_id=1,
            device="cuda:0",
            step=7,
            peak_allocated=100.0,
            peak_reserved=200.0,
        )
    )

    assert sample is not None
    assert sample.timestamp == 12.25


def test_process_collection_failure_is_unavailable_not_zero() -> None:
    sampler = ProcessSampler.__new__(ProcessSampler)
    sampler.process = _FailingProcess()
    sampler.logger = _Logger()

    assert sampler._sample_cpu() is None
    assert sampler._sample_ram() is None


def test_system_collection_failure_is_unavailable_not_zero(
    monkeypatch,
) -> None:
    sampler = SystemSampler.__new__(SystemSampler)
    sampler.logger = _Logger()
    sampler.gpu_available = True
    sampler.gpu_count = 1

    monkeypatch.setattr(
        system_module.psutil,
        "cpu_percent",
        lambda interval=None: (_ for _ in ()).throw(RuntimeError("cpu")),
    )
    monkeypatch.setattr(
        system_module.psutil,
        "virtual_memory",
        lambda: (_ for _ in ()).throw(RuntimeError("memory")),
    )
    monkeypatch.setattr(
        system_module,
        "nvmlDeviceGetHandleByIndex",
        lambda index: (_ for _ in ()).throw(RuntimeError("gpu")),
    )

    assert sampler._sample_cpu() is None
    assert sampler._sample_ram() is None
    assert sampler._sample_gpus()[0].to_wire() == [
        None,
        None,
        None,
        None,
        None,
        None,
    ]
