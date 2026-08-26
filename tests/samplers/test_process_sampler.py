# Copyright 2026 OptAI UG (haftungsbeschraenkt)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from traceml_ai.samplers.process_sampler import ProcessSampler


class _Logger:
    def error(self, *args, **kwargs) -> None:
        return None


class _FailingProcess:
    def cpu_percent(self, interval=None):
        raise RuntimeError("cpu unavailable")

    def memory_info(self):
        raise RuntimeError("memory unavailable")


def test_collection_failure_is_unavailable_not_zero() -> None:
    sampler = ProcessSampler.__new__(ProcessSampler)
    sampler.process = _FailingProcess()
    sampler.logger = _Logger()

    assert sampler._sample_cpu() is None
    assert sampler._sample_ram() is None
