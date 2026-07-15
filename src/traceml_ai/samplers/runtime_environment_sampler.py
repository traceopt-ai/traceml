# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Sampler bridge for one-shot runtime environment body rows."""

from __future__ import annotations

from traceml_ai.runtime.environment_state import pop_runtime_environment_record
from traceml_ai.samplers.base_sampler import BaseSampler


class RuntimeEnvironmentSampler(BaseSampler):
    """Emit rank-scoped body rows queued by trace_step(), not TCP metadata."""

    def __init__(self) -> None:
        super().__init__(
            sampler_name="RuntimeEnvironmentSampler",
            table_name="RuntimeEnvironmentTable",
        )

    def sample(self) -> None:
        """Pop one pending runtime environment row into the sampler database."""
        try:
            row = pop_runtime_environment_record()
            if row is not None:
                self._add_record(row)
        except Exception as exc:
            self.logger.error(
                f"[TraceML] RuntimeEnvironmentSampler error: {exc}"
            )
