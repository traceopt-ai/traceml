# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Final-report section contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

from traceml.core.summaries import SummaryResult, SummarySection

LoadedDataT = TypeVar("LoadedDataT")
DiagnosisInputT = TypeVar("DiagnosisInputT")
DiagnosisResultT = TypeVar("DiagnosisResultT")


class BaseSummarySection(
    ABC,
    Generic[LoadedDataT, DiagnosisInputT, DiagnosisResultT],
):
    """
    Shared lifecycle for final-report sections.

    Concrete sections keep their own data types, but follow the same readable
    flow: load persisted telemetry, adapt it for diagnosis, diagnose it, then
    assemble the final section result.
    """

    name: ClassVar[str]

    def build(self, db_path: str) -> SummaryResult:
        """Build the section for a telemetry database."""
        data = self.load(db_path)
        diagnosis_input = self.to_diagnosis_input(data)
        diagnosis_result = self.diagnose(diagnosis_input)
        return self.build_payload(data, diagnosis_result)

    @abstractmethod
    def load(self, db_path: str) -> LoadedDataT:
        """Load the bounded telemetry window needed by this section."""

    @abstractmethod
    def to_diagnosis_input(
        self,
        data: LoadedDataT,
    ) -> DiagnosisInputT:
        """Convert loaded telemetry into the diagnosis contract."""

    @abstractmethod
    def diagnose(
        self,
        diagnosis_input: DiagnosisInputT,
    ) -> DiagnosisResultT:
        """Run section-specific diagnosis."""

    @abstractmethod
    def build_payload(
        self,
        data: LoadedDataT,
        diagnosis_result: DiagnosisResultT,
    ) -> SummaryResult:
        """Assemble the structured payload and human-readable card."""


__all__ = [
    "BaseSummarySection",
    "SummaryResult",
    "SummarySection",
]
