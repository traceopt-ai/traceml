# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Adapters that run summary Step Time diagnosis on canonical windows."""

from dataclasses import dataclass

from traceml_ai.diagnostics.common import DiagnosticResult
from traceml_ai.diagnostics.step_time.api import StepDiagnosis
from traceml_ai.diagnostics.step_time.policy import (
    SUMMARY_STEP_TIME_POLICY,
    StepTimeDiagnosisPolicy,
)
from traceml_ai.utils.step_time_window import (
    StepTimeWindow,
    diagnose_step_time_window,
)

DEFAULT_SUMMARY_DIAG_CONFIG = SUMMARY_STEP_TIME_POLICY


@dataclass(frozen=True)
class StepTimeDiagnosisInput:
    """Canonical Step Time window consumed by final-summary diagnosis."""

    window: StepTimeWindow
    policy: StepTimeDiagnosisPolicy = DEFAULT_SUMMARY_DIAG_CONFIG


def diagnose_step_time_summary(
    data: StepTimeDiagnosisInput,
) -> DiagnosticResult[StepDiagnosis]:
    """Run final-summary Step Time diagnosis from the canonical window."""
    return diagnose_step_time_window(data.window, policy=data.policy)


__all__ = [
    "DEFAULT_SUMMARY_DIAG_CONFIG",
    "StepTimeDiagnosisInput",
    "diagnose_step_time_summary",
]
