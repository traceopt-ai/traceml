# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Public system diagnostics package.

The current system diagnostics are summary-oriented and conservative. They are
intended primarily for end-of-run interpretation and machine-readable summary
payloads rather than live runtime UI changes.
"""

from .api import SystemDiagnosis, diagnose_system
from .policy import (
    DEFAULT_SYSTEM_POLICY,
    GPUUtilizationBands,
    SystemDiagnosisPolicy,
)

__all__ = [
    "DEFAULT_SYSTEM_POLICY",
    "GPUUtilizationBands",
    "SystemDiagnosisPolicy",
    "SystemDiagnosis",
    "diagnose_system",
]
