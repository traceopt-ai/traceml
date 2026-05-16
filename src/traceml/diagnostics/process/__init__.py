# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Public process diagnostics package.

The current process diagnostics are summary-oriented and conservative. They are
intended primarily for end-of-run interpretation and machine-readable summary
payloads rather than live runtime UI changes.
"""

from .api import ProcessDiagnosis, diagnose_process
from .policy import DEFAULT_PROCESS_POLICY, ProcessDiagnosisPolicy

__all__ = [
    "DEFAULT_PROCESS_POLICY",
    "ProcessDiagnosisPolicy",
    "ProcessDiagnosis",
    "diagnose_process",
]
