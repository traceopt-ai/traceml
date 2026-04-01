"""
Central diagnostics package for shared diagnosis contracts and engines.
"""

from .common import BaseDiagnosis, Severity, diagnosis_to_dict

__all__ = [
    "Severity",
    "BaseDiagnosis",
    "diagnosis_to_dict",
]
