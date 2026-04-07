"""Public API for static ML training-script analysis.

This package provides AST-based inspection of Python training
scripts without importing user code, importing torch, or executing anything.

It is designed for observability and recommendation systems where partial
signal is still valuable. Public entry points never raise on ordinary
user-script failures; instead they return structured findings with
``parse_errors`` populated when needed.

Typical usage:
    from traceml.utils.ast_analysis import analyze_script, build_code_manifest

    findings = analyze_script("train.py")
    manifest = build_code_manifest(findings)

Stability
---------
The following are considered public and stable:
- analyze_script
- detect_strategy_hint
- scan_for_optimizer
- build_code_manifest
- public result dataclasses in scanner.py

Private helpers and internal detection logic may change without notice.
"""

from traceml.utils.ast_analysis.code_manifest import build_code_manifest
from traceml.utils.ast_analysis.models import (
    CodeFindings,
    DataLoaderFinding,
    DistributedFinding,
    FineTuningFinding,
    HFTrainingArgumentsFinding,
    ModelFinding,
    OptimizerFinding,
    PrecisionFinding,
    ScriptLocation,
)
from traceml.utils.ast_analysis.scanner import (
    analyze_script,
    detect_strategy_hint,
    scan_for_optimizer,
)

__all__ = [
    # High-level analysis
    "analyze_script",
    "detect_strategy_hint",
    "scan_for_optimizer",
    "build_code_manifest",
    # Data classes
    "CodeFindings",
    "DataLoaderFinding",
    "DistributedFinding",
    "FineTuningFinding",
    "HFTrainingArgumentsFinding",
    "ModelFinding",
    "OptimizerFinding",
    "PrecisionFinding",
    "ScriptLocation",
]
