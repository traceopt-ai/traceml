"""traceml.utils.ast_analysis — Public API.

Provides static analysis of Python training scripts without executing them.

Usage:
    from traceml.utils.ast_analysis import analyze_script, scan_for_optimizer

    findings = analyze_script("train.py")
    optimizer = scan_for_optimizer("train.py")
"""

from traceml.utils.ast_analysis.scanner import (
    CodeFindings,
    DataLoaderFinding,
    DistributedFinding,
    FineTuningFinding,
    ModelFinding,
    OptimizerFinding,
    PrecisionFinding,
    ScriptLocation,
    analyze_script,
    detect_strategy_hint,
    scan_for_optimizer,
)

__all__ = [
    # High-level analysis
    "analyze_script",
    "detect_strategy_hint",
    "scan_for_optimizer",
    # Data classes
    "CodeFindings",
    "DataLoaderFinding",
    "DistributedFinding",
    "FineTuningFinding",
    "ModelFinding",
    "OptimizerFinding",
    "PrecisionFinding",
    "ScriptLocation",
]
