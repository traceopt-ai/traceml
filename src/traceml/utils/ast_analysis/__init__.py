"""traceml.utils.ast_analysis — Public API.

Provides static analysis of Python training scripts without executing them.

Usage:
    from traceml.utils.ast_analysis import analyze_script, estimate_params, scan_for_optimizer

    findings = analyze_script("train.py")
    estimate = estimate_params("train.py", findings)
    optimizer = scan_for_optimizer("train.py")
"""

from traceml.utils.ast_analysis.model_registry import (
    ModelSpec,
    list_known_models,
    lookup_model,
)
from traceml.utils.ast_analysis.param_estimator import (
    ParamEstimate,
    estimate_params,
)
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
    "estimate_params",
    # Data classes
    "CodeFindings",
    "DataLoaderFinding",
    "DistributedFinding",
    "FineTuningFinding",
    "ModelFinding",
    "OptimizerFinding",
    "PrecisionFinding",
    "ParamEstimate",
    "ModelSpec",
    "ScriptLocation",
    # Registry helpers
    "lookup_model",
    "list_known_models",
]
