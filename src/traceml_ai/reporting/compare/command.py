"""
Top-level compare command implementation for TraceML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from traceml_ai.reporting.compare.core import build_compare_payload
from traceml_ai.reporting.compare.io import (
    default_output_base,
    load_summary_json,
    write_compare_artifacts,
)
from traceml_ai.reporting.compare.policy import build_step_time_ci_policy
from traceml_ai.reporting.compare.render import build_compare_text


def compare_summaries(
    lhs_path: str | Path,
    rhs_path: str | Path,
    *,
    output: Optional[str | Path] = None,
    max_step_time_regression_pct: Optional[str] = None,
    print_to_stdout: bool = True,
) -> Dict[str, Any]:
    """
    Compare two TraceML final summary JSON files.

    Parameters
    ----------
    lhs_path:
        Path to the left-hand run summary JSON.
    rhs_path:
        Path to the right-hand run summary JSON.
    output:
        Optional output base path. If omitted, defaults to
        `compare/<lhs>_vs_<rhs>` in the current working directory.
    max_step_time_regression_pct:
        Raw optional CI policy threshold. Policy validation and evaluation
        occur after the compare payload is built. ``None`` keeps the existing
        exploratory comparison behavior.
    print_to_stdout:
        If True, print the rendered compare text after writing artifacts.

    Returns
    -------
    dict
        Structured compare payload.

    Artifacts written
    -----------------
    - `<output_base>.json`
    - `<output_base>.txt`
    """
    lhs_payload = load_summary_json(lhs_path)
    rhs_payload = load_summary_json(rhs_path)

    compare_payload = build_compare_payload(
        lhs_payload=lhs_payload,
        rhs_payload=rhs_payload,
        lhs_path=lhs_path,
        rhs_path=rhs_path,
    )
    if max_step_time_regression_pct is not None:
        compare_payload["ci_policy"] = build_step_time_ci_policy(
            lhs_payload=lhs_payload,
            rhs_payload=rhs_payload,
            compare_payload=compare_payload,
            threshold=max_step_time_regression_pct,
        )
    compare_payload["text"] = build_compare_text(compare_payload)

    output_base = default_output_base(lhs_path, rhs_path, output=output)
    json_path, txt_path = write_compare_artifacts(
        output_base=output_base,
        payload=compare_payload,
    )

    compare_payload.setdefault("artifacts", {})
    compare_payload["artifacts"]["base"] = str(output_base)
    compare_payload["artifacts"]["json"] = str(json_path)
    compare_payload["artifacts"]["txt"] = str(txt_path)

    if print_to_stdout:
        print(compare_payload["text"])
        print(f"[TraceML] Compare JSON: {json_path}")
        print(f"[TraceML] Compare TXT:  {txt_path}")

    return compare_payload
