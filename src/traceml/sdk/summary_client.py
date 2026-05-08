"""
Client-side helper for requesting and reading a finalized TraceML summary.

This module is used by the public `traceml.final_summary()` API. It does not
generate summaries locally. Instead, it requests a fresh summary from the
aggregator process and reads the published summary artifact.
"""

import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

from traceml.sdk.protocol import (
    build_final_summary_request,
    get_final_summary_request_path,
    get_final_summary_response_path,
    is_primary_rank,
    load_json_or_none,
    request_to_json,
    resolve_session_context_from_env,
    write_json_atomic,
)


def _read_text_or_empty(path: Path) -> str:
    """
    Read a text file if possible, otherwise return an empty string.
    """
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return ""


def final_summary(
    *,
    timeout_sec: float = 30.0,
    poll_interval_sec: float = 0.1,
    print_text: bool = False,
    rank0_only: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Request and return a finalized TraceML summary for the current session.

    Parameters
    ----------
    timeout_sec:
        Maximum time to wait for the aggregator to process the request.
    poll_interval_sec:
        Poll interval while waiting for the response.
    print_text:
        If True, print the final summary text artifact after it is generated.
    rank0_only:
        If True, return ``None`` on non-zero ranks.

    Returns
    -------
    Optional[Dict[str, Any]]
        Parsed final summary JSON, or ``None`` when the summary is not
        available (non-zero rank, history disabled, aggregator error, or
        timeout).  A :mod:`warnings` message is emitted for all soft-failure
        cases so the caller can inspect what happened without crashing the
        training run.
    """
    if rank0_only and not is_primary_rank():
        return None

    try:
        ctx = resolve_session_context_from_env()
    except RuntimeError as exc:
        warnings.warn(
            f"[TraceML] final_summary: session context unavailable — {exc}",
            stacklevel=2,
        )
        return None

    if not ctx.history_enabled:
        warnings.warn(
            "[TraceML] final_summary: history is disabled; returning None.",
            stacklevel=2,
        )
        return None

    request = build_final_summary_request()
    request_path = get_final_summary_request_path(ctx.session_root)
    response_path = get_final_summary_response_path(ctx.session_root)

    write_json_atomic(request_path, request_to_json(request))

    deadline = time.time() + float(timeout_sec)
    while time.time() < deadline:
        response = load_json_or_none(response_path)
        if response and response.get("request_id") == request.request_id:
            status = str(response.get("status", "")).strip().lower()
            if status != "ok":
                error = (
                    response.get("error")
                    or "Aggregator failed to generate final summary."
                )
                warnings.warn(
                    f"[TraceML] final_summary: {error}",
                    stacklevel=2,
                )
                return None

            summary_json_path = response.get("summary_json_path")
            if not summary_json_path:
                warnings.warn(
                    "[TraceML] final_summary: aggregator response missing "
                    "summary_json_path.",
                    stacklevel=2,
                )
                return None

            summary = load_json_or_none(Path(summary_json_path))
            if summary is None:
                warnings.warn(
                    f"[TraceML] final_summary: could not read summary JSON "
                    f"from {summary_json_path}.",
                    stacklevel=2,
                )
                return None

            if print_text:
                summary_txt_path = response.get("summary_txt_path")
                if summary_txt_path:
                    text = _read_text_or_empty(Path(summary_txt_path))
                    if text:
                        print(text)

            return summary

        time.sleep(float(poll_interval_sec))

    warnings.warn(
        "[TraceML] final_summary: timed out waiting for aggregator response; "
        "returning None.",
        stacklevel=2,
    )
    return None
