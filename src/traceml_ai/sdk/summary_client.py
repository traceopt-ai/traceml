"""
Client-side helper for requesting and reading a finalized TraceML summary.

This module backs the public `traceml.summary()` and `traceml.final_summary()`
APIs. It does not generate summaries locally. Instead, it requests a fresh
summary from the aggregator process and reads the published summary artifact.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

from traceml_ai.sdk.protocol import (
    build_final_summary_request,
    get_final_summary_request_path,
    get_final_summary_response_path,
    is_primary_rank,
    load_json_or_none,
    request_to_json,
    resolve_session_context_from_env,
)
from traceml_ai.sdk.summary_projection import compact_summary
from traceml_ai.utils.atomic_io import write_json_atomic


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
    Request and return the full final_summary.json payload for this session.

    Most users should call ``traceml.summary()`` for compact W&B/MLflow-friendly
    scalar output. Use this function when you need the full structured report.

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
        Parsed final summary JSON, or ``None`` on non-zero ranks when
        `rank0_only=True`.

    Raises
    ------
    RuntimeError
        If TraceML session context is unavailable, history is disabled, the
        aggregator reports an error, or the request times out.
    """
    if rank0_only and not is_primary_rank():
        return None

    ctx = resolve_session_context_from_env()
    if not ctx.history_enabled:
        raise RuntimeError(
            "TraceML final_summary() requires history to be enabled."
        )

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
                raise RuntimeError(
                    response.get("error")
                    or "Aggregator failed to generate final summary."
                )

            summary_json_path = response.get("summary_json_path")
            if not summary_json_path:
                raise RuntimeError(
                    "Aggregator response did not include summary_json_path."
                )

            summary = load_json_or_none(Path(summary_json_path))
            if summary is None:
                raise RuntimeError(
                    f"Final summary JSON could not be read: {summary_json_path}"
                )

            if print_text:
                summary_txt_path = response.get("summary_txt_path")
                if summary_txt_path:
                    text = _read_text_or_empty(Path(summary_txt_path))
                    if text:
                        print(text)

            return summary

        time.sleep(float(poll_interval_sec))

    raise RuntimeError(
        "Timed out waiting for TraceML final summary from the aggregator."
    )


def summary(
    *,
    timeout_sec: float = 30.0,
    poll_interval_sec: float = 0.1,
    print_text: bool = False,
    rank0_only: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Return a compact tracker-friendly TraceML summary for the current session.

    The returned dict is flat and intended for experiment trackers such as W&B,
    MLflow, and internal dashboards. Use ``final_summary()`` when you need the
    complete structured JSON artifact.
    """
    payload = final_summary(
        timeout_sec=timeout_sec,
        poll_interval_sec=poll_interval_sec,
        print_text=print_text,
        rank0_only=rank0_only,
    )
    return compact_summary(payload)
