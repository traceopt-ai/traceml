"""
Shared protocol helpers for on-demand final summary generation.

This module defines the small file-based contract used between:

- training worker code calling `traceml.final_summary()`
- the TraceML aggregator process that owns history and summary generation

Design
---------------
The aggregator already runs out-of-process and owns the SQLite history. For v1,
a file-based request/response protocol is the simplest robust cross-process
contract:
- easy to debug
- easy to inspect manually
- works in local and production environments
- avoids introducing RPC complexity too early
"""

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TraceMLSessionContext:
    """
    TraceML session context resolved from environment variables.
    """

    session_id: str
    logs_dir: str
    session_root: Path
    history_enabled: bool
    mode: str


@dataclass(frozen=True)
class FinalSummaryRequest:
    """
    Request payload written by user code to ask the aggregator for a fresh
    finalized summary.
    """

    request_id: str
    created_at: str
    pid: int
    rank: int
    local_rank: int


@dataclass(frozen=True)
class FinalSummaryResponse:
    """
    Response payload written by the aggregator after processing a summary
    request.
    """

    request_id: str
    status: str
    completed_at: str
    summary_json_path: Optional[str] = None
    summary_txt_path: Optional[str] = None
    error: Optional[str] = None


def utc_now_iso() -> str:
    """
    Return the current UTC timestamp as ISO-8601 text.
    """
    return datetime.now(timezone.utc).isoformat()


def make_request_id() -> str:
    """
    Create a unique request id for one final-summary request.
    """
    return f"final_summary_{uuid.uuid4().hex}"


def is_primary_rank() -> bool:
    """
    Return True when the current process is rank 0 or rank is unset.
    """
    rank = os.environ.get("RANK", "").strip()
    if not rank:
        return True
    try:
        return int(rank) == 0
    except Exception:
        return False


def resolve_session_context_from_env() -> TraceMLSessionContext:
    """
    Resolve the active TraceML session context from environment variables.

    Raises
    ------
    RuntimeError
        If the current process is not running under a TraceML-launched session.
    """
    session_id = os.environ.get("TRACEML_SESSION_ID", "").strip()
    logs_dir = os.environ.get("TRACEML_LOGS_DIR", "").strip()
    mode = (
        os.environ.get("TRACEML_UI_MODE", "").strip()
        or os.environ.get("TRACEML_MODE", "cli").strip()
    )
    history_enabled = (
        os.environ.get("TRACEML_HISTORY_ENABLED", "1").strip() == "1"
    )

    if not session_id or not logs_dir:
        raise RuntimeError(
            "TraceML session context is not available. "
            "Make sure the script was launched with `traceml run ...`."
        )

    session_root = Path(logs_dir).resolve() / session_id
    return TraceMLSessionContext(
        session_id=session_id,
        logs_dir=logs_dir,
        session_root=session_root,
        history_enabled=history_enabled,
        mode=mode,
    )


def get_control_dir(session_root: Path) -> Path:
    """
    Return the control directory used for summary request/response files.
    """
    return Path(session_root).resolve() / "control"


def get_final_summary_request_path(session_root: Path) -> Path:
    """
    Return the path to the final-summary request file.
    """
    return get_control_dir(session_root) / "final_summary_request.json"


def get_final_summary_response_path(session_root: Path) -> Path:
    """
    Return the path to the final-summary response file.
    """
    return get_control_dir(session_root) / "final_summary_response.json"


def get_final_summary_json_path(session_root: Path) -> Path:
    """
    Return the canonical final summary JSON artifact path.
    """
    return Path(session_root).resolve() / "final_summary.json"


def get_final_summary_txt_path(session_root: Path) -> Path:
    """
    Return the canonical final summary text artifact path.
    """
    return Path(session_root).resolve() / "final_summary.txt"


def load_json_or_none(path: Path) -> Optional[Dict[str, Any]]:
    """
    Load JSON from disk if possible, otherwise return None.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """
    Write JSON atomically to avoid partial files being observed by another
    process.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)

    os.replace(tmp_path, path)


def write_text_atomic(path: Path, text: str) -> None:
    """
    Write text atomically to avoid partial files being observed by another
    process.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)

    os.replace(tmp_path, path)


def build_final_summary_request() -> FinalSummaryRequest:
    """
    Build a final-summary request payload from the current process context.
    """
    rank = 0
    local_rank = 0

    try:
        rank = int(os.environ.get("RANK", "0"))
    except Exception:
        rank = 0

    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except Exception:
        local_rank = 0

    return FinalSummaryRequest(
        request_id=make_request_id(),
        created_at=utc_now_iso(),
        pid=os.getpid(),
        rank=rank,
        local_rank=local_rank,
    )


def request_to_json(request: FinalSummaryRequest) -> Dict[str, Any]:
    """
    Serialize a final-summary request to JSON.
    """
    return asdict(request)


def response_to_json(response: FinalSummaryResponse) -> Dict[str, Any]:
    """
    Serialize a final-summary response to JSON.
    """
    return asdict(response)
