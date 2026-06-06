"""
On-demand final summary service for the TraceML aggregator.

This service watches for a file-based final-summary request, flushes persisted
history, generates a fresh final summary, writes canonical summary artifacts,
and publishes a small response file for the requesting process to read.
"""

from pathlib import Path
from typing import Any, Callable, Optional

from traceml_ai.reporting.config import DEFAULT_SUMMARY_WINDOW_ROWS
from traceml_ai.reporting.final import generate_summary
from traceml_ai.sdk.protocol import (
    FinalSummaryResponse,
    get_final_summary_json_path,
    get_final_summary_request_path,
    get_final_summary_response_path,
    get_final_summary_txt_path,
    load_json_or_none,
    response_to_json,
    utc_now_iso,
)
from traceml_ai.utils.atomic_io import write_json_atomic


class FinalSummaryService:
    """
    Aggregator-side service that handles on-demand final-summary requests.
    """

    def __init__(
        self,
        logger: Any,
        session_root: Path,
        db_path: str,
        flush_history: Callable[[float], bool],
        settle_telemetry: Optional[Callable[[float], bool]] = None,
        summary_window_rows: int = DEFAULT_SUMMARY_WINDOW_ROWS,
    ) -> None:
        self._logger = logger
        self._session_root = Path(session_root).resolve()
        self._db_path = str(db_path)
        self._flush_history = flush_history
        self._settle_telemetry = settle_telemetry
        self._summary_window_rows = int(summary_window_rows)
        self._last_request_id: Optional[str] = None

    def poll(self) -> None:
        """
        Check for a new final-summary request and handle it once.
        """
        request_path = get_final_summary_request_path(self._session_root)
        request = load_json_or_none(request_path)
        if not request:
            return

        request_id = str(request.get("request_id", "")).strip()
        if not request_id or request_id == self._last_request_id:
            return

        self._handle_request(request_id=request_id)

    def _handle_request(self, *, request_id: str) -> None:
        """
        Flush history, generate a fresh summary, and publish the response.
        """
        response_path = get_final_summary_response_path(self._session_root)

        try:
            if self._summary_exists():
                self._write_ok_response(response_path, request_id=request_id)
                self._last_request_id = request_id
                return

            settled = self._settle_before_generation(5.0)
            if not settled:
                raise RuntimeError(
                    "Timed out while waiting for TraceML telemetry to settle."
                )

            generate_summary(
                self._db_path,
                session_root=str(self._session_root),
                print_to_stdout=False,
                summary_window_rows=self._summary_window_rows,
            )

            self._write_ok_response(response_path, request_id=request_id)
            self._last_request_id = request_id

        except Exception as exc:
            try:
                self._logger.exception(
                    "[TraceML] Final summary request failed"
                )
            except Exception:
                pass

            response = FinalSummaryResponse(
                request_id=request_id,
                status="error",
                completed_at=utc_now_iso(),
                summary_json_path=None,
                summary_txt_path=None,
                error=str(exc),
            )
            write_json_atomic(response_path, response_to_json(response))
            self._last_request_id = request_id

    def _summary_exists(self) -> bool:
        """Return True when the canonical JSON artifact already exists."""
        return get_final_summary_json_path(self._session_root).is_file()

    def _settle_before_generation(self, timeout_sec: float) -> bool:
        """Best-effort telemetry settle before first summary generation."""
        if self._settle_telemetry is not None:
            return bool(self._settle_telemetry(float(timeout_sec)))
        return bool(self._flush_history(float(timeout_sec)))

    def _write_ok_response(
        self,
        response_path: Path,
        *,
        request_id: str,
    ) -> None:
        """Publish an OK response pointing at canonical summary artifacts."""
        response = FinalSummaryResponse(
            request_id=request_id,
            status="ok",
            completed_at=utc_now_iso(),
            summary_json_path=str(
                get_final_summary_json_path(self._session_root)
            ),
            summary_txt_path=str(
                get_final_summary_txt_path(self._session_root)
            ),
            error=None,
        )
        write_json_atomic(response_path, response_to_json(response))
